from shiny import App, render, ui, reactive
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon, Patch
from matplotlib.collections import PatchCollection
import numpy as np
import os
import tempfile

# Set environment variable to avoid memory leak issues with multithreading
os.environ['OMP_NUM_THREADS'] = '1'

# Load data
df_migration = pd.read_excel("/Users/shenzekai/Documents/GitHub/30538_final_project/avg_migration_18-23.xlsx")
df_cooperation = pd.read_excel("/Users/shenzekai/Documents/GitHub/30538_final_project/citypair_output.xlsx", sheet_name="sum")
city_boundaries = gpd.read_file("/Users/shenzekai/Documents/GitHub/30538_final_project/长三角城市数据/长三角地市_1.shp").to_crs(epsg=3857)

# Prepare city center dictionary
city_boundaries['centroid'] = city_boundaries['geometry'].centroid
city_centers = city_boundaries.set_index('cityE')['centroid'].to_dict()

# Add geometry for migration lines
def create_migration_linestring(row):
    point1 = city_centers.get(row['city1'])
    point2 = city_centers.get(row['city2'])
    if point1 is None or point2 is None:
        return None
    return Line2D([point1.x, point2.x], [point1.y, point2.y])

df_migration['geometry'] = df_migration.apply(lambda row: create_migration_linestring(row), axis=1)

# Define a standard mapping for commonly accepted English translations of Chinese city names
city_name_mapping = {
    "上海": "Shanghai",
    "杭州": "Hangzhou",
    "宁波": "Ningbo",
    "温州": "Wenzhou",
    "嘉兴": "Jiaxing",
    "湖州": "Huzhou",
    "绍兴": "Shaoxing",
    "金华": "Jinhua",
    "台州": "TAIzhou",
    "舟山": "Zhoushan",
    "衢州": "Quzhou",
    "丽水": "Lishui",
    "南京": "Nanjing",
    "无锡": "Wuxi",
    "常州": "Changzhou",
    "苏州": "SUzhou",
    "南通": "Nantong",
    "扬州": "Yangzhou",
    "镇江": "Zhenjiang",
    "泰州": "Taizhou",
    "徐州": "Xuzhou",
    "盐城": "Yancheng",
    "淮安": "Huaian",
    "连云港": "Lianyungang",
    "宿迁": "Suqian",
    "芜湖": "Wuhu",
    "马鞍山": "Maanshan",
    "铜陵": "Tongling",
    "安庆": "Anqing",
    "黄山": "Huangshan",
    "滁州": "Chuzhou",
    "阜阳": "Fuyang",
    "宿州": "Suzhou",
    "六安": "Luan",
    "池州": "Chizhou",
    "宣城": "Xuancheng",
    "合肥": "Hefei",
    "淮北": "Huaibei",
    "蚌埠": "Bengbu",
    "亳州": "Bozhou",
    "滁州": "Chuzhou",
    "淮南": "Huainan",
}

# Apply the mapping to cityi and cityj columns
df_cooperation['cityi'] = df_cooperation['cityi'].map(city_name_mapping)
df_cooperation['cityj'] = df_cooperation['cityj'].map(city_name_mapping)
df_cooperation['sum'] = df_cooperation['meeting_count'] + df_cooperation['agreement_count'] + df_cooperation['project_count'] + df_cooperation['visit_count']

# Add geometry for cooperation lines
def create_cooperation_linestring(row):
    point1 = city_centers.get(row['cityi'])
    point2 = city_centers.get(row['cityj'])
    if point1 is None or point2 is None:
        return None
    return Line2D([point1.x, point2.x], [point1.y, point2.y])

df_cooperation['geometry'] = df_cooperation.apply(lambda row: create_cooperation_linestring(row), axis=1)

# Define get_color_and_width function for migration
def get_color_and_width(avg_migration_index):
    if avg_migration_index <= 300:
        return 'lightgrey', 0.3
    elif avg_migration_index <= 1000:
        return 'lightgreen', 1.5
    elif avg_migration_index <= 2500:
        return 'orange', 2.0
    else:
        return 'red', 3.0

# Define function to add north arrow
def add_north(ax, x, y, text_size, arrow_width, text_pad, arrow_height, line_width, add_circle):
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    width = x_max - x_min
    height = y_max - y_min

    left = (x_min + width * (x - arrow_width / 2), y_min + height * (y - arrow_height))
    right = (x_min + width * (x + arrow_width / 2), left[1])
    top = (x_min + width * x, y_min + height * y)

    left_patch = Polygon([left, top, (top[0], left[1])], closed=True, edgecolor='black', facecolor='black')
    ax.add_patch(left_patch)

    right_patch = Polygon([right, top, (top[0], left[1])], closed=True, edgecolor='black', facecolor='white')
    ax.add_patch(right_patch)

    if add_circle:
        x0, y0, r = get_circle(left[0], left[1], top[0], top[1], right[0], right[1])
        circle = Circle((x0, y0), r, edgecolor='black', facecolor='none', linewidth=line_width)
        ax.add_patch(circle)
        ax.text(x0, y0 + 1.2*r + text_pad, 'N', horizontalalignment='center', verticalalignment='center', 
                fontsize=text_size, family='Arial', zorder=5)

# Define function to calculate circle parameters
def get_circle(x1, y1, x2, y2, x3, y3):
    a = x1 - x2
    b = y1 - y2
    c = x1 - x3
    d = y1 - y3
    a1 = ((x1 ** 2 - x2 ** 2) + (y1 ** 2 - y2 ** 2)) / 2.0
    a2 = ((x1 ** 2 - x3 ** 2) + (y1 ** 2 - y3 ** 2)) / 2.0
    theta = b * c - a * d
    if abs(theta) < 1e-7:
        raise ValueError("Three collinear points do not define a circle.")
    x0 = (b * a2 - d * a1) / theta
    y0 = (c * a1 - a * a2) / theta
    r = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    return x0, y0, r

# UI definition
app_ui = ui.page_fluid(
    ui.h2("Inter-City Networks (2018–2023)"),
    ui.input_slider("year", "Select Year:", min=2018, max=2023, value=2018),
    ui.output_image("network_plot")
)

# Server logic
def server(input, output, session):
    filtered_migration_data = reactive.Value(df_migration[df_migration['year'] == 2018])
    filtered_cooperation_data = reactive.Value(df_cooperation[df_cooperation['year'] == 2018])
    
    @reactive.Effect
    def update_filtered_data():
        filtered_migration_data.set(df_migration[df_migration['year'] == input.year()])
        filtered_cooperation_data.set(df_cooperation[df_cooperation['year'] == input.year()])
    
    @output
    @render.image
    def network_plot():
        # Filter data for the selected year
        migration_data = filtered_migration_data.get()
        cooperation_data = filtered_cooperation_data.get()
        
        # Create the plot with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 18), dpi=100)
        
        # Plot migration data
        city_boundaries.plot(ax=ax1, color="none", edgecolor="black")
        for idx, row in city_boundaries.iterrows():
            ax1.text(row.geometry.centroid.x, row.geometry.centroid.y, row['cityE'], fontsize=12,
                     ha='center', va='center', color='black', zorder=6)
            ax1.plot(row.geometry.centroid.x, row.geometry.centroid.y, 'o', color='red', markersize=8, zorder=4)
        for _, row in migration_data.iterrows():
            line = row['geometry']
            if line is not None:
                color, width = get_color_and_width(row['avg_migration_index'])
                ax1.plot(line.get_xdata(), line.get_ydata(), color=color, linewidth=width)
        ax1.set_title(f"Migration Network - Year {input.year()}")
        ax1.axis("off")
        
        # Add legend for migration plot
        legend_elements_migration = [
            Line2D([0], [0], color='lightgrey', lw=0.3, label='0-300'),
            Line2D([0], [0], color='lightgreen', lw=1.5, label='300-1000'),
            Line2D([0], [0], color='orange', lw=2.0, label='1000-2500'),
            Line2D([0], [0], color='red', lw=2.5, label='2500-9000')
        ]
        ax1.legend(handles=legend_elements_migration, loc='lower left', fontsize=12, title="Migration Index")
        
        # Plot cooperation data
        city_boundaries.plot(ax=ax2, color="none", edgecolor="grey")
        city_names = city_boundaries[['cityE', 'geometry']]
        core_cities = ['Bengbu', 'Changzhou', 'Chuzhou', 'Hangzhou', 'Hefei', 'Huaian', 'Huainan', 'Huangshan', 'Huzhou', 'Jiaxing', 'Liuan', 'Maanshan', 'Nanjing', 'Nantong', 'Ningbo', 'Quzhou', 'Shanghai', 'Shaoxing', 'SUzhou', 'TAIzhou', 'Wuhu', 'Wuxi', 'Xuancheng', 'Yangzhou', 'Zhenjiang', 'Zhoushan']
        city_names['city_type'] = 'peripheral'
        city_names.loc[city_names['cityE'].isin(core_cities), 'city_type'] = 'core'
        color_map = {'core': 'white', 'peripheral': '#f5f5f5'}
        for idx, row in city_names.iterrows():
            city_boundary = city_boundaries[city_boundaries['cityE'] == row['cityE']]
            city_boundary.plot(ax=ax2, color=color_map[row['city_type']], edgecolor='grey')
            ax2.text(row.geometry.centroid.x, row.geometry.centroid.y, row['cityE'], fontsize=8, ha='center', va='center', color='black')
        year_data = cooperation_data
        for _, row in year_data.iterrows():
            line = row['geometry']
            if line is not None:
                ax2.plot(line.get_xdata(), line.get_ydata(), linewidth=row['sum']/65, color='#AFC2DC', zorder=2)
        cities_to_mark = ['Nanjing', 'Hefei', 'Shanghai', 'Hangzhou']
        patches = []
        for city in cities_to_mark:
            city_point = city_centers[city]
            circle = Circle((city_point.x, city_point.y), 5000, color='red', alpha=0.5)
            patches.append(circle)
        p = PatchCollection(patches, match_original=True, zorder=3)
        ax2.add_collection(p)
        scalebar_location = (0.11, 0.06)
        scale_length = 100000
        x_length = scale_length / (ax2.get_xlim()[1] - ax2.get_xlim()[0])
        ax2.plot([scalebar_location[0], scalebar_location[0] + x_length],
                 [scalebar_location[1], scalebar_location[1]],
                 color='black', transform=ax2.transAxes, lw=2)
        ax2.text(scalebar_location[0] + x_length / 2, scalebar_location[1] - 0.01, '100 km',
                 horizontalalignment='center', verticalalignment='top',
                 transform=ax2.transAxes, fontsize=12)
        add_north(ax2, x=0.88, y=0.95, text_size=10, arrow_width=0.02, 
                  text_pad=0.01, arrow_height=0.1, line_width=0.5, add_circle=True)
        legend_elements_cooperation = [
            Patch(facecolor='#f5f5f5', edgecolor='lightgrey', label='Periphery'),
            Patch(facecolor='none', edgecolor='k', label='Core'),
            Line2D([0], [0], color='#AFC2DC', lw=4, label='Cooperation Frequency'),
            Line2D([0], [0], marker='o', color='w', markersize=10, markerfacecolor='red', label='Provincial Capitals')
        ]
        leg = ax2.legend(handles=legend_elements_cooperation, title='LEGEND', loc='upper left', bbox_to_anchor=(0.092,0.22), fontsize='small', title_fontsize='medium')
        leg.get_frame().set_linewidth(0.0)
        ax2.set_title(f"Cooperation Network - Year {input.year()}")
        ax2.axis("off")
        
        # Save the plot to a temporary file
        tmpfile = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        fig.savefig(tmpfile.name)
        plt.close(fig)
        
        return {"src": tmpfile.name, "width": "100%", "height": "auto"}

# Run the app
app = App(app_ui, server)