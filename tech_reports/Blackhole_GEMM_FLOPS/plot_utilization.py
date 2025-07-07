# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

# Configuration options
square_matrices_only = False  # Changed to False to avoid filtering out all matches

# Read CSV files and add a 'source' column to each
df1 = pd.read_csv("tech_reports/GEMM_FLOPS/n150-8x8.csv")
df1["source"] = "n150"
df2 = pd.read_csv("tech_reports/GEMM_FLOPS/p150-12x10.csv")
df2["source"] = "p150"

# Combine dataframes
dataframes = [df1, df2]

# Concatenate all dataframes
df = pd.concat(dataframes, ignore_index=True)

# Create a new column for matrix size as a string
df["matrix_size"] = df.apply(lambda row: f"{row['m']}x{row['k']}x{row['n']}", axis=1)

# Filter for square matrices if option is enabled
if square_matrices_only:
    print(f"Before square filter: {len(df)} rows")
    # Square matrix means all three dimensions are equal
    df = df[(df["m"] == df["n"]) & (df["m"] == df["k"])]
    print(f"After square filter: {len(df)} rows")
    if len(df) == 0:
        raise ValueError("No square matrices found in the dataset!")

# Create a combined dtype+fidelity+source label for legend and color
df["legend_label"] = (
    df["dtype"].str.replace("DataType.", "", regex=False) + "_" +
    df["math_fidelity"].str.replace("MathFidelity.", "", regex=False) +
    " (" + df["source"] + ")"
)
df["dtype_source"] = df["legend_label"]  # for compatibility below

# Create a column without source for pairing
df["dtype_fidelity"] = (
    df["dtype"].str.replace("DataType.", "", regex=False) + "_" +
    df["math_fidelity"].str.replace("MathFidelity.", "", regex=False)
)

# Get unique values for dtype_fidelity and sources
unique_dtype_fidelity = df["dtype_fidelity"].unique()
unique_sources = df["source"].unique()

# Create a color map for each unique dtype_fidelity
base_colors = plt.cm.tab10.colors  # Use fewer base colors for more distinction
dtype_fidelity_colors = {df: base_colors[i % len(base_colors)] for i, df in enumerate(unique_dtype_fidelity)}

# Create a shade adjustment for each source
source_shade = {"n150": 0.7, "p150": 1.0}

# Get unique matrix sizes for consistent order
matrix_sizes = df["matrix_size"].unique()

# Filter for combinations available in both n150 and p150
n150_combinations = df[df["source"] == "n150"][["matrix_size", "dtype_fidelity"]].drop_duplicates()
p150_combinations = df[df["source"] == "p150"][["matrix_size", "dtype_fidelity"]].drop_duplicates()

# Merge to find common combinations
common_combinations = pd.merge(
    n150_combinations, p150_combinations, 
    on=["matrix_size", "dtype_fidelity"], 
    how="inner"
)

# Filter dataset to only include matching combinations
df = pd.merge(
    df, common_combinations, 
    on=["matrix_size", "dtype_fidelity"], 
    how="inner"
)

# If dataframe is empty after filtering, raise an error
if len(df) == 0:
    raise ValueError("No matching data between n150 and p150 after filtering! Check if matrices sizes and dtypes align between datasets.")

# Determine matrix elements (m×k×n) for sorting
df["total_elements"] = df["m"] * df["k"] * df["n"]

# Sort matrix sizes by total elements to ensure they're always increasing
matrix_sizes_sorted = []
matrix_elements_dict = {}

for matrix_size in matrix_sizes:
    # Get the total elements for this matrix size
    group = df[df["matrix_size"] == matrix_size]
    if not group.empty:
        total_elements = group["total_elements"].iloc[0]
        matrix_elements_dict[matrix_size] = total_elements
        
# Sort matrix sizes by their total elements
matrix_sizes_sorted = sorted(matrix_elements_dict.keys(), key=lambda x: matrix_elements_dict[x])

bar_width = 0.05  # Make bars extremely skinny
gap = 0.01  # Reduce gap between clusters even more
positions = []
heights = []
dtype_colors = []
cluster_centers = []
labels = []
current_pos = 0

for matrix_size in matrix_sizes_sorted:
    group = df[df["matrix_size"] == matrix_size]
    dtype_count = 0
    cluster_start = current_pos
    
    # Get unique dtype_fidelity combos for this matrix size
    combos = group["dtype_fidelity"].unique()
    
    # Sort combos based on utilization from n150 (or p150 if n150 doesn't exist)
    def get_sort_value(combo):
        n150_entry = group[(group["dtype_fidelity"] == combo) & (group["source"] == "n150")]
        if not n150_entry.empty:
            return n150_entry["Utilization (vs 8x8 full grid)"].values[0]
        p150_entry = group[(group["dtype_fidelity"] == combo) & (group["source"] == "p150")]
        if not p150_entry.empty:
            return p150_entry["Utilization (vs 8x8 full grid)"].values[0]
        return 0
    
    sorted_combos = sorted(combos, key=get_sort_value)
    
    # Process each dtype_fidelity combo, placing all sources side by side
    for combo in sorted_combos:
        # Process each source for this combo
        for source in unique_sources:
            source_entry = group[(group["dtype_fidelity"] == combo) & (group["source"] == source)]
            
            if not source_entry.empty:
                entry = source_entry.iloc[0]
                pos = current_pos
                # Use Utilization instead of TFLOPs
                height = entry["Utilization (vs 8x8 full grid)"]
                # Convert from percentage string to float if needed
                if isinstance(height, str) and '%' in height:
                    height = float(height.strip('%'))
                positions.append(pos)
                heights.append(height)
                
                # Get base color for this dtype_fidelity and adjust for source
                base_color = dtype_fidelity_colors[combo]
                adjusted_color = mcolors.to_rgba(base_color, alpha=source_shade[source])
                dtype_colors.append(adjusted_color)
                
                dtype_count += 1
                current_pos += bar_width
        
        # Add gap between different dtype_fidelity combos within the cluster
        current_pos += 0.02  # Reduce spacing to minimum
            
    if dtype_count > 0:
        # Center label under the cluster
        cluster_center = cluster_start + (dtype_count - 1) * bar_width / 2
        cluster_centers.append(cluster_center)
        labels.append(matrix_size)
        current_pos += gap

# Determine title suffix based on included sources
sources_text = "n150 vs p150"

# Create figure with tighter margins and less scaling
plt.figure(figsize=(max(10, len(positions) * 0.3), 18))  # Increase height further to 18
bars = plt.bar(positions, heights, width=bar_width, color=dtype_colors)

# Tighten the plot boundaries to minimize whitespace
if positions:
    # Set xlim to focus only on the actual data range with minimal padding
    leftmost = min(positions) - bar_width * 2
    rightmost = max(positions) + bar_width * 2
    plt.xlim(leftmost, rightmost)

plt.xticks(cluster_centers, labels, rotation=45, ha='right')
plt.ylabel("Utilization (vs 8x8 full grid) %")
plt.xlabel("Matrix Size")
plt.title(f"Grid Utilization by Matrix Size and Data Type ({sources_text})")

# Create legend with proper colors for each dtype_fidelity and source combination
legend_handles = []
legend_labels = []

# Create a custom legend that shows the different shades by source
for dtype_fidelity in unique_dtype_fidelity:
    base_color = dtype_fidelity_colors[dtype_fidelity]
    
    for source in unique_sources:
        legend_handles.append(plt.Rectangle(
            (0, 0), 1, 1, 
            color=mcolors.to_rgba(base_color, alpha=source_shade[source])
        ))
        legend_labels.append(f"{dtype_fidelity} ({source})")

# Even tighter margins with much less bottom space
plt.subplots_adjust(left=0.03, right=0.97, bottom=0.15)  # Reduce bottom margin significantly from 0.3 to 0.15

plt.legend(
    legend_handles, legend_labels, title="DType_Fidelity (Source)",
    loc='lower center',
    bbox_to_anchor=(0.5, -0.25),  # Move legend way up from -0.6 to -0.25
    ncol=min(len(legend_labels), 4)
)

filename = "tech_reports/GEMM_FLOPS/utilization_by_matrix_size_and_type"
if square_matrices_only:
    filename += "_square"
filename += ".png"

# Use tight bounding box when saving
plt.savefig(filename, bbox_inches="tight")
plt.close()