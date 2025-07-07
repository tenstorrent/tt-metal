# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

# Configuration options
square_matrices_only = False  # Set to False to include all matrices (not just square)

# Filter by dtype - set to None to include all, or specify list of dtypes to include
# Example: ["BFLOAT8_B", "BFLOAT16"] or None
#filter_dtypes = None  # Set to None to include all dtypes
filter_dtypes = ["BFLOAT4_B", "BFLOAT8_B", "BFLOAT16"]  # Uncomment and modify to filter specific dtypes

# Read CSV files and add a 'source' column to each
df1 = pd.read_csv("tech_reports/GEMM_FLOPS/n150-8x8.csv")
df1["source"] = "n150"
df2 = pd.read_csv("tech_reports/GEMM_FLOPS/p150-8x8.csv")
df2["source"] = "p150"

# Combine dataframes
dataframes = [df1, df2]

# Remove all OOB-related code
# if include_p150oob:
#     df3 = pd.read_csv("p150oob.csv")
#     df3["source"] = "p150oob"
#     dataframes.append(df3)

# Concatenate all dataframes
df = pd.concat(dataframes, ignore_index=True)

# Create short dtype string without "DataType." prefix
df["dtype_short"] = df["dtype"].str.replace("DataType.", "", regex=False)

# Filter by selected dtypes if specified
if filter_dtypes is not None:
    df = df[df["dtype_short"].isin(filter_dtypes)]
    if len(df) == 0:
        raise ValueError("No data matches the selected dtype filter!")

# Create a new column for matrix size as a string
df["matrix_size"] = df.apply(lambda row: f"{row['m']}x{row['k']}x{row['n']}", axis=1)

# Filter for square matrices if option is enabled - fixed to check dimensions first
if square_matrices_only:
    print(f"Before square filter: {len(df)} rows")
    # Square matrix means all three dimensions are equal
    df = df[(df["m"] == df["n"]) & (df["n"] == df["k"])]
    print(f"After square filter: {len(df)} rows")
    if len(df) == 0:
        raise ValueError("No square matrices found in the dataset!")

# Create a combined dtype+fidelity+source label for legend and color
df["legend_label"] = (
    df["dtype_short"] + "_" +
    df["math_fidelity"].str.replace("MathFidelity.", "", regex=False) +
    " (" + df["source"] + ")"
)
df["dtype_source"] = df["legend_label"]  # for compatibility below

# Create a column without source for pairing
df["dtype_fidelity"] = (
    df["dtype_short"] + "_" +
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

# Get unique dtype+source and matrix sizes for consistent order
dtypes_sources = df["dtype_source"].unique()
matrix_sizes = df["matrix_size"].unique()

# Filter for combinations available in both n150 and p150 FIRST
n150_combinations = df[df["source"] == "n150"][["matrix_size", "dtype_fidelity"]].drop_duplicates()
p150_combinations = df[df["source"] == "p150"][["matrix_size", "dtype_fidelity"]].drop_duplicates()

# Merge to find common combinations
common_combinations = pd.merge(
    n150_combinations, p150_combinations, 
    on=["matrix_size", "dtype_fidelity"], 
    how="inner"
)

# Filter dataset to only include matching combinations BEFORE creating df_best
df = pd.merge(
    df, common_combinations, 
    on=["matrix_size", "dtype_fidelity"], 
    how="inner"
)

# If dataframe is empty after filtering, raise an error
if len(df) == 0:
    raise ValueError("No matching data between n150 and p150 after filtering!")

# For each combination of matrix_size, dtype_fidelity, and source, keep only the best performing entry
best_performance_rows = []
for (matrix_size, dtype_fidelity, source), group in df.groupby(["matrix_size", "dtype_fidelity", "source"]):
    # Find the entry with the highest TFLOPs
    best_entry = group.loc[group["TFLOPs (avg)"].idxmax()]
    best_performance_rows.append(best_entry)

# Create a new dataframe with only the best entries
df_best = pd.DataFrame(best_performance_rows)

# Determine matrix elements (m×k×n) for sorting
df_best["total_elements"] = df_best["m"] * df_best["k"] * df_best["n"]

# Sort matrix sizes by total elements to ensure they're always increasing
matrix_sizes_sorted = []
matrix_elements_dict = {}

for matrix_size in matrix_sizes:
    # Get the total elements for this matrix size
    group = df_best[df_best["matrix_size"] == matrix_size]
    if not group.empty:
        total_elements = group["total_elements"].iloc[0]
        matrix_elements_dict[matrix_size] = total_elements
        
# Sort matrix sizes by their total elements
matrix_sizes_sorted = sorted(matrix_elements_dict.keys(), key=lambda x: matrix_elements_dict[x])

# Store annotation information: position, height, ratio text
multipliers = []
bar_width = 0.05  # Make bars extremely skinny
gap = 0.01  # Reduce gap between clusters even more
positions = []
heights = []
dtype_colors = []
cluster_centers = []
labels = []
current_pos = 0

# Use sorted matrix sizes in plot
for matrix_size in matrix_sizes_sorted:
    group = df_best[df_best["matrix_size"] == matrix_size]
    dtype_count = 0
    cluster_start = current_pos
    
    # Get unique dtype_fidelity combos for this matrix size
    combos = group["dtype_fidelity"].unique()
    
    # Sort combos based on n150 TFLOPs (or p150 if n150 doesn't exist)
    def get_sort_value(combo):
        n150_entry = group[(group["dtype_fidelity"] == combo) & (group["source"] == "n150")]
        if not n150_entry.empty:
            return n150_entry["TFLOPs (avg)"].values[0]
        p150_entry = group[(group["dtype_fidelity"] == combo) & (group["source"] == "p150")]
        if not p150_entry.empty:
            return p150_entry["TFLOPs (avg)"].values[0]
        return 0
    
    sorted_combos = sorted(combos, key=get_sort_value)
    
    # Dictionary to store n150 values for ratio calculation
    n150_values = {}
    
    # Process each dtype_fidelity combo, placing all sources side by side
    for combo in sorted_combos:
        # Process each source for this combo
        for source in unique_sources:
            source_entry = group[(group["dtype_fidelity"] == combo) & (group["source"] == source)]
            
            if not source_entry.empty:
                entry = source_entry.iloc[0]
                pos = current_pos
                height = entry["TFLOPs (avg)"]
                positions.append(pos)
                heights.append(height)
                
                # Get base color for this dtype_fidelity and adjust for source
                base_color = dtype_fidelity_colors[combo]
                adjusted_color = mcolors.to_rgba(base_color, alpha=source_shade[source])
                dtype_colors.append(adjusted_color)
                
                # Store n150 values for ratio calculation
                if source == "n150":
                    n150_values[combo] = height
                
                # Calculate ratio if n150 exists for this combo
                elif source != "n150" and combo in n150_values:
                    n150_val = n150_values[combo]
                    ratio = height / n150_val if n150_val > 0 else float('inf')
                    multipliers.append((pos, height, f"{ratio:.2f}x"))
                
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

# Determine title suffix based on included sources and dtype filter
sources_text = "n150 vs p150"
# Remove OOB-related code
# if include_p150oob:
#     sources_text = "n150 vs p150 vs p150oob"

dtype_text = ""
if filter_dtypes is not None:
    dtype_text = f" - {', '.join(filter_dtypes)}"

# Create figure with wider aspect ratio
plt.figure(figsize=(max(14, len(positions) * 0.4), 14))  # Increased width from 0.3 to 0.4 multiplier and minimum from 10 to 14
bars = plt.bar(positions, heights, width=bar_width, color=dtype_colors)

# Calculate maximum height needed for y-axis to fit all annotations
max_bar_height = max(heights) if heights else 0
padding = max_bar_height * 0.20  # Increase padding from 0.15 to 0.20 for more space at top

# Set y-axis limits with padding to accommodate annotations
plt.ylim(0, max_bar_height + padding)

# Add the multiplier annotations with clear spacing and formatting
for pos, height, text in multipliers:
    # Position text above bar with padding
    y_pos = height + max(height * 0.03, 1)  # Reduced minimum offset
    plt.annotate(
        text,
        xy=(pos, y_pos),
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )

plt.xticks(cluster_centers, labels, rotation=45, ha='right')
plt.ylabel("TFLOPs (avg)")
plt.xlabel("Matrix Size")
plt.title(f"TFLOPs by Matrix Size and Data Type 8x8 Compute Grid ({sources_text}){dtype_text}")

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

# Tighten the plot boundaries to minimize whitespace
if positions:
    # Set xlim to focus only on the actual data range with minimal padding
    leftmost = min(positions) - bar_width * 2
    rightmost = max(positions) + bar_width * 2
    plt.xlim(leftmost, rightmost)

# Even tighter margins with much less bottom space but more room for legend
plt.subplots_adjust(left=0.03, right=0.97, bottom=0.25)  # Increased bottom margin for legend

plt.legend(
    legend_handles, legend_labels, title="DType_Fidelity (Source)",
    loc='upper center',  # Move legend to bottom center
    bbox_to_anchor=(0.5, -0.12),  # Position just below the plot
    ncol=min(len(legend_labels), 6)  # Increase number of columns from 4 to 6
)

# Create filename with appropriate tags
filename = "tech_reports/GEMM_FLOPS/flops_by_matrix_size_and_type_sorted"
# Remove OOB-related code
# if include_p150oob:
#     filename += "_with_oob"
if filter_dtypes is not None:
    filename += "_filtered"
if square_matrices_only:
    filename += "_square"
filename += ".png"

# Use tight bounding box when saving
plt.savefig(filename, bbox_inches="tight")
plt.close()

