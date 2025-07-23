# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

# Configuration options
square_matrices_only = True  # Set to False to include all matrices (not just square)

# Filter by dtype - set to None to include all, or specify list of dtypes to include
# Example: ["BFLOAT8_B", "BFLOAT16"] or None
# filter_dtypes = None  # Set to None to include all dtypes
filter_dtypes = ["BFLOAT4_B", "BFLOAT8_B", "BFLOAT16"]  # Uncomment and modify to filter specific dtypes

# Read CSV files and add a 'source' column to each
df1 = pd.read_csv("tech_reports/GEMM_FLOPS/n150.csv")
df1["source"] = "n150"
df2 = pd.read_csv("tech_reports/GEMM_FLOPS/p150.csv")
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
    # For n150, all three dimensions should be equal
    # For p150, only k and n should be equal (second and third dimensions)
    n150_mask = (df["source"] == "n150") & (df["m"] == df["n"]) & (df["n"] == df["k"])
    p150_mask = (df["source"] == "p150") & (df["k"] == df["n"])
    df = df[n150_mask | p150_mask]
    print(f"After square/near-square filter: {len(df)} rows")
    if len(df) == 0:
        raise ValueError("No square/near-square matrices found in the dataset!")

# Create a combined dtype+fidelity+source label for legend and color
df["legend_label"] = (
    df["dtype_short"]
    + "_"
    + df["math_fidelity"].str.replace("MathFidelity.", "", regex=False)
    + " ("
    + df["source"]
    + ")"
)
df["dtype_source"] = df["legend_label"]  # for compatibility below

# Create a column without source for pairing
df["dtype_fidelity"] = df["dtype_short"] + "_" + df["math_fidelity"].str.replace("MathFidelity.", "", regex=False)

# Get unique values for dtype_fidelity and sources
unique_dtype_fidelity = df["dtype_fidelity"].unique()
unique_sources = df["source"].unique()

# Create a color map for each unique dtype_fidelity
base_colors = plt.cm.tab10.colors  # Use fewer base colors for more distinction
dtype_fidelity_colors = {df: base_colors[i % len(base_colors)] for i, df in enumerate(unique_dtype_fidelity)}

# Create a shade adjustment for each source
source_shade = {"n150": 0.7, "p150": 1.0}

# For each combination of matrix_size, dtype_fidelity, and source, keep only the best performing entry
best_performance_rows = []
for (matrix_size, dtype_fidelity, source), group in df.groupby(["matrix_size", "dtype_fidelity", "source"]):
    # Find the entry with the highest TFLOPs
    best_entry = group.loc[group["TFLOPs (avg)"].idxmax()]
    best_performance_rows.append(best_entry)

# Create a new dataframe with only the best entries
df_best = pd.DataFrame(best_performance_rows)

# Create a new column for matrix size as a string
df_best["matrix_size"] = df_best.apply(lambda row: f"{row['m']}x{row['k']}x{row['n']}", axis=1)

# Get unique matrix sizes for consistent order
matrix_sizes = df_best["matrix_size"].unique()

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

# Define matrix size mapping between n150 and p150
matrix_size_mapping = {
    "512": "640",  # Maps to 640x832x832
    "1024": "1280",  # Maps to 1280x1664x1664
    "2048": "2560",  # Maps to 2560x3328x3328
    "3072": "3840",  # Maps to 3840x4992x4992
    "4096": "5120",  # Maps to 5120x6656x6656
    "8192": "10240",  # Maps to 10240x13312x13312
    "16384": "20480",  # Maps to 20480x26624x26624
}

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

# Define the precision-fidelity pairs we want to compare
precision_fidelity_pairs = [("BFLOAT16", "HiFi4"), ("BFLOAT8_B", "HiFi2"), ("BFLOAT4_B", "LoFi")]

# Group matrix sizes by their mapping
grouped_sizes = {}
for n150_size in matrix_sizes_sorted:
    n150_dim = n150_size.split("x")[0]  # Get the first dimension
    if n150_dim in matrix_size_mapping:
        p150_base = matrix_size_mapping[n150_dim]
        # For p150, we need to handle non-square matrices
        if p150_base == "640":
            p150_size = "640x832x832"
        elif p150_base == "1280":
            p150_size = "1280x1664x1664"
        elif p150_base == "2560":
            p150_size = "2560x3328x3328"
        elif p150_base == "3840":
            p150_size = "3840x4992x4992"
        elif p150_base == "5120":
            p150_size = "5120x6656x6656"
        elif p150_base == "10240":
            p150_size = "10240x13312x13312"
        elif p150_base == "20480":
            p150_size = "20480x26624x26624"
        else:
            p150_size = f"{p150_base}x{p150_base}x{p150_base}"

        # Debug print
        print(f"Mapping {n150_size} to {p150_size}")
        grouped_sizes[f"{n150_dim}/{p150_base}"] = (n150_size, p150_size)

# Use grouped sizes for plotting
for combined_label, (n150_size, p150_size) in grouped_sizes.items():
    group_n150 = df_best[df_best["matrix_size"] == n150_size]
    group_p150 = df_best[df_best["matrix_size"] == p150_size]

    # Debug print
    print(f"\nProcessing sizes: n150={n150_size}, p150={p150_size}")
    print(f"Found {len(group_n150)} n150 entries and {len(group_p150)} p150 entries")

    if len(group_n150) == 0 or len(group_p150) == 0:
        print(f"Skipping {combined_label} due to missing data")
        continue

    dtype_count = 0
    cluster_start = current_pos

    # Dictionary to store values for each precision and source
    values = {"n150": {"positions": {}, "heights": {}}, "p150": {"positions": {}, "heights": {}}}

    # First pass: collect values and positions for both sources
    for source, group in [("n150", group_n150), ("p150", group_p150)]:
        for dtype, fidelity in precision_fidelity_pairs:
            source_entry = group[
                (group["dtype_short"] == dtype)
                & (group["math_fidelity"].str.contains(fidelity))
                & (group["source"] == source)  # Add source filter back
            ]

            if not source_entry.empty:
                entry = source_entry.iloc[0]
                pos = current_pos
                height = entry["TFLOPs (avg)"]
                positions.append(pos)
                heights.append(height)
                values[source]["positions"][dtype] = pos
                values[source]["heights"][dtype] = height

                # Debug print
                print(f"Added {source} {dtype} {fidelity}: height={height}")

                # Get base color for this dtype_fidelity and adjust for source
                combo = entry["dtype_fidelity"]
                base_color = dtype_fidelity_colors[combo]
                adjusted_color = mcolors.to_rgba(base_color, alpha=source_shade[source])
                dtype_colors.append(adjusted_color)

                dtype_count += 1
                current_pos += bar_width

            else:
                print(f"No data found for {source} {dtype} {fidelity}")

            # Add gap between different dtype_fidelity combos within the cluster
            current_pos += 0.02

        # Add larger gap between n150 and p150 groups
        if source == "n150":
            current_pos += gap * 3

    # Second pass: calculate multipliers based on BFLOAT16 within each device
    for source in ["n150", "p150"]:
        if "BFLOAT16" in values[source]["heights"]:
            base_height = values[source]["heights"]["BFLOAT16"]

            # Add 1.00x for the base (BFLOAT16)
            multipliers.append(
                (values[source]["positions"]["BFLOAT16"], values[source]["heights"]["BFLOAT16"], "1.00x")
            )

            # Calculate multipliers for other precisions relative to BFLOAT16 in this device
            for dtype in ["BFLOAT8_B", "BFLOAT4_B"]:
                if dtype in values[source]["heights"]:
                    ratio = values[source]["heights"][dtype] / base_height
                    multipliers.append(
                        (values[source]["positions"][dtype], values[source]["heights"][dtype], f"{ratio:.2f}x")
                    )

    if dtype_count > 0:
        # Center label under the cluster
        cluster_center = cluster_start + (dtype_count - 1) * bar_width / 2
        cluster_centers.append(cluster_center)
        labels.append(combined_label)
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
plt.figure(
    figsize=(max(14, len(positions) * 0.4), 14)
)  # Increased width from 0.3 to 0.4 multiplier and minimum from 10 to 14
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
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

plt.xticks(cluster_centers, labels, rotation=45, ha="right")
plt.ylabel("TFLOPs (avg)")
plt.xlabel("Matrix Size")
plt.title(f"TFLOPs by Matrix Size and Data Type")

# Create legend with proper colors for each dtype_fidelity and source combination
legend_handles = []
legend_labels = []

# Create a custom legend that shows the different shades by source
for dtype_fidelity in unique_dtype_fidelity:
    base_color = dtype_fidelity_colors[dtype_fidelity]

    for source in unique_sources:
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, color=mcolors.to_rgba(base_color, alpha=source_shade[source]))
        )
        legend_labels.append(f"{dtype_fidelity} ({source})")

# Tighten the plot boundaries to minimize whitespace
if positions:
    # Set xlim to focus only on the actual data range with minimal padding
    leftmost = min(positions) - bar_width * 2
    rightmost = max(positions) + bar_width * 2
    plt.xlim(leftmost, rightmost)

# Even tighter margins with much less bottom space but more room for legend
plt.subplots_adjust(left=0.03, right=0.97, bottom=0.25)  # Increased bottom margin for legend

# Adjust legend position - move it down more
plt.legend(
    legend_handles,
    legend_labels,
    title="DType_Fidelity (Source)",
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
    ncol=min(len(legend_labels), 6),
)

# Create filename with appropriate tags
filename = "tech_reports/GEMM_FLOPS/images/flops_by_matrix_size_and_type_sorted"
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
