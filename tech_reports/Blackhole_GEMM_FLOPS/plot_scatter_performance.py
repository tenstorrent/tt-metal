# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

# Arrow annotation configuration
arrow_label_config = {
    "enable_flipping": True,  # Enable alternating left/right labels
    "first_label_right": False,  # First label goes on the right
    "force_right": False,  # Force all labels to the right
    "force_left": False   # Force all labels to the left
}

# Filter by dtype - set to None to include all, or specify list of dtypes to include
filter_dtypes = ["BFLOAT8_B", "BFLOAT4_B"]  # Only plot BFLOAT8_B

# Filter by math fidelity - set to None to include all, or specify list of fidelities
filter_fidelities = ["LoFi"]  # Only plot LoFi and HiFi2

# Read CSV files and add a 'source' column to each
df1 = pd.read_csv("tech_reports/GEMM_FLOPS/n150-8x8.csv")
df1["source"] = "n150"
df2 = pd.read_csv("tech_reports/GEMM_FLOPS/p150-12x10.csv")
df2["source"] = "p150"

# Combine dataframes
dataframes = [df1, df2]
df = pd.concat(dataframes, ignore_index=True)

# Create short dtype string without "DataType." prefix
df["dtype_short"] = df["dtype"].str.replace("DataType.", "", regex=False)
df["math_fidelity_short"] = df["math_fidelity"].str.replace("MathFidelity.", "", regex=False)
df["dtype_fidelity"] = df["dtype_short"] + "_" + df["math_fidelity_short"]

# Optionally filter by dtype and math fidelity
if filter_dtypes is not None:
    df = df[df["dtype_short"].isin(filter_dtypes)]
    if len(df) == 0:
        raise ValueError("No data matches the selected dtype filter!")

if filter_fidelities is not None:
    df = df[df["math_fidelity_short"].isin(filter_fidelities)]
    if len(df) == 0:
        raise ValueError("No data matches the selected fidelity filter!")

# Calculate total matrix elements
df["matrix_elements"] = df["m"] * df["k"] * df["n"]

# Get unique values for dtype_fidelity and sources
unique_dtype_fidelity = df["dtype_fidelity"].unique()
unique_sources = df["source"].unique()

# Create a color map for each unique dtype_fidelity
base_colors = plt.cm.tab10.colors
dtype_fidelity_colors = {dfid: base_colors[i % len(base_colors)] for i, dfid in enumerate(unique_dtype_fidelity)}
source_markers = {"n150": "o", "p150": "^"}

# Storage precedence: L1,L1,L1 > L1,DRAM,L1 > DRAM,DRAM,DRAM
storage_orders = [
    ("L1", "L1", "L1"),
    ("L1", "DRAM", "L1"),
    ("DRAM", "DRAM", "DRAM"),
]

# Increase width for better horizontal ratio (16:12 instead of 12:12)
plt.figure(figsize=(16, 12))

# Track which side the label should go on (left or right)
label_on_right = arrow_label_config["first_label_right"]
arrow_count = 0

for dtype_fidelity in unique_dtype_fidelity:
    # --- p150 ---
    group_p150 = df[(df["dtype_fidelity"] == dtype_fidelity) & (df["source"] == "p150")]
    group_p150 = group_p150.sort_values("matrix_elements")
    selected_rows_p150 = []
    for matrix_elements in group_p150["matrix_elements"].unique():
        subg = group_p150[group_p150["matrix_elements"] == matrix_elements]
        for storage in storage_orders:
            match = subg[
                (subg["in0_storage_type"] == storage[0]) &
                (subg["in1_storage_type"] == storage[1]) &
                (subg["out_storage_type"] == storage[2])
            ]
            if not match.empty:
                selected_rows_p150.append(match.iloc[0])
                break
    if not selected_rows_p150:
        continue
    selected_df_p150 = pd.DataFrame(selected_rows_p150).sort_values("matrix_elements")
    p150_x_cutoff = selected_df_p150["matrix_elements"].max()

    # --- n150, cutoff at p150 max x ---
    group_n150 = df[(df["dtype_fidelity"] == dtype_fidelity) & (df["source"] == "n150")]
    group_n150 = group_n150.sort_values("matrix_elements")
    selected_rows_n150 = []
    for matrix_elements in group_n150["matrix_elements"].unique():
        if matrix_elements > p150_x_cutoff:
            continue
        subg = group_n150[group_n150["matrix_elements"] == matrix_elements]
        for storage in storage_orders:
            match = subg[
                (subg["in0_storage_type"] == storage[0]) &
                (subg["in1_storage_type"] == storage[1]) &
                (subg["out_storage_type"] == storage[2])
            ]
            if not match.empty:
                selected_rows_n150.append(match.iloc[0])
                break

    # Plot p150 (up to its own max x)
    plt.plot(
        selected_df_p150["matrix_elements"],
        selected_df_p150["TFLOPs (avg)"],
        color=dtype_fidelity_colors[dtype_fidelity],
        alpha=0.7,
        linewidth=1.5,
        marker=source_markers["p150"],
        label=f"{dtype_fidelity} (p150)"
    )
    plt.scatter(
        selected_df_p150["matrix_elements"],
        selected_df_p150["TFLOPs (avg)"],
        color=dtype_fidelity_colors[dtype_fidelity],
        marker=source_markers["p150"],
        s=60,
        alpha=0.8,
        zorder=10
    )

    # Plot n150 (cut off at p150's max x)
    if selected_rows_n150:
        selected_df_n150 = pd.DataFrame(selected_rows_n150).sort_values("matrix_elements")
        plt.plot(
            selected_df_n150["matrix_elements"],
            selected_df_n150["TFLOPs (avg)"],
            color=dtype_fidelity_colors[dtype_fidelity],
            alpha=0.7,
            linewidth=1.5,
            marker=source_markers["n150"],
            label=f"{dtype_fidelity} (n150)"
        )
        plt.scatter(
            selected_df_n150["matrix_elements"],
            selected_df_n150["TFLOPs (avg)"],
            color=dtype_fidelity_colors[dtype_fidelity],
            marker=source_markers["n150"],
            s=60,
            alpha=0.8,
            zorder=10
        )

        # Draw arrow and multiplier from n150 to p150 at the rightmost point (vertical arrow)
        # Use the rightmost x value (p150_x_cutoff)
        p150_rightmost = selected_df_p150[selected_df_p150["matrix_elements"] == p150_x_cutoff]
        n150_at_rightmost_x = selected_df_n150[selected_df_n150["matrix_elements"] <= p150_x_cutoff]
        
        if not p150_rightmost.empty and not n150_at_rightmost_x.empty:
            # Get the closest n150 point to the rightmost p150 point
            n150_rightmost = n150_at_rightmost_x.iloc[-1]
            p150_rightmost_point = p150_rightmost.iloc[0]
            
            # Create a vertical arrow
            x_pos = n150_rightmost["matrix_elements"]
            arrow_start = (x_pos, n150_rightmost["TFLOPs (avg)"])
            arrow_end = (x_pos, p150_rightmost_point["TFLOPs (avg)"])
            ratio = p150_rightmost_point["TFLOPs (avg)"] / n150_rightmost["TFLOPs (avg)"]
            
            # Draw the arrow with thicker linewidth
            plt.annotate(
                "",  # No text on the arrow itself
                xy=arrow_end,
                xytext=arrow_start,
                arrowprops=dict(facecolor='black', arrowstyle="->", linewidth=2.5),
                fontsize=12,
                color="black"
            )
            
            # Determine label position based on configuration
            if arrow_label_config["force_right"]:
                label_on_right = True
            elif arrow_label_config["force_left"]:
                label_on_right = False
            
            # Position for the multiplier text
            text_x_pos = x_pos * (1.2 if label_on_right else 0.85)
            text_y_pos = (arrow_start[1] + arrow_end[1]) / 2  # Middle of arrow (vertically)
            
            # Text alignment
            ha_align = "left" if label_on_right else "right"
            
            # Create boxed and bold text
            bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec="black", lw=2)
            plt.text(
                text_x_pos,
                text_y_pos,  # This should already be the middle of the arrow vertically
                f"×{ratio:.2f}",
                fontsize=14,
                fontweight='bold',
                color="black",
                ha=ha_align,
                va="center",  # Ensure vertical centering
                bbox=bbox_props
            )
            
            # Update label side for next arrow if flipping is enabled
            if arrow_label_config["enable_flipping"]:
                label_on_right = not label_on_right
            
            arrow_count += 1

plt.xscale("log")
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
plt.xlabel("Flops Required to Compute Matrix Multiply - Log Scale")
plt.ylabel("TFLOPs (avg)")
plt.title("TFLOPs vs Matrix Size (n150 vs p150) - LoFi Comparison")
# Move legend to the bottom of the plot and make it wider
plt.legend(
    title="DType_Fidelity (Source)",
    loc='upper center',
    bbox_to_anchor=(0.5, -0.15),  # Position below the plot
    ncol=3,  # Use 3 columns to make it more horizontal
    fontsize=9
)
plt.tight_layout()
plt.savefig("tech_reports/GEMM_FLOPS/flops_vs_matrix_elements_bfloat8b_fidelity_comparison.png", bbox_inches="tight")
plt.close()