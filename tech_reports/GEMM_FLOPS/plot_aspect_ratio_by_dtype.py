# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import gcd
from functools import reduce


def gcd_of_three(a, b, c):
    """Calculate GCD of three numbers"""
    return reduce(gcd, [a, b, c])


def calculate_aspect_ratio(m, k, n, grid_x, grid_y):
    """Calculate aspect ratio from scaled dimensions by reversing grid scaling"""
    base_m = m // grid_y
    base_k = k // grid_x
    base_n = n // grid_x

    divisor = gcd_of_three(base_m, base_k, base_n)
    ratio_m = base_m // divisor
    ratio_k = base_k // divisor
    ratio_n = base_n // divisor

    return f"{ratio_m}:{ratio_k}:{ratio_n}"


# Load N150 and P150 data
df_n150 = pd.read_csv("tech_reports/GEMM_FLOPS/n150-sweep.csv")
df_n150["source"] = "N150"
df_n150["grid_x"] = 8
df_n150["grid_y"] = 8

df_p150 = pd.read_csv("tech_reports/GEMM_FLOPS/p150-sweep.csv")
df_p150["source"] = "P150"
df_p150["grid_x"] = 13
df_p150["grid_y"] = 10

# Standardize column names
if "TFLOPs (avg)" in df_n150.columns:
    df_n150.rename(columns={"TFLOPs (avg)": "tflops"}, inplace=True)
if "TFLOPs (avg)" in df_p150.columns:
    df_p150.rename(columns={"TFLOPs (avg)": "tflops"}, inplace=True)

df_n150["tflops"] = pd.to_numeric(df_n150["tflops"], errors="coerce")
df_p150["tflops"] = pd.to_numeric(df_p150["tflops"], errors="coerce")

# Create dtype_fidelity column
df_n150["dtype_fidelity"] = (
    df_n150["dtype"].astype(str).str.replace("DataType.", "")
    + "_"
    + df_n150["math_fidelity"].astype(str).str.replace("MathFidelity.", "")
)
df_p150["dtype_fidelity"] = (
    df_p150["dtype"].astype(str).str.replace("DataType.", "")
    + "_"
    + df_p150["math_fidelity"].astype(str).str.replace("MathFidelity.", "")
)

# Calculate aspect ratios
print("Calculating aspect ratios...")
df_n150["aspect_ratio"] = df_n150.apply(
    lambda row: calculate_aspect_ratio(row["m"], row["k"], row["n"], row["grid_x"], row["grid_y"]), axis=1
)
df_p150["aspect_ratio"] = df_p150.apply(
    lambda row: calculate_aspect_ratio(row["m"], row["k"], row["n"], row["grid_x"], row["grid_y"]), axis=1
)

df = pd.concat([df_n150, df_p150], ignore_index=True)

# Define dtype-fidelity pairs and aspect ratios (colors match scatter/bar plots)
dtype_configs = [
    ("BFLOAT4_B_LoFi", "BFLOAT4_B-LoFi", "#2ca02c"),  # Green
    ("BFLOAT8_B_HiFi2", "BFLOAT8_B-HiFi2", "#ff7f0e"),  # Orange
    ("BFLOAT16_HiFi4", "BFLOAT16-HiFi4", "#1f77b4"),  # Blue
]

aspect_ratios = ["1:1:1", "1:2:4"]
aspect_labels = {"1:1:1": "Square\n(1:1:1)", "1:2:4": "Rectangular\n(1:2:4)"}

# Create plot for each device
for source in ["N150", "P150"]:
    device_data = df[df["source"] == source].copy()

    if device_data.empty:
        print(f"\n❌ No data for {source}")
        continue

    # Collect data: for each aspect ratio, get all 3 dtypes
    # Use balanced sampling: select top N matrix sizes for fair comparison
    TOP_N_SIZES = 3  # Use the 3 largest matrix sizes from each aspect ratio
    summary_data = []

    for aspect_ratio in aspect_ratios:
        for dtype_fidelity, dtype_label, color in dtype_configs:
            # Filter for this specific dtype and aspect ratio
            filtered = device_data[
                (device_data["dtype_fidelity"] == dtype_fidelity) & (device_data["aspect_ratio"] == aspect_ratio)
            ]

            if filtered.empty:
                print(f"  ⚠️  No data for {source} {dtype_fidelity} with aspect {aspect_ratio}")
                summary_data.append(
                    {
                        "aspect_ratio": aspect_ratio,
                        "dtype_fidelity": dtype_fidelity,
                        "dtype_label": dtype_label,
                        "avg_tflops": 0,
                        "color": color,
                    }
                )
                continue

            # Group by matrix size and get best TFLOPs for each size
            size_tflops = []
            for matrix_size, group in filtered.groupby(["m", "k", "n"]):
                total_elements = matrix_size[0] * matrix_size[1] * matrix_size[2]
                best_tflops = group["tflops"].max()
                size_tflops.append((total_elements, best_tflops))

            # Sort by total elements and take TOP_N largest
            size_tflops.sort(key=lambda x: x[0], reverse=True)
            top_n_tflops = [tflops for _, tflops in size_tflops[:TOP_N_SIZES]]

            if not top_n_tflops:
                avg_tflops = 0
            else:
                avg_tflops = np.mean(top_n_tflops)

            summary_data.append(
                {
                    "aspect_ratio": aspect_ratio,
                    "dtype_fidelity": dtype_fidelity,
                    "dtype_label": dtype_label,
                    "avg_tflops": avg_tflops,
                    "color": color,
                }
            )

            print(
                f"  ✓ {source} {dtype_fidelity} ({aspect_ratio}): {avg_tflops:.1f} TFLOPs (avg of top {len(top_n_tflops)} sizes out of {len(size_tflops)} total)"
            )

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(16, 8))

    # Set up positions: 3 aspect ratios, each with 3 dtypes
    n_aspects = len(aspect_ratios)
    n_dtypes = len(dtype_configs)
    bar_width = 0.22
    group_gap = 0.5  # Larger gap between aspect ratio groups

    aspect_positions = []
    current_pos = 0

    for aspect_idx, aspect_ratio in enumerate(aspect_ratios):
        # Plot 3 bars for this aspect ratio (one per dtype)
        for dtype_idx, (dtype_fidelity, dtype_label, color) in enumerate(dtype_configs):
            # Find the data for this combination
            data_point = next(
                (
                    d
                    for d in summary_data
                    if d["aspect_ratio"] == aspect_ratio and d["dtype_fidelity"] == dtype_fidelity
                ),
                None,
            )

            if data_point:
                value = data_point["avg_tflops"]
                bar_pos = current_pos + dtype_idx * bar_width

                bar = ax.bar(
                    bar_pos,
                    value,
                    bar_width,
                    color=color,
                    alpha=0.85,
                    edgecolor="black",
                    linewidth=1.3,
                    label=dtype_label if aspect_idx == 0 else "",
                )  # Only label once

                # Add value label
                if value > 0:
                    ax.text(bar_pos, value, f"{value:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        # Store center position for this aspect ratio group
        aspect_center = current_pos + (n_dtypes * bar_width) / 2 - bar_width / 2
        aspect_positions.append(aspect_center)

        # Move to next group
        current_pos += n_dtypes * bar_width + group_gap

    ax.set_ylabel("Average TFLOPs", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_xlabel(
        "Matrix Aspect Ratio (m:k:n)\n(m=input rows, k=inner dim, n=output cols)",
        fontsize=13,
        fontweight="bold",
        labelpad=10,
    )

    # Title matching our other plots
    device_name = "N150 (Wormhole)" if source == "N150" else "P150 (Blackhole)"
    fig.suptitle(f"Performance Comparison: {device_name}", fontsize=18, fontweight="bold", y=0.98)
    ax.set_title("TFLOPs by Matrix Aspect Ratio and Data Type", fontsize=14, pad=10, fontweight="bold")

    # Set x-axis labels for aspect ratios
    ax.set_xticks(aspect_positions)
    ax.set_xticklabels([aspect_labels[r] for r in aspect_ratios], fontsize=12)

    # Add legend for dtypes - position to avoid overlap with bars
    ax.legend(
        fontsize=11,
        loc="upper right",
        framealpha=0.95,
        edgecolor="black",
        title="Data Type (Math Fidelity)",
        title_fontsize=12,
    )

    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(
        f"tech_reports/GEMM_FLOPS/images/aspect_ratio_by_dtype_{source.lower()}.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(f"\n✅ Saved: aspect_ratio_by_dtype_{source.lower()}.png")

print("\n✅ Aspect ratio by dtype comparison complete!")
