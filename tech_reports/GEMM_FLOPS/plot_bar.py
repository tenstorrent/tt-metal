# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from config_utils import load_sweep_data, get_bar_chart_data

# --- Configuration ---
precision_fidelity_pairs = [("BFLOAT16", "HiFi4"), ("BFLOAT8_B", "HiFi2"), ("BFLOAT4_B", "LoFi")]
matrix_size_mapping = {
    "512": "640",
    "1024": "1280",
    "2048": "2560",
    "3072": "3840",
    "4096": "5120",
    "8192": "10240",
    "16384": "20480",
}

# --- Data Loading and Processing ---
df = load_sweep_data()
chart_data = get_bar_chart_data(df, precision_fidelity_pairs, matrix_size_mapping)

if not chart_data:
    print("No data available for plotting.")
    exit()

# --- Plotting Setup ---
bar_width, gap = 0.05, 0.15
positions, heights, dtype_colors, cluster_centers, labels, multipliers = [], [], [], [], [], []
current_pos = 0

base_colors = plt.cm.tab10.colors
dtype_color_map = {pair[0]: color for pair, color in zip(precision_fidelity_pairs, base_colors)}
source_shade = {"n150": 0.7, "p150": 1.0}

# --- Plotting Logic ---
for combined_label, data in chart_data.items():
    cluster_start = current_pos
    values = {"n150": {"positions": {}, "heights": {}}, "p150": {"positions": {}, "heights": {}}}

    for source in ["n150", "p150"]:
        for dtype, fidelity in precision_fidelity_pairs:
            if dtype in data[source]:
                height = data[source][dtype]
                positions.append(current_pos)
                heights.append(height)
                values[source]["positions"][dtype] = current_pos
                values[source]["heights"][dtype] = height

                color = mcolors.to_rgba(dtype_color_map.get(dtype, "gray"), alpha=source_shade[source])
                dtype_colors.append(color)
                current_pos += bar_width
            current_pos += 0.02
        # ADJUSTMENT: Reduced the gap between n150 and p150 bars
        if source == "n150":
            current_pos += gap * 0.2

    # Use n150 BFLOAT16 as the reference for all multipliers (light blue bar)
    if "BFLOAT16" in values["n150"]["heights"]:
        base_height = values["n150"]["heights"]["BFLOAT16"]
        for source in ["n150", "p150"]:
            for dtype, _ in precision_fidelity_pairs:
                if dtype in values[source]["heights"]:
                    ratio = values[source]["heights"][dtype] / base_height
                    multipliers.append(
                        (values[source]["positions"][dtype], values[source]["heights"][dtype], f"{ratio:.2f}x")
                    )

    cluster_centers.append((cluster_start + current_pos - gap) / 2)
    labels.append(combined_label)
    current_pos += gap

# --- Final Plot Rendering ---
if positions:
    # ADJUSTMENT: Increased the figure width multiplier from 2.0 to 3.0 for more space
    plt.figure(figsize=(max(20, len(labels) * 3.5), 14))
    plt.bar(positions, heights, width=bar_width, color=dtype_colors)
    plt.ylim(0, max(heights) * 1.2)

    for pos, height, text in multipliers:
        plt.annotate(
            text,
            (pos, height + max(heights) * 0.02),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
        )

    plt.xticks(cluster_centers, labels, rotation=45, ha="right")
    plt.tick_params(axis="x", which="both", bottom=False, top=False)
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.ylabel("TFLOPs (avg)")
    plt.xlabel("Matrix Size (n150 / p150 base 'm' dim)")
    plt.title("TFLOPs by Matrix Size and Data Type")

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=mcolors.to_rgba(dtype_color_map.get(d, "gray"), alpha=s_shade))
        for d, f in precision_fidelity_pairs
        for s, s_shade in source_shade.items()
    ]
    legend_labels = [f"{d}-{f} ({s})" for d, f in precision_fidelity_pairs for s in source_shade.keys()]

    plt.xlim(min(positions) - bar_width * 2, max(positions) + bar_width * 2)
    plt.subplots_adjust(left=0.03, right=0.97, bottom=0.25)
    plt.legend(
        legend_handles,
        legend_labels,
        title="DType_Fidelity (Source)",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
    )

    plt.savefig("tech_reports/GEMM_FLOPS/images/flops_by_matrix_size_and_type_sorted.png", bbox_inches="tight")
    plt.close()

print("Bar chart saved!")
