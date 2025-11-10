#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Utilization Scatter Plot - Device-based utilization comparison between N150 and P150

Usage:
1. Generate performance data using manually selected GEMM configurations
2. Rename output files to n150-manual.csv and p150-manual.csv
3. Place the CSV files in tech_reports/GEMM_FLOPS/
4. Run this script from the tt-metal root directory
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Configuration
dtype_configs = [
    ("BFLOAT4_B-LoFi", "#2ca02c"),  # Green
    ("BFLOAT8_B-HiFi2", "#ff7f0e"),  # Orange
    ("BFLOAT16-HiFi4", "#1f77b4"),  # Blue
]

device_configs = [
    ("P150", "^", "-"),  # Upward triangle, solid line
    ("N150", "v", "--"),  # Downward triangle, dashed line
]

# Load data
df_n150 = pd.read_csv("tech_reports/GEMM_FLOPS/n150-manual.csv")
df_p150 = pd.read_csv("tech_reports/GEMM_FLOPS/p150-manual.csv")

# Add device column
df_n150["device"] = "N150"
df_p150["device"] = "P150"

# Combine data
df = pd.concat([df_n150, df_p150], ignore_index=True)

# Filter: only non-traced data for clarity
df = df[df["use_trace"] == False]


# Create dtype-fidelity labels
def get_dtype_label(row):
    dtype = str(row["dtype"]).split(".")[-1]
    fidelity = str(row["math_fidelity"]).split(".")[-1]
    return f"{dtype}-{fidelity}"


df["dtype_label"] = df.apply(get_dtype_label, axis=1)

# Calculate total matrix elements
df["matrix_elements"] = df["m"] * df["k"] * df["n"]


# Extract device utilization - handle different grid size column names
def get_device_utilization(row):
    if row["device"] == "N150":
        col_name = "Device based utilization[%] (vs user selected grid 8x8)"
    else:  # P150
        col_name = "Device based utilization[%] (vs user selected grid 13x10)"

    if col_name in row.index:
        return row[col_name]
    else:
        return np.nan


df["device_utilization"] = df.apply(get_device_utilization, axis=1)

# Create plot
fig, ax = plt.subplots(figsize=(14, 8))

# Plot each combination of dtype and device
for dtype_label, dtype_color in dtype_configs:
    for device_name, marker, linestyle in device_configs:
        subset = df[(df["dtype_label"] == dtype_label) & (df["device"] == device_name)]

        if len(subset) > 0:
            # Sort by matrix size for line plot
            subset_sorted = subset.sort_values("matrix_elements")

            # Determine marker properties
            if device_name == "N150":
                markerfacecolor = "white"  # Hollow for N150
                markeredgewidth = 2
            else:
                markerfacecolor = dtype_color  # Filled for P150
                markeredgewidth = 1

            ax.plot(
                subset_sorted["matrix_elements"],
                subset_sorted["device_utilization"],
                marker=marker,
                linestyle=linestyle,
                color=dtype_color,
                markersize=8,
                markerfacecolor=markerfacecolor,
                markeredgecolor=dtype_color,
                markeredgewidth=markeredgewidth,
                linewidth=2,
                alpha=0.8,
            )

# Formatting
ax.set_xlabel("Total Matrix Elements (m × k × n)", fontsize=12, fontweight="bold")
# Add explanation below x-axis (non-bold)
ax.text(
    0.5,
    -0.12,
    "[(m,k) = input matrix size, (k,n) = weight matrix size]",
    transform=ax.transAxes,
    ha="center",
    va="top",
    fontsize=10,
)
ax.set_ylabel("Utilization (%)", fontsize=12, fontweight="bold")

# Set main title and subtitle
fig.suptitle(
    "Utilization Comparison: N150 (Wormhole) vs P150 (Blackhole)",
    fontsize=16,
    fontweight="bold",
    y=0.98,
)
ax.set_title(
    "Utilization vs Matrix Size for Different Data Types and Math Fidelities",
    fontsize=12,
    fontweight="bold",
    pad=10,
)

ax.set_xscale("log")
ax.grid(True, alpha=0.3, linestyle="--")
ax.set_ylim(0, 110)

# Create legend with bold headers
legend_elements = []

# Dtype section header (bold)
legend_elements.append(Line2D([0], [0], color="none", marker="", linestyle="", label="Dtype (Math Fidelity)"))

# Dtype entries (just colored lines, no markers)
for dtype_label, color in dtype_configs:
    parts = dtype_label.split("-")
    dtype_part = parts[0]  # Keep uppercase format
    fidelity_part = parts[1]
    formatted_label = f"{dtype_part} ({fidelity_part})"
    legend_elements.append(Line2D([0], [0], color=color, linewidth=3, label=f"{formatted_label}"))

# Empty line for spacing
legend_elements.append(Line2D([0], [0], color="none", marker="", linestyle="", label=""))

# Device section header (bold)
legend_elements.append(Line2D([0], [0], color="none", marker="", linestyle="", label="Device"))

# Device entries with markers and line styles
legend_elements.append(
    Line2D(
        [0],
        [0],
        color="gray",
        linewidth=2.5,
        linestyle="-",
        marker="^",
        markersize=8,
        markerfacecolor="gray",
        markeredgecolor="black",
        markeredgewidth=1,
        label="P150",
    )
)
legend_elements.append(
    Line2D(
        [0],
        [0],
        color="gray",
        linewidth=2.5,
        linestyle="--",
        marker="v",
        markersize=8,
        markerfacecolor="white",
        markeredgecolor="gray",
        markeredgewidth=2,
        label="N150",
    )
)

# Create legend with custom handler to make headers bold
legend = ax.legend(
    handles=legend_elements, loc="lower right", fontsize=10, framealpha=0.95, edgecolor="black", handlelength=3.5
)

# Make the headers bold
for text in legend.get_texts():
    if ":" in text.get_text():
        text.set_weight("bold")

plt.tight_layout()
plt.savefig("tech_reports/GEMM_FLOPS/images/utilization_comparison.png", dpi=300, bbox_inches="tight")
print("✓ Utilization scatter plot saved: tech_reports/GEMM_FLOPS/images/utilization_comparison.png")
plt.close()
