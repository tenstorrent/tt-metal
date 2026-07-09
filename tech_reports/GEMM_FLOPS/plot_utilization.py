#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Utilization Scatter Plot - Device-based utilization comparison between N150 and P150

Usage:
1. Run the benchmark via run_bench.sh on both devices
2. CSVs are placed in tech_reports/GEMM_FLOPS/data/{wh,bh}.csv
3. Run this script from the tt-metal root directory
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DATA_DIR = Path("tech_reports/GEMM_FLOPS/data")
IMG_DIR = Path("tech_reports/GEMM_FLOPS/images")

DEVICE_FILES = {
    "N150": DATA_DIR / "wh.csv",
    "P150": DATA_DIR / "bh.csv",
}

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


def safe_read_csv(path):
    """Return the CSV as a DataFrame, or an empty DataFrame if the file is missing."""
    if path.exists():
        return pd.read_csv(path)
    print(f"WARNING: {path} not found — skipping that device.")
    return pd.DataFrame()


def find_device_util_col(df):
    """Find the 'Device based utilization[%] (vs user selected grid ...)' column."""
    for col in df.columns:
        if col.startswith("Device based utilization") and "user selected grid" in col:
            return col
    return None


def load_device_data(path, device_name):
    """Load and preprocess a device CSV."""
    df = safe_read_csv(path)
    if df.empty:
        return df
    df["device"] = device_name
    # Filter to tuned_2d_l1 mode (equivalent to old sharded benchmark data)
    if "mode" in df.columns:
        df = df[df["mode"] == "tuned_2d_l1"].copy()
    return df


# Load data
df_n150 = load_device_data(DEVICE_FILES["N150"], "N150")
df_p150 = load_device_data(DEVICE_FILES["P150"], "P150")

# Combine data
df = pd.concat([df_n150, df_p150], ignore_index=True)

if df.empty:
    print("ERROR: No data available for any device. Exiting.")
    raise SystemExit(1)

# Filter: only non-traced data for clarity
if df["use_trace"].dtype == object:
    df["use_trace"] = df["use_trace"].astype(str).str.lower() == "true"
df = df[df["use_trace"] == False]


# Create dtype-fidelity labels
def get_dtype_label(row):
    dtype = str(row["dtype"]).split(".")[-1]
    fidelity = str(row["math_fidelity"]).split(".")[-1]
    return f"{dtype}-{fidelity}"


df["dtype_label"] = df.apply(get_dtype_label, axis=1)

# Calculate total matrix elements
df["matrix_elements"] = df["m"] * df["k"] * df["n"]

# Find the device utilization column dynamically
device_util_col = find_device_util_col(df)
if device_util_col is None:
    print("WARNING: No device-based utilization column found. Using host-based.")
    for col in df.columns:
        if col.startswith("Host based utilization") and "user selected grid" in col:
            device_util_col = col
            break

if device_util_col is None:
    print("ERROR: No utilization column found at all. Exiting.")
    raise SystemExit(1)

df["device_utilization"] = pd.to_numeric(df[device_util_col], errors="coerce")

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
ax.set_xlabel("Total Matrix Elements (M × K × N)", fontsize=12, fontweight="bold")
# Add explanation below x-axis (smaller, non-bold)
ax.text(
    0.5,
    -0.10,
    "[(M,K) = input matrix size, (K,N) = weight matrix size]",
    transform=ax.transAxes,
    ha="center",
    va="top",
    fontsize=9,
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
header_labels = {"Dtype (Math Fidelity)", "Device"}
for text in legend.get_texts():
    if text.get_text() in header_labels:
        text.set_weight("bold")

IMG_DIR.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(IMG_DIR / "utilization_comparison.png", dpi=300, bbox_inches="tight")
print("✓ Utilization scatter plot saved: tech_reports/GEMM_FLOPS/images/utilization_comparison.png")
plt.close()
