#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Utilization Scatter Plot - Device-based utilization comparison between N150 and P150

One subplot per benchmark mode (program config). No stitching across modes.

Usage:
1. Run the benchmark via run_bench.sh on both devices
2. CSVs are placed in tech_reports/GEMM_FLOPS/data/{wh,bh}.csv
3. Run this script from the tt-metal root directory
"""

from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_GEMM_FLOPS_DIR = Path(__file__).resolve().parent
if str(_GEMM_FLOPS_DIR) not in sys.path:
    sys.path.insert(0, str(_GEMM_FLOPS_DIR))
from benchmark_modes import MODE_DISPLAY, MODE_ORDER as ALL_MODE_ORDER, add_shape_column, normalize_modes

DATA_DIR = _GEMM_FLOPS_DIR / "data"
IMG_DIR = _GEMM_FLOPS_DIR / "images"

DEVICE_FILES = {
    "N150": DATA_DIR / "wh.csv",
    "P150": DATA_DIR / "bh.csv",
}

PLOT_MODE_ORDER = [mode for mode in ALL_MODE_ORDER if mode != "oob"] + ["oob"]

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


def find_util_col(df, kind):
    """Find the utilization column that has data. kind is 'Host' or 'Device'."""
    prefix = f"{kind} based utilization"
    for col in df.columns:
        if col.startswith(prefix) and "user selected grid" in col and df[col].notna().any():
            return col
    return None


def get_best_utilization_by_shape(df_slice):
    """Pick the highest-utilization row per (m, k, n) shape."""
    best_rows = []
    df_slice = df_slice.dropna(subset=["device_utilization"])
    for _, group in df_slice.groupby("shape", sort=True):
        best_rows.append(group.loc[group["device_utilization"].idxmax()])
    if not best_rows:
        return pd.DataFrame()
    return pd.DataFrame(best_rows).sort_values("matrix_elements")


def load_device_data(path, device_name):
    """Load and preprocess a device CSV."""
    df = safe_read_csv(path)
    if df.empty:
        return df
    df["device"] = device_name

    util_col = find_util_col(df, "Device") or find_util_col(df, "Host")
    if util_col is None:
        print(f"WARNING: No utilization column found in {path}")
    else:
        df["device_utilization"] = pd.to_numeric(df[util_col], errors="coerce")
    df = add_shape_column(df)
    return normalize_modes(df)


def get_dtype_label(row):
    dtype = str(row["dtype"]).split(".")[-1]
    fidelity = str(row["math_fidelity"]).split(".")[-1]
    return f"{dtype}-{fidelity}"


def build_legend_elements():
    """Shared legend for dtype and device encodings."""
    legend_elements = [
        Line2D([0], [0], color="none", marker="", linestyle="", label="Dtype (Math Fidelity)"),
    ]
    for dtype_label, color in dtype_configs:
        parts = dtype_label.split("-")
        formatted_label = f"{parts[0]} ({parts[1]})"
        legend_elements.append(Line2D([0], [0], color=color, linewidth=3, label=formatted_label))

    legend_elements.append(Line2D([0], [0], color="none", marker="", linestyle="", label=""))
    legend_elements.append(Line2D([0], [0], color="none", marker="", linestyle="", label="Device"))
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
    return legend_elements


def plot_mode_panel(ax, df_mode):
    """Plot utilization curves for one benchmark mode."""
    for dtype_label, dtype_color in dtype_configs:
        for device_name, marker, linestyle in device_configs:
            subset = df_mode[(df_mode["dtype_label"] == dtype_label) & (df_mode["device"] == device_name)]
            if subset.empty:
                continue

            subset_sorted = get_best_utilization_by_shape(subset)
            if subset_sorted.empty:
                continue
            if device_name == "N150":
                markerfacecolor = "white"
                markeredgewidth = 2
            else:
                markerfacecolor = dtype_color
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

    ax.set_xscale("log")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Utilization (%)", fontsize=11, fontweight="bold")


# Load data
df_n150 = load_device_data(DEVICE_FILES["N150"], "N150")
df_p150 = load_device_data(DEVICE_FILES["P150"], "P150")
df = pd.concat([df_n150, df_p150], ignore_index=True)

if df.empty:
    print("ERROR: No data available for any device. Exiting.")
    raise SystemExit(1)

if "device_utilization" not in df.columns or df["device_utilization"].isna().all():
    print("ERROR: No utilization data found for any device. Exiting.")
    raise SystemExit(1)

if df["use_trace"].dtype == object:
    df["use_trace"] = df["use_trace"].astype(str).str.lower() == "true"
df = df[df["use_trace"] == False].copy()

df["dtype_label"] = df.apply(get_dtype_label, axis=1)
df["matrix_elements"] = df["m"] * df["k"] * df["n"]

available_modes = [mode for mode in PLOT_MODE_ORDER if mode in df["mode"].values and not df[df["mode"] == mode].empty]
if not available_modes:
    print("ERROR: No mode data available after filtering. Exiting.")
    raise SystemExit(1)

n_modes = len(available_modes)
fig, axes = plt.subplots(n_modes, 1, figsize=(14, 5.5 * n_modes), sharex=True)
if n_modes == 1:
    axes = [axes]

for ax, mode in zip(axes, available_modes):
    df_mode = df[df["mode"] == mode]
    plot_mode_panel(ax, df_mode)
    ax.set_title(
        f"Mode: {MODE_DISPLAY.get(mode, mode)}",
        fontsize=12,
        fontweight="bold",
        pad=8,
    )

axes[-1].set_xlabel("Total Matrix Elements (M × K × N)", fontsize=12, fontweight="bold")
fig.text(
    0.5,
    0.02,
    "[(M,K) = input matrix size, (K,N) = weight matrix size]",
    ha="center",
    va="bottom",
    fontsize=9,
)

fig.suptitle(
    "Utilization Comparison: N150 (Wormhole) vs P150 (Blackhole)",
    fontsize=16,
    fontweight="bold",
    y=0.995,
)
fig.subplots_adjust(top=0.94, bottom=0.06, hspace=0.28)

legend = axes[0].legend(
    handles=build_legend_elements(),
    loc="upper right",
    fontsize=10,
    framealpha=0.95,
    edgecolor="black",
    handlelength=3.5,
)
header_labels = {"Dtype (Math Fidelity)", "Device"}
for text in legend.get_texts():
    if text.get_text() in header_labels:
        text.set_weight("bold")

IMG_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(IMG_DIR / "utilization_comparison.png", dpi=300, bbox_inches="tight")
print("✓ Utilization scatter plot saved: tech_reports/GEMM_FLOPS/images/utilization_comparison.png")
plt.close()
