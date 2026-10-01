# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Performance scatter plot: TFLOPs comparison N150 vs P150.

Usage:
1. Run the benchmark via run_bench.sh on both devices
2. CSVs are placed in tech_reports/GEMM_FLOPS/data/{wh,bh}.csv
3. Run this script from the tt-metal root directory
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DATA_DIR = Path("tech_reports/GEMM_FLOPS/data")
IMG_DIR = Path("tech_reports/GEMM_FLOPS/images")

DEVICE_FILES = {
    "N150": DATA_DIR / "wh.csv",
    "P150": DATA_DIR / "bh.csv",
}

BASE_SHAPE_COLUMNS = ["base_m", "base_k", "base_n"]


def safe_read_csv(path):
    """Return the CSV as a DataFrame, or an empty DataFrame if the file is missing."""
    if path.exists():
        return pd.read_csv(path)
    print(f"WARNING: {path} not found — skipping that device.")
    return pd.DataFrame()


def parse_grid_size(raw):
    cleaned = str(raw).strip("() ")
    grid_x, grid_y = [int(x.strip()) for x in cleaned.split(",")]
    return grid_x, grid_y


def add_base_shape_columns(df):
    """Ensure base_m/base_k/base_n exist while preserving full scaled m/k/n."""
    if not all(col in df.columns for col in BASE_SHAPE_COLUMNS):
        grid_dims = df["grid_size"].apply(parse_grid_size)
        df["base_m"] = [m // grid_y for m, (_, grid_y) in zip(df["m"], grid_dims)]
        df["base_k"] = [k // grid_x for k, (grid_x, _) in zip(df["k"], grid_dims)]
        df["base_n"] = [n // grid_x for n, (grid_x, _) in zip(df["n"], grid_dims)]
    df["base_shape"] = list(zip(df["base_m"], df["base_k"], df["base_n"]))
    return df


def load_and_prepare(path, source):
    """Load CSV and prepare derived columns."""
    df = safe_read_csv(path)
    if df.empty:
        return df
    df["source"] = source
    # Use best performance across all tuned modes
    if "mode" in df.columns:
        df = df[df["mode"] != "oob"].copy()
    if "TFLOPs (avg)" in df.columns:
        df.rename(columns={"TFLOPs (avg)": "tflops"}, inplace=True)
    df["tflops"] = pd.to_numeric(df["tflops"], errors="coerce")
    df["dtype_fidelity"] = (
        df["dtype"].astype(str).str.replace("DataType.", "")
        + "_"
        + df["math_fidelity"].astype(str).str.replace("MathFidelity.", "")
    )
    df["matrix_elements"] = df["m"] * df["k"] * df["n"]
    df = add_base_shape_columns(df)
    return df


def get_best_by_base_shape(df_slice):
    best_rows = []
    for _, group in df_slice.groupby("base_shape", sort=True):
        best_rows.append(group.loc[group["tflops"].idxmax()])
    if not best_rows:
        return pd.DataFrame()
    return pd.DataFrame(best_rows).sort_values("matrix_elements")


frames = []
for source, path in DEVICE_FILES.items():
    loaded = load_and_prepare(path, source)
    if not loaded.empty:
        frames.append(loaded)

if not frames:
    print("ERROR: No data available for any device. Exiting.")
    raise SystemExit(1)

df = pd.concat(frames, ignore_index=True)

# dtype-fidelity configurations to plot with colors
dtype_configs = [
    ("BFLOAT4_B_LoFi", "#2ca02c", "BFLOAT4_B (LoFi)"),  # Green
    ("BFLOAT8_B_HiFi2", "#ff7f0e", "BFLOAT8_B (HiFi2)"),  # Orange
    ("BFLOAT16_HiFi4", "#1f77b4", "BFLOAT16 (HiFi4)"),  # Blue
]

# Create figure
fig, ax = plt.subplots(figsize=(18, 10))

for dtype_fidelity, color, label_short in dtype_configs:
    # P150 Data
    p150_data = df[(df["dtype_fidelity"] == dtype_fidelity) & (df["source"] == "P150")].copy()

    if not p150_data.empty:
        # Get best (max tflops) for each base shape; plot its scaled matrix size.
        p150_best = get_best_by_base_shape(p150_data)

        # Plot P150: solid line with filled upward triangles
        ax.plot(
            p150_best["matrix_elements"],
            p150_best["tflops"],
            color=color,
            alpha=0.8,
            linewidth=3.0,
            linestyle="-",
            marker="^",
            markersize=10,
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=1.5,
            label=f"{label_short} (P150)",
            zorder=5,
        )

    # N150 Data
    n150_data = df[(df["dtype_fidelity"] == dtype_fidelity) & (df["source"] == "N150")].copy()

    if not n150_data.empty:
        # Get best (max tflops) for each base shape; plot its scaled matrix size.
        n150_best = get_best_by_base_shape(n150_data)

        # Plot N150: dashed line with hollow downward triangles
        ax.plot(
            n150_best["matrix_elements"],
            n150_best["tflops"],
            color=color,
            alpha=0.8,
            linewidth=3.0,
            linestyle="--",
            marker="v",
            markersize=10,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=2.5,
            label=f"{label_short} (N150)",
            zorder=5,
        )

# Configure axes
ax.set_xscale("log")
ax.set_yscale("linear")
ax.grid(True, which="both", linestyle=":", linewidth=0.8, alpha=0.4, color="gray")
ax.set_axisbelow(True)

# Axis labels
ax.set_xlabel("Total Matrix Elements (M × K × N)", fontsize=15, fontweight="bold", labelpad=10)
ax.set_ylabel("Performance (TFLOPs)", fontsize=15, fontweight="bold", labelpad=10)

# Add explanation below x-axis
ax.text(
    0.5,
    -0.08,
    "where (M,K) = input matrix size, (K,N) = weight matrix size",
    transform=ax.transAxes,
    ha="center",
    va="top",
    fontsize=10,
    style="italic",
)

# Title
fig.suptitle("Performance Comparison: N150 (Wormhole) vs P150 (Blackhole)", fontsize=18, fontweight="bold", y=0.98)
ax.set_title(
    "TFLOPs vs Matrix Size for Different Data Types and Math Fidelities", fontsize=14, pad=10, fontweight="bold"
)

# Create custom legend
legend_elements = []

# Dtype section
legend_elements.append(Line2D([0], [0], color="none", label=r"$\mathbf{Dtype\ (Math\ Fidelity)}$"))
for dtype_fidelity, color, label_short in dtype_configs:
    legend_elements.append(Line2D([0], [0], color=color, linewidth=4, label=label_short))

legend_elements.append(Line2D([0], [0], color="none", label=""))

# Device section
legend_elements.append(Line2D([0], [0], color="none", label=r"$\mathbf{Device}$"))
legend_elements.append(
    Line2D(
        [0],
        [0],
        color="gray",
        linewidth=3,
        linestyle="-",
        marker="^",
        markersize=10,
        markerfacecolor="gray",
        markeredgecolor="black",
        markeredgewidth=1.5,
        label="P150",
    )
)
legend_elements.append(
    Line2D(
        [0],
        [0],
        color="gray",
        linewidth=3,
        linestyle=(0, (5, 5)),
        marker="v",
        markersize=10,
        markerfacecolor="none",
        markeredgecolor="gray",
        markeredgewidth=2.5,
        label="N150",
    )
)

ax.legend(
    handles=legend_elements,
    loc="upper left",
    fontsize=12,
    framealpha=0.95,
    edgecolor="black",
    fancybox=True,
    shadow=True,
    handlelength=3.5,
)

IMG_DIR.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(IMG_DIR / "flops_vs_matrix_elements_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

print("✓ Performance scatter plot saved!")
n150_count = df[df["source"] == "N150"].groupby(["m", "k", "n"]).ngroups
p150_count = df[df["source"] == "P150"].groupby(["m", "k", "n"]).ngroups
print(f"  - N150: {n150_count} unique matrix sizes")
print(f"  - P150: {p150_count} unique matrix sizes")
print(f"  - Configurations plotted: BFLOAT4_B_LoFi, BFLOAT8_B_HiFi2, BFLOAT16_HiFi4")
