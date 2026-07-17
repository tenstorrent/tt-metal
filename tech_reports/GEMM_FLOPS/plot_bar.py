# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Bar chart: paired N150 vs P150 TFLOPs comparison for square-ish matrices.

Usage:
1. Run the benchmark via run_bench.sh on both devices
2. CSVs are placed in tech_reports/GEMM_FLOPS/data/{wh,bh}.csv
3. Run this script from the tt-metal root directory
"""

from pathlib import Path

import matplotlib.colors as mcolors
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

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
    df = add_base_shape_columns(df)
    return df


# Load data
df_n150 = load_and_prepare(DEVICE_FILES["N150"], "N150")
df_p150 = load_and_prepare(DEVICE_FILES["P150"], "P150")

if df_n150.empty or df_p150.empty:
    missing = []
    if df_n150.empty:
        missing.append("wh.csv")
    if df_p150.empty:
        missing.append("bh.csv")
    print(
        f"NOTE: plot_bar.py produces a paired N150/P150 comparison chart and "
        f"requires both device CSVs. Missing: {', '.join(missing)}. Skipping."
    )
    raise SystemExit(0)

df = pd.concat([df_n150, df_p150], ignore_index=True)

# Filter for square base matrices and get best performance per base shape/source/dtype.
df_square = df[df["base_k"] == df["base_n"]].copy()
best_data = []
for (_, source, dtype_fidelity), group in df_square.groupby(["base_shape", "source", "dtype_fidelity"]):
    best_row = group.loc[group["tflops"].idxmax()]
    best_data.append(best_row)

df_best = pd.DataFrame(best_data)

dtype_configs = [
    ("BFLOAT16_HiFi4", "BFLOAT16 (HiFi4)"),
    ("BFLOAT8_B_HiFi2", "BFLOAT8_B (HiFi2)"),
    ("BFLOAT4_B_LoFi", "BFLOAT4_B (LoFi)"),
]

# Pair N150 and P150 by base shape; use scaled M values only for display.
n150_shapes = set(df_best[df_best["source"] == "N150"]["base_shape"])
p150_shapes = set(df_best[df_best["source"] == "P150"]["base_shape"])
paired_base_shapes = sorted(n150_shapes & p150_shapes)

combined_labels = []
all_base_shapes = []
for base_shape in paired_base_shapes:
    n150_m = df_best[(df_best["source"] == "N150") & (df_best["base_shape"] == base_shape)]["m"].iloc[0]
    p150_m = df_best[(df_best["source"] == "P150") & (df_best["base_shape"] == base_shape)]["m"].iloc[0]
    combined_labels.append(f"{n150_m} / {p150_m}")
    all_base_shapes.append(base_shape)

if not all_base_shapes:
    print("ERROR: No paired base matrix shapes available for N150/P150 comparison. Exiting.")
    raise SystemExit(1)

print(f"✓ Matrix sizes: {len(combined_labels)} pairs")
print(f"✓ Total configurations: {len(df_best)}")

fig, ax = plt.subplots(figsize=(max(20, len(combined_labels) * 3.5), 14))

# Bar spacing configuration
bar_width = 0.05
gap_within_dtype = 0.02
gap_between_n150_p150 = 0.10
gap_between_clusters = 0.30

dtype_color_map = {
    "BFLOAT16_HiFi4": "#1f77b4",  # Blue (same as scatter plots)
    "BFLOAT8_B_HiFi2": "#ff7f0e",  # Orange (same as scatter plots)
    "BFLOAT4_B_LoFi": "#2ca02c",  # Green (same as scatter plots)
}

positions = []
heights = []
colors_list = []
cluster_centers = []
bar_info = []

current_pos = 0

for pair_idx, base_shape in enumerate(all_base_shapes):
    cluster_start = current_pos

    # N150 bars (lighter)
    for dtype_fidelity, label in dtype_configs:
        n150_data = df_best[
            (df_best["base_shape"] == base_shape)
            & (df_best["source"] == "N150")
            & (df_best["dtype_fidelity"] == dtype_fidelity)
        ]
        if len(n150_data) > 0:
            val = n150_data["tflops"].values[0]
            positions.append(current_pos)
            heights.append(val)
            colors_list.append(mcolors.to_rgba(dtype_color_map[dtype_fidelity], alpha=0.7))
            bar_info.append(
                {"x": current_pos, "height": val, "pair_idx": pair_idx, "dtype": dtype_fidelity, "source": "N150"}
            )
        current_pos += bar_width + gap_within_dtype

    current_pos += gap_between_n150_p150

    # P150 bars (darker)
    for dtype_fidelity, label in dtype_configs:
        p150_data = df_best[
            (df_best["base_shape"] == base_shape)
            & (df_best["source"] == "P150")
            & (df_best["dtype_fidelity"] == dtype_fidelity)
        ]
        if len(p150_data) > 0:
            val = p150_data["tflops"].values[0]
            positions.append(current_pos)
            heights.append(val)
            colors_list.append(mcolors.to_rgba(dtype_color_map[dtype_fidelity], alpha=1.0))
            bar_info.append(
                {"x": current_pos, "height": val, "pair_idx": pair_idx, "dtype": dtype_fidelity, "source": "P150"}
            )
        current_pos += bar_width + gap_within_dtype

    cluster_centers.append((cluster_start + current_pos) / 2)
    current_pos += gap_between_clusters

ax.bar(positions, heights, width=bar_width, color=colors_list, edgecolor="black", linewidth=0.5)

# Add performance multiplier annotations (baseline: N150 BFLOAT16-HiFi4 per matrix size)
max_height = max(heights)
for pair_idx in range(len(all_base_shapes)):
    baseline = None
    for info in bar_info:
        if info["pair_idx"] == pair_idx and info["dtype"] == "BFLOAT16_HiFi4" and info["source"] == "N150":
            baseline = info["height"]
            break

    if baseline and baseline > 0:
        for info in bar_info:
            if info["pair_idx"] == pair_idx and info["height"] > 0:
                ratio = info["height"] / baseline
                ax.annotate(
                    f"{ratio:.2f}x",
                    (info["x"], info["height"] + max_height * 0.02),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8, linewidth=1),
                )

ax.set_xlabel("Base Matrix Dimension M (Input Rows): N150 / P150", fontsize=13, fontweight="bold", labelpad=10)
ax.set_ylabel("Performance (TFLOPs)", fontsize=14, fontweight="bold")

fig.suptitle("Performance Comparison: N150 (Wormhole) vs P150 (Blackhole)", fontsize=18, fontweight="bold", y=0.98)
ax.set_title(
    "TFLOPs vs Matrix Size for Different Data Types and Math Fidelities", fontsize=14, pad=10, fontweight="bold"
)

ax.set_xticks(cluster_centers)
ax.set_xticklabels(combined_labels, rotation=45, ha="right", fontsize=11)
ax.tick_params(axis="x", which="both", bottom=False, top=False)

ax.grid(True, axis="y", linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
ax.set_ylim(0, max_height * 1.2)

# Legend
legend_elements = []

# Dtype section header
legend_elements.append(Line2D([0], [0], color="none", label=r"$\mathbf{Dtype\ (Math\ Fidelity)}$"))

# Add each dtype with its color
for dtype_fidelity, label in dtype_configs:
    legend_elements.append(Line2D([0], [0], color=dtype_color_map[dtype_fidelity], linewidth=4, label=label))

# Spacer
legend_elements.append(Line2D([0], [0], color="none", label=""))

# Device section header
legend_elements.append(Line2D([0], [0], color="none", label=r"$\mathbf{Device}$"))

# N150 (lighter)
legend_elements.append(
    Rectangle((0, 0), 1, 1, facecolor="gray", alpha=0.7, edgecolor="black", linewidth=1, label="N150")
)

# P150 (darker)
legend_elements.append(
    Rectangle((0, 0), 1, 1, facecolor="gray", alpha=1.0, edgecolor="black", linewidth=1, label="P150")
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
plt.subplots_adjust(left=0.03, right=0.97, bottom=0.12, top=0.93)
plt.xlim(min(positions) - bar_width * 2, max(positions) + bar_width * 2)
plt.savefig(IMG_DIR / "flops_by_matrix_size_and_type_sorted.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"✓ Bar chart saved!")
print(f"  - Plotted {len(combined_labels)} matrix size pairs")
