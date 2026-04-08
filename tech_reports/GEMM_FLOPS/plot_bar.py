# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
1. Generate performance data using manually selected GEMM configurations
2. Rename output files to n150-manual.csv and p150-manual.csv
3. Place the CSV files in tech_reports/GEMM_FLOPS/
4. Run this script from the tt-metal root directory
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Load N150 and P150 data
df_n150 = pd.read_csv("tech_reports/GEMM_FLOPS/n150-manual.csv")
df_n150["source"] = "N150"

df_p150 = pd.read_csv("tech_reports/GEMM_FLOPS/p150-manual.csv")
df_p150["source"] = "P150"

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

df = pd.concat([df_n150, df_p150], ignore_index=True)

# Filter for square matrices and get best performance per (m, source, dtype_fidelity)
df_square = df[df["k"] == df["n"]].copy()
best_data = []
for (m_val, source, dtype_fidelity), group in df_square.groupby(["m", "source", "dtype_fidelity"]):
    best_row = group.loc[group["tflops"].idxmax()]
    best_data.append(best_row)

df_best = pd.DataFrame(best_data)

dtype_configs = [
    ("BFLOAT16_HiFi4", "BFLOAT16 (HiFi4)"),
    ("BFLOAT8_B_HiFi2", "BFLOAT8_B (HiFi2)"),
    ("BFLOAT4_B_LoFi", "BFLOAT4_B (LoFi)"),
]

# Pair N150 and P150 M values for x-axis labels
n150_m_values = sorted(df_best[df_best["source"] == "N150"]["m"].unique())
p150_m_values = sorted(df_best[df_best["source"] == "P150"]["m"].unique())

combined_labels = []
all_m_values = []
for n150_m, p150_m in zip(n150_m_values, p150_m_values):
    combined_labels.append(f"{n150_m} / {p150_m}")
    all_m_values.append((n150_m, p150_m))

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

for pair_idx, (n150_m, p150_m) in enumerate(all_m_values):
    cluster_start = current_pos

    # N150 bars (lighter)
    for dtype_fidelity, label in dtype_configs:
        n150_data = df_best[
            (df_best["m"] == n150_m) & (df_best["source"] == "N150") & (df_best["dtype_fidelity"] == dtype_fidelity)
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
            (df_best["m"] == p150_m) & (df_best["source"] == "P150") & (df_best["dtype_fidelity"] == dtype_fidelity)
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
for pair_idx in range(len(all_m_values)):
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

# Legend - cleaner format like scatter plots
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

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

plt.subplots_adjust(left=0.03, right=0.97, bottom=0.12, top=0.93)
plt.xlim(min(positions) - bar_width * 2, max(positions) + bar_width * 2)
plt.savefig("tech_reports/GEMM_FLOPS/images/flops_by_matrix_size_and_type_sorted.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"✓ Bar chart saved!")
print(f"  - Plotted {len(combined_labels)} matrix size pairs")
