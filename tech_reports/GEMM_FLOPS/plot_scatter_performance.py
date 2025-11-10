# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Load N150 and P150 data
df_n150 = pd.read_csv("tech_reports/GEMM_FLOPS/n150-sweep.csv")
df_n150["source"] = "N150"

df_p150 = pd.read_csv("tech_reports/GEMM_FLOPS/p150-sweep.csv")
df_p150["source"] = "P150"

# Standardize column names
if "TFLOPs (avg)" in df_n150.columns:
    df_n150.rename(columns={"TFLOPs (avg)": "tflops"}, inplace=True)
if "TFLOPs (avg)" in df_p150.columns:
    df_p150.rename(columns={"TFLOPs (avg)": "tflops"}, inplace=True)

# Convert tflops to numeric
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

# Calculate matrix elements (M × K × N)
df_n150["matrix_elements"] = df_n150["m"] * df_n150["k"] * df_n150["n"]
df_p150["matrix_elements"] = df_p150["m"] * df_p150["k"] * df_p150["n"]

# Combine dataframes
df = pd.concat([df_n150, df_p150], ignore_index=True)

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
        # Get best (max tflops) for each matrix size
        p150_best = (
            p150_data.groupby("matrix_elements")
            .agg({"tflops": "max", "m": "first", "k": "first", "n": "first"})
            .reset_index()
            .sort_values("matrix_elements")
        )

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
        # Get best (max tflops) for each matrix size
        n150_best = (
            n150_data.groupby("matrix_elements")
            .agg({"tflops": "max", "m": "first", "k": "first", "n": "first"})
            .reset_index()
            .sort_values("matrix_elements")
        )

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
ax.set_xlabel("Total Matrix Elements (m × k × n)", fontsize=15, fontweight="bold", labelpad=10)
ax.set_ylabel("Performance (TFLOPs)", fontsize=15, fontweight="bold", labelpad=10)

# Add explanation below x-axis
ax.text(
    0.5,
    -0.08,
    "where (m,k) = input matrix size, (k,n) = weight matrix size",
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
legend_elements.append(Line2D([0], [0], color="none", label=r"$\mathbf{Dtype\ (Math\ Fidelity):}$"))
for dtype_fidelity, color, label_short in dtype_configs:
    legend_elements.append(Line2D([0], [0], color=color, linewidth=4, label=f"  {label_short}"))

legend_elements.append(Line2D([0], [0], color="none", label=""))

# Device section
legend_elements.append(Line2D([0], [0], color="none", label=r"$\mathbf{Device:}$"))
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
        label="  P150",
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
        label="  N150",
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

plt.tight_layout()
plt.savefig("tech_reports/GEMM_FLOPS/images/flops_vs_matrix_elements_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

print("✓ Performance scatter plot saved!")
print(f"  - N150: {df[df['source'] == 'N150'].groupby(['m','k','n']).ngroups} unique matrix sizes")
print(f"  - P150: {df[df['source'] == 'P150'].groupby(['m','k','n']).ngroups} unique matrix sizes")
print(f"  - Configurations plotted: BFLOAT4_B_LoFi, BFLOAT8_B_HiFi2, BFLOAT16_HiFi4")
