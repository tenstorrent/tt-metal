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
from matplotlib.lines import Line2D

dtype_configs = [
    ("BFLOAT4_B_LoFi", "BFLOAT4_B (LoFi)", "#2ca02c"),  # Green
    ("BFLOAT8_B_HiFi2", "BFLOAT8_B (HiFi2)", "#ff7f0e"),  # Orange
    ("BFLOAT16_HiFi4", "BFLOAT16 (HiFi4)", "#1f77b4"),  # Blue
]

df_n150 = pd.read_csv("tech_reports/GEMM_FLOPS/n150-manual.csv")
df_n150["source"] = "n150"
df_p150 = pd.read_csv("tech_reports/GEMM_FLOPS/p150-manual.csv")
df_p150["source"] = "p150"

if "TFLOPs (avg)" in df_n150.columns:
    df_n150.rename(columns={"TFLOPs (avg)": "tflops"}, inplace=True)
if "TFLOPs (avg)" in df_p150.columns:
    df_p150.rename(columns={"TFLOPs (avg)": "tflops"}, inplace=True)

df_n150["tflops"] = pd.to_numeric(df_n150["tflops"], errors="coerce")
df_p150["tflops"] = pd.to_numeric(df_p150["tflops"], errors="coerce")

df = pd.concat([df_n150, df_p150], ignore_index=True)
df["dtype_fidelity"] = (
    df["dtype"].astype(str).str.replace("DataType.", "")
    + "_"
    + df["math_fidelity"].astype(str).str.replace("MathFidelity.", "")
)
df["matrix_elements"] = df["m"] * df["k"] * df["n"]

if df["use_trace"].dtype == object:
    df["use_trace"] = df["use_trace"].astype(str).str.lower() == "true"

for source in ["n150", "p150"]:
    fig, ax = plt.subplots(figsize=(16, 10))
    device_data = df[df["source"] == source].copy()

    for dtype_fidelity, dtype_label, color in dtype_configs:
        traced_perf = []
        nontraced_perf = []
        df_slice = device_data[device_data["dtype_fidelity"] == dtype_fidelity]

        for matrix_size in sorted(df_slice["matrix_elements"].unique()):
            size_group = df_slice[df_slice["matrix_elements"] == matrix_size]

            traced_group = size_group[size_group["use_trace"] == True]
            if not traced_group.empty:
                best_traced_tflops = traced_group.groupby(["m", "k", "n"])["tflops"].max().max()
                traced_perf.append((matrix_size, best_traced_tflops))

            nontraced_group = size_group[size_group["use_trace"] == False]
            if not nontraced_group.empty:
                best_nontraced_tflops = nontraced_group.groupby(["m", "k", "n"])["tflops"].max().max()
                nontraced_perf.append((matrix_size, best_nontraced_tflops))

        if not traced_perf or not nontraced_perf:
            continue

        traced_x, traced_y = zip(*traced_perf)
        nontraced_x, nontraced_y = zip(*nontraced_perf)

        ax.plot(
            traced_x, traced_y, color=color, linestyle="-", linewidth=2.5, label=f"{dtype_label} (Traced)", zorder=3
        )
        ax.scatter(traced_x, traced_y, color=color, marker="^", s=120, edgecolors="black", linewidths=1.2, zorder=4)

        ax.plot(
            nontraced_x,
            nontraced_y,
            color=color,
            linestyle="--",
            linewidth=2.5,
            label=f"{dtype_label} (Non-traced)",
            zorder=3,
        )
        ax.scatter(
            nontraced_x,
            nontraced_y,
            color=color,
            marker="v",
            s=120,
            facecolors="none",
            edgecolors=color,
            linewidths=2,
            zorder=4,
        )

        for i, (x, y_trace) in enumerate(zip(traced_x, traced_y)):
            if i < len(nontraced_y):
                y_nontrace = nontraced_y[i]
                ratio = y_trace / y_nontrace
                y_pos = max(y_trace, y_nontrace)
                ax.annotate(
                    f"{ratio:.2f}×",
                    (x, y_pos),
                    xytext=(0, 12),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.85, linewidth=1),
                    zorder=5,
                )

    ax.set_xscale("log")
    ax.text(
        0.5,
        -0.08,
        "Total Matrix Elements (M × K × N)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
    )
    ax.text(
        0.5,
        -0.12,
        "[(M,K) = input matrix size, (K,N) = weight matrix size]",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )
    ax.set_ylabel("Performance (TFLOPs)", fontsize=14, fontweight="bold", labelpad=10)

    device_name = "N150 (Wormhole)" if source == "n150" else "P150 (Blackhole)"
    fig.suptitle(f"Performance Comparison: {device_name}", fontsize=18, fontweight="bold", y=0.98)
    ax.set_title(
        "Traced vs Non-Traced Execution Performance (All Matrix Sizes)", fontsize=14, pad=10, fontweight="bold"
    )

    ax.grid(True, which="both", linestyle="--", alpha=0.4, zorder=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=11)

    legend_elements = []

    # Dtype section header
    legend_elements.append(Line2D([0], [0], color="none", label=r"$\mathbf{Dtype\ (Math\ Fidelity)}$"))

    # Add each dtype with its color
    for dtype_fidelity, dtype_label, color in dtype_configs:
        legend_elements.append(Line2D([0], [0], color=color, linewidth=4, label=dtype_label))

    # Spacer
    legend_elements.append(Line2D([0], [0], color="none", label=""))

    # Execution type section header
    legend_elements.append(Line2D([0], [0], color="none", label=r"$\mathbf{Execution\ Type}$"))

    # Traced (solid line, filled triangles)
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
            label="Traced",
        )
    )

    # Non-traced (dashed line, hollow triangles)
    legend_elements.append(
        Line2D(
            [0],
            [0],
            color="gray",
            linewidth=3,
            linestyle="--",
            marker="v",
            markersize=10,
            markerfacecolor="none",
            markeredgecolor="gray",
            markeredgewidth=2.5,
            label="Non-traced",
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
    plt.savefig(f"tech_reports/GEMM_FLOPS/images/trace_comparison_{source}.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved: trace_comparison_{source}.png")

print("\n✅ Tracing comparison plots complete!")
