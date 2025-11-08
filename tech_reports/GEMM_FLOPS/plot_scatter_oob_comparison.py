#!/usr/bin/env python3
"""
Generate scatter plots comparing Out-of-Box (OOB) vs Hand-tuned performance
for N150 and P150 devices.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Dtype configurations matching hand-tuned fidelities
# Both OOB and hand-tuned now use: BFLOAT16-HiFi4, BFLOAT8_B-HiFi2, BFLOAT4_B-LoFi
dtype_configs = [
    ("BFLOAT4_B-LoFi", "#1f77b4"),  # Blue
    ("BFLOAT8_B-HiFi2", "#2ca02c"),  # Green
    ("BFLOAT16-HiFi4", "#ff7f0e"),  # Orange
]


def load_and_prepare_data(oob_file, tuned_file, device_name):
    """Load OOB and hand-tuned data, standardize columns"""
    df_oob = pd.read_csv(oob_file)
    df_tuned = pd.read_csv(tuned_file)

    # Filter out traced data for fair comparison (DRAM vs L1 without trace overhead)
    df_oob = df_oob[df_oob["use_trace"] == False]
    df_tuned = df_tuned[df_tuned["use_trace"] == False]

    # Standardize column names
    for df in [df_oob, df_tuned]:
        if "TFLOPs (avg)" in df.columns:
            df.rename(columns={"TFLOPs (avg)": "tflops"}, inplace=True)

        # Create dtype_fidelity column
        df["dtype_fidelity"] = (
            df["dtype"].str.replace("DataType.", "", regex=False).str.upper()
            + "-"
            + df["math_fidelity"].str.replace("MathFidelity.", "", regex=False)
        )

        # Calculate matrix elements
        df["matrix_elements"] = df["m"] * df["k"] * df["n"]

    # Add labels
    df_oob["config_type"] = "OOB"
    df_tuned["config_type"] = "Hand-tuned"

    return df_oob, df_tuned


def plot_oob_comparison(device_name, oob_file, tuned_file):
    """Generate comparison plot for a single device"""

    df_oob, df_tuned = load_and_prepare_data(oob_file, tuned_file, device_name)

    fig, ax = plt.subplots(figsize=(16, 10))

    # Plot each dtype
    for dtype_fidelity, color in dtype_configs:
        # Get data for this dtype
        oob_data = df_oob[df_oob["dtype_fidelity"] == dtype_fidelity]
        tuned_data = df_tuned[df_tuned["dtype_fidelity"] == dtype_fidelity]

        # Group by matrix size and get best performance
        oob_grouped = oob_data.groupby("matrix_elements")["tflops"].max().reset_index().sort_values("matrix_elements")
        tuned_grouped = (
            tuned_data.groupby("matrix_elements")["tflops"].max().reset_index().sort_values("matrix_elements")
        )

        if len(oob_grouped) > 0:
            # Plot OOB (dashed line, hollow triangles)
            ax.plot(
                oob_grouped["matrix_elements"],
                oob_grouped["tflops"],
                color=color,
                linewidth=2.5,
                linestyle="--",
                marker="v",
                markersize=8,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=2,
                alpha=0.8,
            )

        if len(tuned_grouped) > 0:
            # Plot Hand-tuned (solid line, filled triangles)
            ax.plot(
                tuned_grouped["matrix_elements"],
                tuned_grouped["tflops"],
                color=color,
                linewidth=2.5,
                linestyle="-",
                marker="^",
                markersize=8,
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=1,
                alpha=0.8,
            )

            # Calculate and annotate speedup ratios for ALL comparisons
            for _, oob_row in oob_grouped.iterrows():
                matrix_elem = oob_row["matrix_elements"]
                oob_perf = oob_row["tflops"]

                # Find matching tuned performance
                tuned_row = tuned_grouped[tuned_grouped["matrix_elements"] == matrix_elem]
                if len(tuned_row) > 0:
                    tuned_perf = tuned_row["tflops"].values[0]
                    speedup = tuned_perf / oob_perf

                    # Annotate speedup (show all, even < 1.0x which means OOB is faster)
                    y_pos = max(tuned_perf, oob_perf)
                    ax.annotate(
                        f"{speedup:.2f}×",
                        xy=(matrix_elem, y_pos),
                        xytext=(0, 10),
                        textcoords="offset points",
                        ha="center",
                        fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                    )

    # Formatting
    ax.set_xscale("log")
    ax.grid(True, which="both", linestyle=":", linewidth=0.8, alpha=0.4, color="gray")
    ax.set_axisbelow(True)

    # Labels
    main_label = "Total Matrix Elements (m × k × n)"
    sub_label = "[(m,k) = input matrix size, (k,n) = weight matrix size]"
    ax.set_xlabel(f"{main_label}\n{sub_label}", fontsize=14, fontweight="bold", labelpad=10)
    ax.text(0.5, -0.12, sub_label, transform=ax.transAxes, fontsize=10, ha="center", va="top")

    ax.set_ylabel("Performance (TFLOPs)", fontsize=15, fontweight="bold", labelpad=10)

    # Title
    device_full = "N150 (Wormhole)" if device_name == "N150" else "P150 (Blackhole)"
    fig.suptitle(f"Performance Comparison: {device_full}", fontsize=18, fontweight="bold", y=0.98)
    ax.set_title("Out-of-Box vs Hand-Tuned Configuration Performance", fontsize=14, pad=10, fontweight="bold")

    # Custom legend
    legend_elements = []

    # Dtype section
    legend_elements.append(Line2D([0], [0], color="none", label=r"$\mathbf{Data\ Type\ (Math\ Fidelity):}$"))
    for dtype_label, color in dtype_configs:
        # Format: BFLOAT16 (HiFi4), BFLOAT8_B (HiFi2), BFLOAT4_B (LoFi)
        parts = dtype_label.split("-")
        dtype_part = parts[0].replace("_", r"\_")
        fidelity_part = parts[1]
        formatted_label = f"{dtype_part} ({fidelity_part})"
        legend_elements.append(Line2D([0], [0], color=color, linewidth=3, label=f"  {formatted_label}"))

    # Config type section
    legend_elements.append(Line2D([0], [0], color="none", label=""))
    legend_elements.append(Line2D([0], [0], color="none", label=r"$\mathbf{Configuration:}$"))
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
            label="  Hand-tuned (L1 SRAM)",
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
            label="  Out-of-Box (DRAM)",
        )
    )

    ax.legend(
        handles=legend_elements,
        loc="lower right",
        fontsize=12,
        framealpha=0.95,
        edgecolor="black",
        fancybox=True,
        shadow=True,
    )

    plt.tight_layout()
    output_file = f"tech_reports/GEMM_FLOPS/images/oob_comparison_{device_name.lower()}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✓ OOB comparison plot saved: {output_file}")
    plt.close()


# Generate plots for both devices
plot_oob_comparison("N150", "tech_reports/GEMM_FLOPS/n150-oob.csv", "tech_reports/GEMM_FLOPS/n150-sweep.csv")
plot_oob_comparison("P150", "tech_reports/GEMM_FLOPS/p150-oob.csv", "tech_reports/GEMM_FLOPS/p150-sweep.csv")

print("✓ All OOB comparison plots generated!")
