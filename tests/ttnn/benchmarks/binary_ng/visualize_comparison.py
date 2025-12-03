#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Visualize kernel time comparison between max_ab and max_abc strategies.

Creates multiple charts:
1. Scatter plot of kernel times
2. Bar chart of percentage differences
3. Histogram of differences
4. Heatmap by sharding configuration
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
import seaborn as sns

# Set style for better-looking plots
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


def load_and_merge_data(csv1_path, csv2_path):
    """Load both CSVs and merge them on configuration columns."""
    print(f"Loading {csv1_path}...")
    df1 = pd.read_csv(csv1_path)

    print(f"Loading {csv2_path}...")
    df2 = pd.read_csv(csv2_path)

    # Filter out error rows
    df1_clean = df1[df1["error"].isna() | (df1["error"] == "")].copy()
    df2_clean = df2[df2["error"].isna() | (df2["error"] == "")].copy()

    print(f"Rows after filtering errors: {len(df1_clean)} (max_ab), {len(df2_clean)} (max_abc)")

    # Merge on configuration columns
    config_cols = [
        "op_category",
        "op_type",
        "a_shape",
        "a_sharding",
        "a_cores",
        "b_shape",
        "b_sharding",
        "b_cores",
        "c_shape",
        "c_sharding",
        "c_cores",
    ]

    merged = pd.merge(df1_clean, df2_clean, on=config_cols, suffixes=("_max_ab", "_max_abc"), how="inner")

    print(f"Merged rows (common configurations): {len(merged)}")

    # Calculate differences
    merged["time_diff"] = merged["kernel_time_us_max_abc"] - merged["kernel_time_us_max_ab"]
    merged["time_diff_pct"] = (merged["time_diff"] / merged["kernel_time_us_max_ab"]) * 100

    # Add faster strategy indicator
    merged["faster_strategy"] = merged["time_diff"].apply(
        lambda x: "max_ab" if x > 0 else ("max_abc" if x < 0 else "tie")
    )

    return merged


def create_scatter_plot(merged_df, output_dir):
    """Create scatter plot comparing kernel times."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create scatter plot with color by faster strategy
    colors = {
        "max_ab": "#ff6b6b",  # Red (max_ab faster)
        "max_abc": "#4ecdc4",  # Teal (max_abc faster)
        "tie": "#95a5a6",  # Gray (tie)
    }

    for strategy, color in colors.items():
        mask = merged_df["faster_strategy"] == strategy
        if mask.any():
            ax.scatter(
                merged_df.loc[mask, "kernel_time_us_max_ab"],
                merged_df.loc[mask, "kernel_time_us_max_abc"],
                c=color,
                alpha=0.6,
                s=100,
                label=f"{strategy.upper()} faster" if strategy != "tie" else "Tie",
                edgecolors="black",
                linewidth=0.5,
            )

    # Add diagonal line (where both are equal)
    max_time = max(merged_df["kernel_time_us_max_ab"].max(), merged_df["kernel_time_us_max_abc"].max())
    ax.plot([0, max_time], [0, max_time], "k--", alpha=0.3, linewidth=2, label="Equal Performance")

    ax.set_xlabel("Kernel Time (μs) - max_ab Strategy", fontsize=12, fontweight="bold")
    ax.set_ylabel("Kernel Time (μs) - max_abc Strategy", fontsize=12, fontweight="bold")
    ax.set_title(
        "Kernel Time Comparison: max_ab vs max_abc\n(Points below diagonal = max_abc faster)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add statistics text box
    stats_text = (
        f"Total Configs: {len(merged_df)}\n"
        f"max_ab faster: {(merged_df['faster_strategy'] == 'max_ab').sum()}\n"
        f"max_abc faster: {(merged_df['faster_strategy'] == 'max_abc').sum()}\n"
        f"Ties: {(merged_df['faster_strategy'] == 'tie').sum()}"
    )
    ax.text(
        0.98,
        0.02,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    output_path = output_dir / "scatter_kernel_time_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {output_path}")
    plt.close()


def create_percentage_diff_chart(merged_df, output_dir):
    """Create bar chart of percentage differences."""
    # Sort by percentage difference
    sorted_df = merged_df.sort_values("time_diff_pct", ascending=True)

    fig, ax = plt.subplots(figsize=(14, 10))

    # Create color array based on sign
    colors = ["#4ecdc4" if x < 0 else "#ff6b6b" for x in sorted_df["time_diff_pct"]]

    # Create bars
    bars = ax.barh(
        range(len(sorted_df)), sorted_df["time_diff_pct"], color=colors, alpha=0.7, edgecolor="black", linewidth=0.5
    )

    # Add vertical line at 0
    ax.axvline(x=0, color="black", linestyle="-", linewidth=2)

    ax.set_xlabel("Performance Difference (%)\n← max_abc faster  |  max_ab faster →", fontsize=12, fontweight="bold")
    ax.set_ylabel("Configuration Index", fontsize=12, fontweight="bold")
    ax.set_title(
        "Kernel Time Performance Difference: max_abc vs max_ab\n(Negative = max_abc faster)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.grid(True, alpha=0.3, axis="x")

    # Create custom legend
    max_abc_patch = mpatches.Patch(color="#4ecdc4", label="max_abc faster", alpha=0.7)
    max_ab_patch = mpatches.Patch(color="#ff6b6b", label="max_ab faster", alpha=0.7)
    ax.legend(handles=[max_abc_patch, max_ab_patch], loc="best", fontsize=10)

    # Add statistics
    stats_text = (
        f"Mean Diff: {sorted_df['time_diff_pct'].mean():.2f}%\n"
        f"Median Diff: {sorted_df['time_diff_pct'].median():.2f}%\n"
        f"Std Dev: {sorted_df['time_diff_pct'].std():.2f}%\n"
        f"Range: [{sorted_df['time_diff_pct'].min():.2f}%, {sorted_df['time_diff_pct'].max():.2f}%]"
    )
    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    output_path = output_dir / "bar_percentage_differences.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {output_path}")
    plt.close()


def create_histogram(merged_df, output_dir):
    """Create histogram of time differences."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Histogram of absolute differences
    ax1.hist(merged_df["time_diff"], bins=50, color="#3498db", alpha=0.7, edgecolor="black")
    ax1.axvline(x=0, color="red", linestyle="--", linewidth=2, label="No difference")
    ax1.axvline(
        x=merged_df["time_diff"].mean(),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f'Mean: {merged_df["time_diff"].mean():.2f}μs',
    )
    ax1.axvline(
        x=merged_df["time_diff"].median(),
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f'Median: {merged_df["time_diff"].median():.2f}μs',
    )
    ax1.set_xlabel("Time Difference (μs)\n← max_abc faster  |  max_ab faster →", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Frequency (# Configs)", fontsize=11, fontweight="bold")
    ax1.set_title("Distribution of Kernel Time Differences\n(Absolute Time)", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Histogram of percentage differences
    ax2.hist(merged_df["time_diff_pct"], bins=50, color="#e74c3c", alpha=0.7, edgecolor="black")
    ax2.axvline(x=0, color="red", linestyle="--", linewidth=2, label="No difference")
    ax2.axvline(
        x=merged_df["time_diff_pct"].mean(),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f'Mean: {merged_df["time_diff_pct"].mean():.2f}%',
    )
    ax2.axvline(
        x=merged_df["time_diff_pct"].median(),
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f'Median: {merged_df["time_diff_pct"].median():.2f}%',
    )
    ax2.set_xlabel("Time Difference (%)\n← max_abc faster  |  max_ab faster →", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Frequency (# Configs)", fontsize=11, fontweight="bold")
    ax2.set_title("Distribution of Kernel Time Differences\n(Percentage)", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "histogram_time_differences.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {output_path}")
    plt.close()


def create_sharding_heatmap(merged_df, output_dir):
    """Create heatmap showing average time difference by sharding configuration."""
    # Create pivot table for C sharding vs A sharding
    pivot_data = merged_df.pivot_table(values="time_diff_pct", index="c_sharding", columns="a_sharding", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    im = ax.imshow(pivot_data.values, cmap="RdYlGn_r", aspect="auto", vmin=-20, vmax=20)

    # Set ticks and labels
    ax.set_xticks(range(len(pivot_data.columns)))
    ax.set_yticks(range(len(pivot_data.index)))
    ax.set_xticklabels(pivot_data.columns, fontsize=11)
    ax.set_yticklabels(pivot_data.index, fontsize=11)

    ax.set_xlabel("Tensor A Sharding", fontsize=12, fontweight="bold")
    ax.set_ylabel("Output Tensor C Sharding", fontsize=12, fontweight="bold")
    ax.set_title(
        "Average Performance Difference by Sharding Configuration\n(Red = max_ab faster, Green = max_abc faster)",
        fontsize=13,
        fontweight="bold",
        pad=20,
    )

    # Add text annotations
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            value = pivot_data.values[i, j]
            if not np.isnan(value):
                text_color = "white" if abs(value) > 10 else "black"
                ax.text(
                    j, i, f"{value:.1f}%", ha="center", va="center", color=text_color, fontsize=10, fontweight="bold"
                )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(
        "Avg Time Diff (%)\n← max_abc faster  |  max_ab faster →",
        rotation=270,
        labelpad=25,
        fontsize=11,
        fontweight="bold",
    )

    plt.tight_layout()
    output_path = output_dir / "heatmap_sharding_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {output_path}")
    plt.close()


def create_compute_cores_analysis(merged_df, output_dir):
    """Analyze performance by number of compute cores."""
    # Group by compute cores
    cores_analysis = (
        merged_df.groupby("compute_cores_max_ab")
        .agg(
            {
                "time_diff": ["mean", "std", "count"],
                "time_diff_pct": ["mean", "std"],
                "kernel_time_us_max_ab": "mean",
                "kernel_time_us_max_abc": "mean",
            }
        )
        .reset_index()
    )

    cores_analysis.columns = [
        "compute_cores",
        "time_diff_mean",
        "time_diff_std",
        "count",
        "time_diff_pct_mean",
        "time_diff_pct_std",
        "avg_time_max_ab",
        "avg_time_max_abc",
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Bar chart of average times by cores
    x = range(len(cores_analysis))
    width = 0.35

    bars1 = ax1.bar(
        [i - width / 2 for i in x],
        cores_analysis["avg_time_max_ab"],
        width,
        label="max_ab",
        color="#ff6b6b",
        alpha=0.7,
        edgecolor="black",
    )
    bars2 = ax1.bar(
        [i + width / 2 for i in x],
        cores_analysis["avg_time_max_abc"],
        width,
        label="max_abc",
        color="#4ecdc4",
        alpha=0.7,
        edgecolor="black",
    )

    ax1.set_xlabel("Number of Compute Cores", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Average Kernel Time (μs)", fontsize=11, fontweight="bold")
    ax1.set_title("Average Kernel Time by Compute Cores", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(cores_analysis["compute_cores"].astype(int))
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.1f}", ha="center", va="bottom", fontsize=8)

    # Line plot of percentage difference by cores
    ax2.errorbar(
        cores_analysis["compute_cores"],
        cores_analysis["time_diff_pct_mean"],
        yerr=cores_analysis["time_diff_pct_std"],
        marker="o",
        markersize=10,
        linewidth=2,
        capsize=5,
        capthick=2,
        color="#9b59b6",
        ecolor="#e74c3c",
        alpha=0.7,
    )

    ax2.axhline(y=0, color="black", linestyle="--", linewidth=2, alpha=0.5)
    ax2.set_xlabel("Number of Compute Cores", fontsize=11, fontweight="bold")
    ax2.set_ylabel(
        "Avg Performance Difference (%)\n← max_abc faster  |  max_ab faster →", fontsize=11, fontweight="bold"
    )
    ax2.set_title("Performance Difference by Compute Cores\n(with std deviation)", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # Add count annotations
    for i, row in cores_analysis.iterrows():
        ax2.text(row["compute_cores"], row["time_diff_pct_mean"], f"  n={int(row['count'])}", fontsize=8, va="center")

    plt.tight_layout()
    output_path = output_dir / "compute_cores_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {output_path}")
    plt.close()


def generate_summary_report(merged_df, output_dir):
    """Generate text summary report."""
    report = []
    report.append("=" * 80)
    report.append("KERNEL TIME COMPARISON SUMMARY: max_ab vs max_abc")
    report.append("=" * 80)
    report.append("")

    # Overall statistics
    report.append("📊 OVERALL STATISTICS")
    report.append("-" * 80)
    report.append(f"Total Configurations Compared: {len(merged_df)}")
    report.append(
        f"max_ab Faster:  {(merged_df['faster_strategy'] == 'max_ab').sum()} configs ({(merged_df['faster_strategy'] == 'max_ab').sum() / len(merged_df) * 100:.1f}%)"
    )
    report.append(
        f"max_abc Faster: {(merged_df['faster_strategy'] == 'max_abc').sum()} configs ({(merged_df['faster_strategy'] == 'max_abc').sum() / len(merged_df) * 100:.1f}%)"
    )
    report.append(
        f"Ties:           {(merged_df['faster_strategy'] == 'tie').sum()} configs ({(merged_df['faster_strategy'] == 'tie').sum() / len(merged_df) * 100:.1f}%)"
    )
    report.append("")

    report.append("📈 TIME DIFFERENCE STATISTICS")
    report.append("-" * 80)
    report.append(
        f"Mean Difference:     {merged_df['time_diff'].mean():+.3f} μs  ({merged_df['time_diff_pct'].mean():+.2f}%)"
    )
    report.append(
        f"Median Difference:   {merged_df['time_diff'].median():+.3f} μs  ({merged_df['time_diff_pct'].median():+.2f}%)"
    )
    report.append(
        f"Std Deviation:       {merged_df['time_diff'].std():.3f} μs  ({merged_df['time_diff_pct'].std():.2f}%)"
    )
    report.append(
        f"Min Difference:      {merged_df['time_diff'].min():+.3f} μs  ({merged_df['time_diff_pct'].min():+.2f}%) [max_abc much faster]"
    )
    report.append(
        f"Max Difference:      {merged_df['time_diff'].max():+.3f} μs  ({merged_df['time_diff_pct'].max():+.2f}%) [max_ab much faster]"
    )
    report.append("")

    # Top 10 where max_abc is faster
    report.append("🏆 TOP 10 CONFIGS WHERE max_abc IS FASTER")
    report.append("-" * 80)
    top_max_abc = merged_df.nsmallest(10, "time_diff_pct")
    for idx, row in top_max_abc.iterrows():
        report.append(
            f"  {row['a_sharding']:>11} + {row['b_sharding']:>11} → {row['c_sharding']:>11}  |  "
            f"{row['time_diff_pct']:+6.2f}%  ({row['time_diff']:+6.2f}μs)"
        )
    report.append("")

    # Top 10 where max_ab is faster
    report.append("🏆 TOP 10 CONFIGS WHERE max_ab IS FASTER")
    report.append("-" * 80)
    top_max_ab = merged_df.nlargest(10, "time_diff_pct")
    for idx, row in top_max_ab.iterrows():
        report.append(
            f"  {row['a_sharding']:>11} + {row['b_sharding']:>11} → {row['c_sharding']:>11}  |  "
            f"{row['time_diff_pct']:+6.2f}%  ({row['time_diff']:+6.2f}μs)"
        )
    report.append("")

    # By sharding strategy
    report.append("📋 PERFORMANCE BY OUTPUT SHARDING (C tensor)")
    report.append("-" * 80)
    for c_sharding in sorted(merged_df["c_sharding"].unique()):
        subset = merged_df[merged_df["c_sharding"] == c_sharding]
        max_ab_wins = (subset["faster_strategy"] == "max_ab").sum()
        max_abc_wins = (subset["faster_strategy"] == "max_abc").sum()
        avg_diff = subset["time_diff_pct"].mean()
        report.append(
            f"{c_sharding:>11}:  {len(subset):3d} configs  |  "
            f"avg diff: {avg_diff:+6.2f}%  |  "
            f"max_ab wins: {max_ab_wins:3d}  |  max_abc wins: {max_abc_wins:3d}"
        )
    report.append("")

    report.append("=" * 80)
    report.append("Charts generated in: " + str(output_dir))
    report.append("=" * 80)

    # Write to file
    output_path = output_dir / "visualization_summary.txt"
    with open(output_path, "w") as f:
        f.write("\n".join(report))

    # Also print to console
    print("\n".join(report))
    print(f"\n✅ Saved: {output_path}")


def main():
    # Define paths
    base_dir = Path(__file__).parent
    results_dir = base_dir / "results"
    output_dir = base_dir / "visualizations"
    output_dir.mkdir(exist_ok=True)

    csv1 = results_dir / "example_multiple_ops_max_ab_20251113_013450.csv"
    csv2 = results_dir / "example_multiple_ops_max_abc_20251113_062946.csv"

    print("=" * 80)
    print("VISUALIZATION: Kernel Time Comparison (max_ab vs max_abc)")
    print("=" * 80)
    print()

    # Load and merge data
    merged_df = load_and_merge_data(csv1, csv2)
    print()

    # Generate visualizations
    print("Generating visualizations...")
    print()

    create_scatter_plot(merged_df, output_dir)
    create_percentage_diff_chart(merged_df, output_dir)
    create_histogram(merged_df, output_dir)
    create_sharding_heatmap(merged_df, output_dir)
    create_compute_cores_analysis(merged_df, output_dir)

    # Generate summary report
    print()
    generate_summary_report(merged_df, output_dir)

    print()
    print("=" * 80)
    print("✅ VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"All charts saved to: {output_dir}/")
    print()
    print("Generated files:")
    print("  1. scatter_kernel_time_comparison.png")
    print("  2. bar_percentage_differences.png")
    print("  3. histogram_time_differences.png")
    print("  4. heatmap_sharding_comparison.png")
    print("  5. compute_cores_analysis.png")
    print("  6. visualization_summary.txt")
    print()


if __name__ == "__main__":
    main()
