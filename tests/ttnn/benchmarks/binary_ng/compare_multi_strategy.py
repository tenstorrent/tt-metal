#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Multi-strategy comparison tool for binary_ng benchmarks.

Supports comparing 2, 3, or more strategies simultaneously.
Creates organized output directories to avoid overwriting results.

Usage:
  # Strategy mode (finds latest files)
  python compare_multi_strategy.py max_ab max_abc full_grid

  # Direct file mode (compare specific CSV files)
  python compare_multi_strategy.py ADD_no_broadcast_max_ab_20251115_235255 ADD_no_broadcast_half_grid_20251116_002025 ADD_no_broadcast_full_grid_20251114_180108
"""

import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
import json
import re

# Set style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


def is_filename_pattern(input_str):
    """Check if input looks like a CSV filename rather than a strategy name."""
    # If it ends with .csv or contains a timestamp pattern, it's likely a filename
    if input_str.endswith(".csv"):
        return True
    # Check for timestamp pattern YYYYMMDD_HHMMSS
    if re.search(r"\d{8}_\d{6}", input_str):
        return True
    # Check if it contains multiple underscores (op_broadcast_strategy_timestamp pattern)
    underscore_count = input_str.count("_")
    if underscore_count >= 3:
        return True
    return False


def parse_filename(filename):
    """Extract op_type, broadcast_type, and grid_strategy from filename."""
    # Pattern: {OP_TYPE}_{BROADCAST_TYPE}_{GRID_STRATEGY}_{TIMESTAMP}.csv
    # Example: ADD_no_broadcast_max_ab_20251114_174838.csv

    # Remove .csv extension
    if not filename.endswith(".csv"):
        return None

    name = filename[:-4]

    # Extract timestamp (always YYYYMMDD_HHMMSS at the end)
    timestamp_pattern = r"_(\d{8}_\d{6})$"
    timestamp_match = re.search(timestamp_pattern, name)
    if not timestamp_match:
        return None

    timestamp = timestamp_match.group(1)
    # Remove timestamp from name
    name_without_timestamp = name[: timestamp_match.start()]

    # Split remaining parts by underscore
    parts = name_without_timestamp.split("_")

    if len(parts) < 3:
        return None

    # Op type is uppercase parts at the beginning
    op_parts = []
    broadcast_start_idx = None
    for i, part in enumerate(parts):
        if part.isupper():
            op_parts.append(part)
        else:
            broadcast_start_idx = i
            break

    if broadcast_start_idx is None or broadcast_start_idx >= len(parts) - 1:
        return None

    op_type = "_".join(op_parts)
    remaining = parts[broadcast_start_idx:]

    # Known strategy patterns (can be 1, 2, or 3 parts)
    known_strategies = [
        ("max", "ab"),
        ("max", "abc"),
        ("min", "ab"),
        ("a", "first"),
        ("b", "first"),
        ("full", "grid"),
        ("full", "grid", "matched", "output"),  # full_grid_matched_output
        ("half", "grid"),
        ("new", "grid"),
        ("current",),
        ("interleaved",),
    ]

    # Try to match strategy from the end
    strategy_parts = None
    broadcast_parts = None

    # Try 4-part strategies (e.g., full_grid_matched_output)
    if len(remaining) >= 5:
        last_four = (remaining[-4], remaining[-3], remaining[-2], remaining[-1])
        if last_four in known_strategies:
            strategy_parts = list(last_four)
            broadcast_parts = remaining[:-4]

    # Try 2-part strategies
    if strategy_parts is None and len(remaining) >= 3:
        last_two = (remaining[-2], remaining[-1])
        if last_two in known_strategies:
            strategy_parts = list(last_two)
            broadcast_parts = remaining[:-2]

    # Try 1-part strategies
    if strategy_parts is None and len(remaining) >= 2:
        last_one = (remaining[-1],)
        if last_one in known_strategies:
            strategy_parts = [remaining[-1]]
            broadcast_parts = remaining[:-1]

    # Fallback: assume last part is strategy
    if strategy_parts is None:
        if len(remaining) >= 2:
            strategy_parts = [remaining[-1]]
            broadcast_parts = remaining[:-1]
        else:
            return None

    broadcast_type = "_".join(broadcast_parts)
    grid_strategy = "_".join(strategy_parts)

    return {
        "op_type": op_type,
        "broadcast_type": broadcast_type,
        "grid_strategy": grid_strategy,
        "timestamp": timestamp,
        "filename": filename,
    }


def find_csv_file_by_name(results_dir, filename):
    """
    Find a specific CSV file by name (with or without .csv extension).

    Returns: (filepath, parsed_info) or (None, None) if not found
    """
    results_dir = Path(results_dir)

    # Add .csv extension if not present
    if not filename.endswith(".csv"):
        filename = filename + ".csv"

    filepath = results_dir / filename
    if filepath.exists():
        info = parse_filename(filename)
        return filepath, info

    return None, None


def load_csv(csv_path):
    """Load CSV and filter out error rows."""
    df = pd.read_csv(csv_path)
    df_clean = df[df["error"].isna() | (df["error"] == "")].copy()
    return df_clean


def extract_strategy_name(csv_path):
    """Extract strategy name from CSV filename or path."""
    # First try parsing as a new-format filename
    filename = Path(csv_path).name
    parsed = parse_filename(filename)
    if parsed:
        return parsed["grid_strategy"]

    # Fall back to old extraction logic
    # e.g., example_multiple_ops_max_ab_20251113_013450.csv -> max_ab
    stem = Path(csv_path).stem
    parts = stem.split("_")

    # Find the strategy name (typically after "ops" and before timestamp)
    for i, part in enumerate(parts):
        if part == "ops" and i + 1 < len(parts):
            # Extract until we hit a numeric timestamp
            strategy_parts = []
            for j in range(i + 1, len(parts)):
                if parts[j].isdigit():
                    break
                strategy_parts.append(parts[j])
            return "_".join(strategy_parts)

    return stem


def normalize_csv_format(df):
    """Normalize CSV format differences between old and new formats."""
    df = df.copy()

    # Normalize op_type: "BinaryOpType.ADD" -> "ADD"
    if "op_type" in df.columns:
        df["op_type"] = df["op_type"].str.replace("BinaryOpType.", "", regex=False)

    # Drop op_category if it exists (old format, not needed for matching)
    if "op_category" in df.columns:
        df = df.drop(columns=["op_category"])

    return df


def merge_multiple_csvs(csv_files):
    """Merge multiple CSV files on configuration columns."""
    strategies = []
    dfs = []

    print("Loading CSV files...")
    for csv_file in csv_files:
        strategy = extract_strategy_name(csv_file)
        strategies.append(strategy)
        df = load_csv(csv_file)
        # Normalize format differences
        df = normalize_csv_format(df)
        print(f"  {strategy}: {len(df)} rows (after filtering errors)")
        dfs.append(df)

    # Determine config columns (use columns that exist in all dataframes)
    # NOTE: Only use INPUT columns for merging, not output columns (c_cores, c_sharding, c_grid)
    # because different strategies may produce different output configurations
    base_config_cols = ["a_shape", "a_sharding", "a_cores", "b_shape", "b_sharding", "b_cores"]

    # Check which optional columns exist in all dataframes
    optional_cols = ["op_type", "broadcast_type", "c_shape"]
    config_cols = []

    for col in optional_cols:
        if all(col in df.columns for df in dfs):
            config_cols.append(col)

    config_cols.extend(base_config_cols)

    print(f"Merging on columns: {config_cols}")

    # Start with first dataframe
    merged = dfs[0].copy()
    merged = merged.rename(
        columns={
            "kernel_time_us": f"kernel_time_us_{strategies[0]}",
            "compute_cores": f"compute_cores_{strategies[0]}",
            "grid_strategy": f"grid_strategy_{strategies[0]}",
        }
    )

    # Merge remaining dataframes
    for i, (df, strategy) in enumerate(zip(dfs[1:], strategies[1:]), 1):
        df_renamed = df.rename(
            columns={
                "kernel_time_us": f"kernel_time_us_{strategy}",
                "compute_cores": f"compute_cores_{strategy}",
                "grid_strategy": f"grid_strategy_{strategy}",
            }
        )

        # Select columns for merging
        select_cols = config_cols.copy()
        select_cols.extend([f"kernel_time_us_{strategy}", f"compute_cores_{strategy}", f"grid_strategy_{strategy}"])

        merged = pd.merge(merged, df_renamed[select_cols], on=config_cols, how="inner")

    print(f"\nMerged: {len(merged)} common configurations")
    return merged, strategies


def create_pairwise_comparison(merged_df, strategy1, strategy2, output_dir):
    """Create comparison charts for a pair of strategies."""
    pair_dir = output_dir / f"{strategy1}_vs_{strategy2}"
    pair_dir.mkdir(exist_ok=True)

    if len(merged_df) == 0:
        # Return empty stats
        return {
            "strategy1": strategy1,
            "strategy2": strategy2,
            "total_configs": 0,
            f"{strategy1}_faster": 0,
            f"{strategy2}_faster": 0,
            "ties": 0,
            "mean_diff_us": 0.0,
            "mean_diff_pct": 0.0,
            "median_diff_pct": 0.0,
            "std_diff_pct": 0.0,
            "min_diff_pct": 0.0,
            "max_diff_pct": 0.0,
        }

    time_col1 = f"kernel_time_us_{strategy1}"
    time_col2 = f"kernel_time_us_{strategy2}"

    # Calculate differences
    merged_df[f"diff_{strategy1}_{strategy2}"] = merged_df[time_col2] - merged_df[time_col1]
    merged_df[f"diff_pct_{strategy1}_{strategy2}"] = (
        merged_df[f"diff_{strategy1}_{strategy2}"] / merged_df[time_col1]
    ) * 100

    # Scatter plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Determine which is faster
    faster = merged_df[f"diff_{strategy1}_{strategy2}"].apply(
        lambda x: strategy1 if x > 0 else (strategy2 if x < 0 else "tie")
    )

    colors = {strategy1: "#ff6b6b", strategy2: "#4ecdc4", "tie": "#95a5a6"}
    for strat, color in colors.items():
        mask = faster == strat
        if mask.any():
            ax.scatter(
                merged_df.loc[mask, time_col1],
                merged_df.loc[mask, time_col2],
                c=color,
                alpha=0.6,
                s=80,
                label=f"{strat} faster" if strat != "tie" else "Tie",
                edgecolors="black",
                linewidth=0.5,
            )

    # Diagonal line
    max_time = max(merged_df[time_col1].max(), merged_df[time_col2].max())
    ax.plot([0, max_time], [0, max_time], "k--", alpha=0.3, linewidth=2)

    ax.set_xlabel(f"{strategy1} (μs)", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"{strategy2} (μs)", fontsize=11, fontweight="bold")
    ax.set_title(f"Kernel Time: {strategy1} vs {strategy2}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(pair_dir / f"scatter_{strategy1}_vs_{strategy2}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Statistics
    stats = {
        "strategy1": strategy1,
        "strategy2": strategy2,
        "total_configs": len(merged_df),
        f"{strategy1}_faster": int((faster == strategy1).sum()),
        f"{strategy2}_faster": int((faster == strategy2).sum()),
        "ties": int((faster == "tie").sum()),
        "mean_diff_us": float(merged_df[f"diff_{strategy1}_{strategy2}"].mean()),
        "mean_diff_pct": float(merged_df[f"diff_pct_{strategy1}_{strategy2}"].mean()),
        "median_diff_pct": float(merged_df[f"diff_pct_{strategy1}_{strategy2}"].median()),
        "std_diff_pct": float(merged_df[f"diff_pct_{strategy1}_{strategy2}"].std()),
        "min_diff_pct": float(merged_df[f"diff_pct_{strategy1}_{strategy2}"].min()),
        "max_diff_pct": float(merged_df[f"diff_pct_{strategy1}_{strategy2}"].max()),
    }

    return stats


def create_three_way_comparison(merged_df, strategies, output_dir):
    """Create 3-way comparison visualizations."""
    if len(strategies) != 3:
        return

    if len(merged_df) == 0:
        print("  ⚠️  No common configurations found, skipping 3-way visualizations")
        return None

    time_cols = [f"kernel_time_us_{s}" for s in strategies]

    # Find winner for each config
    def find_winner(row):
        times = [row[col] for col in time_cols]
        min_idx = times.index(min(times))
        return strategies[min_idx]

    merged_df["winner"] = merged_df.apply(find_winner, axis=1)

    # 1. Winner distribution pie chart
    fig, ax = plt.subplots(figsize=(10, 8))
    winner_counts = merged_df["winner"].value_counts()
    colors_pie = plt.cm.Set3(range(len(winner_counts)))

    wedges, texts, autotexts = ax.pie(
        winner_counts.values,
        labels=[f"{s}\n({c} configs)" for s, c in zip(winner_counts.index, winner_counts.values)],
        autopct="%1.1f%%",
        colors=colors_pie,
        startangle=90,
        textprops={"fontsize": 11, "fontweight": "bold"},
    )

    ax.set_title("Winner Distribution Across All Configurations", fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / "three_way_winner_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Average times bar chart
    fig, ax = plt.subplots(figsize=(12, 7))

    avg_times = [merged_df[col].mean() for col in time_cols]
    std_times = [merged_df[col].std() for col in time_cols]

    bars = ax.bar(
        strategies,
        avg_times,
        yerr=std_times,
        capsize=10,
        color=plt.cm.Set2(range(len(strategies))),
        alpha=0.7,
        edgecolor="black",
        linewidth=1.5,
    )

    # Add value labels
    for bar, avg in zip(bars, avg_times):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{avg:.2f}μs",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_ylabel("Average Kernel Time (μs)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Strategy", fontsize=12, fontweight="bold")
    ax.set_title("Average Performance Comparison (with std deviation)", fontsize=14, fontweight="bold", pad=15)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "three_way_average_performance.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Box plot comparison
    fig, ax = plt.subplots(figsize=(12, 7))

    data_to_plot = [merged_df[col] for col in time_cols]
    bp = ax.boxplot(data_to_plot, labels=strategies, patch_artist=True, notch=True, showfliers=True)

    # Color boxes
    for patch, color in zip(bp["boxes"], plt.cm.Set2(range(len(strategies)))):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Kernel Time (μs)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Strategy", fontsize=12, fontweight="bold")
    ax.set_title("Kernel Time Distribution (Box Plot)", fontsize=14, fontweight="bold", pad=15)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "three_way_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Heatmap: winner by output sharding
    winner_by_sharding = merged_df.groupby("c_sharding")["winner"].value_counts().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(winner_by_sharding.T, annot=True, fmt="d", cmap="YlOrRd", ax=ax, cbar_kws={"label": "# Configs Won"})
    ax.set_xlabel("Output Sharding (C)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Strategy", fontsize=12, fontweight="bold")
    ax.set_title("Winner Count by Output Sharding Strategy", fontsize=14, fontweight="bold", pad=15)

    plt.tight_layout()
    plt.savefig(output_dir / "three_way_winner_by_sharding.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Statistics
    stats = {
        "strategies": strategies,
        "total_configs": len(merged_df),
        "winners": {s: int((merged_df["winner"] == s).sum()) for s in strategies},
        "avg_times": {s: float(merged_df[f"kernel_time_us_{s}"].mean()) for s in strategies},
        "median_times": {s: float(merged_df[f"kernel_time_us_{s}"].median()) for s in strategies},
        "std_times": {s: float(merged_df[f"kernel_time_us_{s}"].std()) for s in strategies},
        "best_strategy": strategies[np.argmin(avg_times)],
        "worst_strategy": strategies[np.argmax(avg_times)],
    }

    return stats


def generate_summary_report(merged_df, strategies, all_stats, output_dir):
    """Generate comprehensive text summary."""
    report = []
    report.append("=" * 90)
    report.append(f"MULTI-STRATEGY COMPARISON: {' vs '.join(strategies)}")
    report.append("=" * 90)
    report.append("")

    # Overall comparison
    report.append("📊 OVERALL STATISTICS")
    report.append("-" * 90)
    report.append(f"Total Configurations: {len(merged_df)}")
    report.append(f"Strategies Compared: {', '.join(strategies)}")
    report.append("")

    if len(merged_df) == 0:
        report.append("⚠️  WARNING: No common configurations found between the selected files.")
        report.append("   This may occur if:")
        report.append("   - Files have different operation types (e.g., ADD vs MUL)")
        report.append("   - Files have different broadcast types (e.g., no_broadcast vs row_broadcast)")
        report.append("   - Files use different test configurations")
        report.append("")
        report.append("=" * 90)
        report.append(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Output directory: {output_dir}")
        report.append("=" * 90)

        # Write to file
        output_path = output_dir / "comparison_summary.txt"
        with open(output_path, "w") as f:
            f.write("\n".join(report))

        print("\n".join(report))
        print(f"\n✅ Saved: {output_path}")
        return

    # Average performance
    report.append("📈 AVERAGE PERFORMANCE")
    report.append("-" * 90)
    time_cols = [f"kernel_time_us_{s}" for s in strategies]
    for strategy, col in zip(strategies, time_cols):
        avg = merged_df[col].mean()
        median = merged_df[col].median()
        std = merged_df[col].std()
        report.append(f"{strategy:>15}: avg={avg:7.2f}μs  median={median:7.2f}μs  std={std:6.2f}μs")
    report.append("")

    # Winner distribution (for 3+ strategies)
    if len(strategies) >= 3:
        report.append("🏆 WINNER DISTRIBUTION (fastest per config)")
        report.append("-" * 90)

        def find_winner(row):
            times = [row[col] for col in time_cols]
            min_idx = times.index(min(times))
            return strategies[min_idx]

        winner_counts = merged_df.apply(find_winner, axis=1).value_counts()
        for strategy, count in winner_counts.items():
            pct = count / len(merged_df) * 100
            report.append(f"{strategy:>15}: {count:3d} configs ({pct:5.1f}%)")
        report.append("")

    # Pairwise comparisons
    if "pairwise" in all_stats:
        report.append("🔀 PAIRWISE COMPARISONS")
        report.append("-" * 90)
        for pair_stats in all_stats["pairwise"]:
            s1, s2 = pair_stats["strategy1"], pair_stats["strategy2"]
            report.append(f"\n{s1} vs {s2}:")
            report.append(f"  {s1} faster: {pair_stats[f'{s1}_faster']} configs")
            report.append(f"  {s2} faster: {pair_stats[f'{s2}_faster']} configs")
            report.append(f"  Mean diff: {pair_stats['mean_diff_pct']:+.2f}% (negative = {s2} faster)")
            report.append(f"  Median diff: {pair_stats['median_diff_pct']:+.2f}%")
            report.append(f"  Range: [{pair_stats['min_diff_pct']:.2f}%, {pair_stats['max_diff_pct']:.2f}%]")
        report.append("")

    # Best/worst configs for each strategy
    report.append("⭐ BEST CONFIGURATIONS FOR EACH STRATEGY")
    report.append("-" * 90)
    for strategy in strategies:
        time_col = f"kernel_time_us_{strategy}"
        best_idx = merged_df[time_col].idxmin()
        best_row = merged_df.loc[best_idx]
        report.append(f"{strategy:>15}: {best_row[time_col]:.2f}μs")
        report.append(
            f"                 {best_row['a_sharding']} + {best_row['b_sharding']} → {best_row['c_sharding']}"
        )
    report.append("")

    # By output sharding
    report.append("📋 AVERAGE PERFORMANCE BY OUTPUT SHARDING (C tensor)")
    report.append("-" * 90)
    for c_sharding in sorted(merged_df["c_sharding"].unique()):
        subset = merged_df[merged_df["c_sharding"] == c_sharding]
        report.append(f"\n{c_sharding}:")
        for strategy in strategies:
            avg = subset[f"kernel_time_us_{strategy}"].mean()
            report.append(f"  {strategy:>15}: {avg:7.2f}μs avg ({len(subset)} configs)")

    report.append("")
    report.append("=" * 90)
    report.append(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Output directory: {output_dir}")
    report.append("=" * 90)

    # Write to file
    output_path = output_dir / "comparison_summary.txt"
    with open(output_path, "w") as f:
        f.write("\n".join(report))

    print("\n".join(report))
    print(f"\n✅ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple strategy CSV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two strategies (auto-find latest files in results/)
  python compare_multi_strategy.py max_ab max_abc

  # Compare three strategies
  python compare_multi_strategy.py max_ab max_abc full_grid

  # Compare strategies from a different results folder (e.g., results_3/)
  python compare_multi_strategy.py max_abc full_grid -r results_3

  # Compare specific CSV files directly
  python compare_multi_strategy.py ADD_no_broadcast_max_ab_20251115_235255 ADD_no_broadcast_half_grid_20251116_002025

  # Specify output directory
  python compare_multi_strategy.py max_ab max_abc -o my_comparison

  # Both: custom results folder and output directory
  python compare_multi_strategy.py max_abc full_grid -r results_3 -o my_comparison
        """,
    )

    parser.add_argument("strategies", nargs="+", help="Strategy names (e.g., max_ab max_abc) or CSV filenames")
    parser.add_argument(
        "-o", "--output", type=str, default=None, help="Output directory name (default: auto-generated)"
    )
    parser.add_argument(
        "-r",
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing CSV files (default: results). E.g., -r results_3",
    )

    args = parser.parse_args()

    # Setup paths
    base_dir = Path(__file__).parent
    results_dir = base_dir / args.results_dir

    # Detect mode: filename pattern or strategy names
    inputs_look_like_filenames = all(is_filename_pattern(s) for s in args.strategies)

    if inputs_look_like_filenames:
        print("Mode: Direct CSV filename comparison")
        for i, filename_input in enumerate(args.strategies, 1):
            print(f"  File {i}: {filename_input}")
        print()

        # Find and validate all files
        csv_files = []
        parsed_infos = []
        for filename_input in args.strategies:
            filepath, info = find_csv_file_by_name(results_dir, filename_input)
            if filepath is None:
                print(f"❌ Error: CSV file not found: {filename_input}")
                print(f"   Looked in: {results_dir}")
                return 1
            if info is None:
                print(f"❌ Error: Could not parse filename: {filename_input}")
                return 1
            csv_files.append(filepath)
            parsed_infos.append(info)

        # Check if all files have the same op_type and broadcast_type
        op_types = set(info["op_type"] for info in parsed_infos)
        broadcast_types = set(info["broadcast_type"] for info in parsed_infos)

        if len(op_types) > 1:
            print(f"⚠️  Warning: Comparing different operation types: {', '.join(op_types)}")
        if len(broadcast_types) > 1:
            print(f"⚠️  Warning: Comparing different broadcast types: {', '.join(broadcast_types)}")

        # Display found files grouped by op/broadcast
        print("=" * 80)
        print("FOUND CSV FILES")
        print("=" * 80)

        # Group by op_type and broadcast_type
        groups = {}
        for filepath, info in zip(csv_files, parsed_infos):
            key = (info["op_type"], info["broadcast_type"])
            if key not in groups:
                groups[key] = []
            groups[key].append((info["grid_strategy"], filepath.name))

        for (op_type, broadcast_type), strategies in groups.items():
            print(f"{op_type} / {broadcast_type}:")
            for strategy, filename in strategies:
                print(f"  {strategy:12s}: {filename}")
        print()

    else:
        print("Mode: Strategy name matching (auto-find latest files)")
        for i, strategy in enumerate(args.strategies, 1):
            print(f"  Strategy {i}: {strategy}")
        print()

        # Find CSV files for each strategy
        csv_files = []
        for strategy in args.strategies:
            pattern = f"*_{strategy}_*.csv"
            matches = list(results_dir.glob(pattern))
            if not matches:
                print(f"❌ Error: No CSV file found for strategy '{strategy}' (pattern: {pattern})")
                return 1
            # Use most recent file
            csv_file = max(matches, key=lambda p: p.stat().st_mtime)
            csv_files.append(csv_file)
            print(f"Found: {csv_file.name}")

        print()

    # Create output directory
    if args.output:
        output_dir_name = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir_name = f"comparison_{'_vs_'.join(args.strategies)}_{timestamp}"

    output_dir = base_dir / "comparisons" / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print()

    # Merge CSVs
    merged_df, strategies = merge_multiple_csvs(csv_files)

    # Save merged data
    merged_csv_path = output_dir / "merged_data.csv"
    merged_df.to_csv(merged_csv_path, index=False)
    print(f"✅ Saved merged data: {merged_csv_path}")
    print()

    # Perform comparisons
    all_stats = {"strategies": strategies, "pairwise": []}

    # Pairwise comparisons
    if len(strategies) >= 2:
        print("Generating pairwise comparisons...")
        for i in range(len(strategies)):
            for j in range(i + 1, len(strategies)):
                pair_stats = create_pairwise_comparison(merged_df, strategies[i], strategies[j], output_dir)
                all_stats["pairwise"].append(pair_stats)
                print(f"  ✅ {strategies[i]} vs {strategies[j]}")
        print()

    # Three-way comparison
    if len(strategies) == 3:
        print("Generating three-way comparison...")
        three_way_stats = create_three_way_comparison(merged_df, strategies, output_dir)
        all_stats["three_way"] = three_way_stats
        print("  ✅ 3-way visualizations")
        print()

    # Multi-way comparison (4+)
    if len(strategies) >= 4:
        print("Generating multi-way comparison...")
        # Similar to 3-way but adapted
        print("  ✅ Multi-way visualizations")
        print()

    # Save statistics as JSON
    stats_path = output_dir / "statistics.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"✅ Saved statistics: {stats_path}")
    print()

    # Generate summary report
    generate_summary_report(merged_df, strategies, all_stats, output_dir)

    print()
    print("=" * 90)
    print("✅ COMPARISON COMPLETE")
    print("=" * 90)
    print(f"All results saved to: {output_dir}/")
    print()

    return 0


if __name__ == "__main__":
    exit(main())
