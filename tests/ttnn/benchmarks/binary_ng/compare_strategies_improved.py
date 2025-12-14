#!/usr/bin/env python3
"""
Compare binary_ng strategy CSV results across operation types and broadcast types.

Usage:
    # Compare strategies (finds latest files for each strategy)
    python compare_strategies_improved.py max_ab min_ab
    python compare_strategies_improved.py max_ab full_grid --op-type ADD

    # Compare specific CSV files by name (with or without .csv extension)
    python compare_strategies_improved.py ADD_no_broadcast_max_ab_20251115_235255 ADD_no_broadcast_half_grid_20251116_002025
    python compare_strategies_improved.py ADD_no_broadcast_max_ab_20251115_235255.csv ADD_no_broadcast_max_ab_20251115_225919.csv
"""

import pandas as pd
import sys
import argparse
import re
import json
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Set style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


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

    # Known strategy patterns (can be 1 or 2 parts)
    known_strategies = [
        ("max", "ab"),
        ("max", "abc"),
        ("min", "ab"),
        ("a", "first"),
        ("b", "first"),
        ("full", "grid"),
        ("half", "grid"),
        ("current",),
        ("interleaved",),
    ]

    # Try to match strategy from the end
    strategy_parts = None
    broadcast_parts = None

    # Try 2-part strategies
    if len(remaining) >= 3:
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


def find_csv_files(results_dir, strategies=None, op_types=None, broadcast_types=None):
    """
    Find all CSV files matching the criteria.

    Returns dict: {(op_type, broadcast_type, strategy): filepath}
    """
    results_dir = Path(results_dir)
    csv_files = {}

    for csv_file in results_dir.glob("*.csv"):
        info = parse_filename(csv_file.name)
        if not info:
            continue

        # Filter by criteria
        if strategies and info["grid_strategy"] not in strategies:
            continue
        if op_types and info["op_type"] not in op_types:
            continue
        if broadcast_types and info["broadcast_type"] not in broadcast_types:
            continue

        key = (info["op_type"], info["broadcast_type"], info["grid_strategy"])
        # Keep the most recent file for each key
        if key not in csv_files or csv_file.stat().st_mtime > csv_files[key].stat().st_mtime:
            csv_files[key] = csv_file

    return csv_files


def load_csv(path):
    """Load CSV and filter out empty/error rows."""
    df = pd.read_csv(path)
    # Remove empty rows
    df = df.dropna(subset=["op_type"])
    return df


def create_config_key(row):
    """Create a unique key for matching configurations."""
    return (
        row["a_shape"],
        row["a_sharding"],
        row["a_cores"],
        row["b_shape"],
        row["b_sharding"],
        row["b_cores"],
        row["c_sharding"],
    )


def compare_two_strategies(df1, df2, strategy1_name, strategy2_name):
    """Compare two strategy DataFrames and return analysis."""
    # Create dictionaries keyed by configuration
    dict1 = {}
    dict2 = {}

    for _, row in df1.iterrows():
        if pd.notna(row["kernel_time_us"]) and row["kernel_time_us"] > 0:
            key = create_config_key(row)
            dict1[key] = row

    for _, row in df2.iterrows():
        if pd.notna(row["kernel_time_us"]) and row["kernel_time_us"] > 0:
            key = create_config_key(row)
            dict2[key] = row

    # Find matching configurations
    common_keys = set(dict1.keys()) & set(dict2.keys())

    if len(common_keys) == 0:
        return None

    # Analyze differences for common configurations
    differences = []

    for key in common_keys:
        row1 = dict1[key]
        row2 = dict2[key]

        time1 = row1["kernel_time_us"]
        time2 = row2["kernel_time_us"]

        diff_us = time2 - time1
        pct_diff = (diff_us / time1) * 100 if time1 > 0 else 0

        differences.append(
            {
                "config": key,
                f"{strategy1_name}_time": time1,
                f"{strategy2_name}_time": time2,
                "diff_us": diff_us,
                "pct_diff": pct_diff,
                f"{strategy1_name}_cores": row1["compute_cores"],
                f"{strategy2_name}_cores": row2["compute_cores"],
                "c_cores": row1["c_cores"],
                "c_sharding": row1["c_sharding"],
                "a_cores": row1["a_cores"],
                "b_cores": row1["b_cores"],
                "a_sharding": row1["a_sharding"],
                "b_sharding": row1["b_sharding"],
            }
        )

    return pd.DataFrame(differences)


def create_scatter_plot(diff_df, strategy1_name, strategy2_name, op_type, broadcast_type, output_dir):
    """Create scatter plot comparing two strategies."""
    if diff_df is None or len(diff_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 10))

    time1_col = f"{strategy1_name}_time"
    time2_col = f"{strategy2_name}_time"

    # Determine which is faster
    # diff_us = time2 - time1, so if diff_us > 0, strategy1 is faster (lower time)
    faster = diff_df["diff_us"].apply(lambda x: strategy1_name if x > 0 else (strategy2_name if x < 0 else "tie"))

    colors = {strategy1_name: "#ff6b6b", strategy2_name: "#4ecdc4", "tie": "#95a5a6"}
    for strat, color in colors.items():
        mask = faster == strat
        if mask.any():
            ax.scatter(
                diff_df.loc[mask, time1_col],
                diff_df.loc[mask, time2_col],
                c=color,
                alpha=0.6,
                s=80,
                label=f"{strat} faster" if strat != "tie" else "Tie",
                edgecolors="black",
                linewidth=0.5,
            )

    # Diagonal line (equal performance)
    max_time = max(diff_df[time1_col].max(), diff_df[time2_col].max())
    ax.plot([0, max_time], [0, max_time], "k--", alpha=0.3, linewidth=2, label="Equal performance")

    ax.set_xlabel(f"{strategy1_name} (μs)", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"{strategy2_name} (μs)", fontsize=11, fontweight="bold")
    ax.set_title(f"{op_type} / {broadcast_type}: {strategy1_name} vs {strategy2_name}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Add annotations for winner count
    faster1 = (faster == strategy1_name).sum()
    faster2 = (faster == strategy2_name).sum()
    ties = (faster == "tie").sum()

    annotation_text = f"{strategy1_name}: {faster1} wins\n{strategy2_name}: {faster2} wins"
    if ties > 0:
        annotation_text += f"\nTies: {ties}"

    ax.text(
        0.98,
        0.02,
        annotation_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"scatter_{strategy1_name}_vs_{strategy2_name}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    return plot_path


def save_comparison_data(diff_df, op_type, broadcast_type, output_dir):
    """Save merged comparison data to CSV."""
    if diff_df is None or len(diff_df) == 0:
        return None

    # Add op_type and broadcast_type to the data
    save_df = diff_df.copy()
    save_df["op_type"] = op_type
    save_df["broadcast_type"] = broadcast_type

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"comparison_data_{op_type}_{broadcast_type}.csv"
    save_df.to_csv(csv_path, index=False)

    return csv_path


def generate_summary_text(all_comparisons, strategy1_name, strategy2_name, output_dir):
    """Generate a text summary file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "comparison_summary.txt"

    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"STRATEGY COMPARISON: {strategy1_name} vs {strategy2_name}\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write("=" * 80 + "\n\n")

        # Overall statistics
        total_configs = sum(comp["total_configs"] for comp in all_comparisons.values())
        total_faster1 = sum(comp["faster1"] for comp in all_comparisons.values())
        total_faster2 = sum(comp["faster2"] for comp in all_comparisons.values())

        f.write("OVERALL SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total configurations: {total_configs}\n")
        f.write(f"{strategy1_name} faster: {total_faster1} cases ({total_faster1/total_configs*100:.1f}%)\n")
        f.write(f"{strategy2_name} faster: {total_faster2} cases ({total_faster2/total_configs*100:.1f}%)\n\n")

        # Per op_type and broadcast_type
        f.write("BREAKDOWN BY OPERATION AND BROADCAST TYPE\n")
        f.write("-" * 80 + "\n\n")

        for (op_type, broadcast_type), comp in sorted(all_comparisons.items()):
            f.write(f"{op_type} / {broadcast_type}:\n")
            f.write(f"  Configurations: {comp['total_configs']}\n")
            f.write(
                f"  {strategy1_name} faster: {comp['faster1']} ({comp['faster1']/comp['total_configs']*100:.1f}%)\n"
            )
            f.write(
                f"  {strategy2_name} faster: {comp['faster2']} ({comp['faster2']/comp['total_configs']*100:.1f}%)\n"
            )
            f.write(f"  Mean time difference: {comp['mean_diff_us']:.3f} μs ({comp['mean_diff_pct']:+.2f}%)\n")
            f.write(f"  Median time difference: {comp['median_diff_us']:.3f} μs ({comp['median_diff_pct']:+.2f}%)\n")
            f.write(f"\n")

        f.write("=" * 80 + "\n")
        f.write("Output files generated:\n")
        f.write(f"  - comparison_summary.txt (this file)\n")
        f.write(f"  - statistics.json\n")
        f.write(f"  - For each comparison:\n")
        f.write(f"      - comparison_data_{{OP_TYPE}}_{{BROADCAST_TYPE}}.csv\n")
        f.write(f"      - {{strategy1}}_vs_{{strategy2}}/scatter_*.png\n")
        f.write("=" * 80 + "\n")

    return summary_path


def save_statistics_json(all_comparisons, strategy1_name, strategy2_name, output_dir):
    """Save statistics to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    stats = {"strategies": [strategy1_name, strategy2_name], "generated": datetime.now().isoformat(), "comparisons": {}}

    for (op_type, broadcast_type), comp in all_comparisons.items():
        key = f"{op_type}_{broadcast_type}"
        stats["comparisons"][key] = comp

    # Overall summary
    total_configs = sum(comp["total_configs"] for comp in all_comparisons.values())
    total_faster1 = sum(comp["faster1"] for comp in all_comparisons.values())
    total_faster2 = sum(comp["faster2"] for comp in all_comparisons.values())

    stats["overall"] = {
        "total_configs": total_configs,
        f"{strategy1_name}_faster": total_faster1,
        f"{strategy2_name}_faster": total_faster2,
        f"{strategy1_name}_win_rate": total_faster1 / total_configs * 100 if total_configs > 0 else 0,
        f"{strategy2_name}_win_rate": total_faster2 / total_configs * 100 if total_configs > 0 else 0,
    }

    json_path = output_dir / "statistics.json"
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)

    return json_path


def print_comparison_summary(diff_df, strategy1_name, strategy2_name, op_type, broadcast_type):
    """Print detailed comparison summary."""
    if diff_df is None or len(diff_df) == 0:
        print(f"  No common configurations to compare")
        return

    print(f"\n{'='*80}")
    print(f"COMPARISON: {strategy1_name} vs {strategy2_name}")
    print(f"Operation: {op_type}, Broadcast: {broadcast_type}")
    print(f"{'='*80}")
    print(f"Total configurations compared: {len(diff_df)}")
    print()

    # Overall statistics
    print(f"Mean time difference: {diff_df['diff_us'].mean():.3f} μs ({diff_df['pct_diff'].mean():.2f}%)")
    print(f"Median time difference: {diff_df['diff_us'].median():.3f} μs ({diff_df['pct_diff'].median():.2f}%)")
    print(f"Std dev: {diff_df['diff_us'].std():.3f} μs")
    print()

    # diff_us = time2 - time1, so:
    # If diff_us > 0: time2 is SLOWER → strategy1 is FASTER
    # If diff_us < 0: time2 is FASTER → strategy2 is FASTER
    faster1 = (diff_df["diff_us"] > 0).sum()  # strategy1 faster (time1 < time2)
    faster2 = (diff_df["diff_us"] < 0).sum()  # strategy2 faster (time2 < time1)
    identical = (diff_df["diff_us"] == 0).sum()

    print(f"{strategy1_name} faster: {faster1} cases ({faster1 / len(diff_df) * 100:.1f}%)")
    print(f"{strategy2_name} faster: {faster2} cases ({faster2 / len(diff_df) * 100:.1f}%)")
    print(f"Identical: {identical} cases")
    print()

    # Compute cores analysis
    diff_df["cores_differ"] = diff_df[f"{strategy1_name}_cores"] != diff_df[f"{strategy2_name}_cores"]
    cores_differ_count = diff_df["cores_differ"].sum()

    if cores_differ_count > 0:
        print(f"Compute cores differ in {cores_differ_count} cases ({cores_differ_count / len(diff_df) * 100:.1f}%)")
        cores_diff_subset = diff_df[diff_df["cores_differ"]]
        print(f"  When cores differ:")
        print(f"    Mean performance delta: {cores_diff_subset['pct_diff'].mean():+.2f}%")
        print(f"    {strategy1_name} faster: {(cores_diff_subset['diff_us'] > 0).sum()} cases")
        print(f"    {strategy2_name} faster: {(cores_diff_subset['diff_us'] < 0).sum()} cases")
        print()

    # Output sharding analysis
    print("Performance by output sharding:")
    for c_sharding in ["height", "width", "block", "interleaved"]:
        subset = diff_df[diff_df["c_sharding"] == c_sharding]
        if len(subset) > 0:
            faster1_count = (subset["diff_us"] > 0).sum()
            faster2_count = (subset["diff_us"] < 0).sum()
            print(
                f"  {c_sharding:12s}: {len(subset):3d} cases, "
                f"avg {subset['pct_diff'].mean():+6.2f}%, "
                f"{strategy1_name}:{faster1_count:3d} {strategy2_name}:{faster2_count:3d}"
            )
    print()

    # Top wins for each strategy
    # diff_us = time2 - time1
    # Largest positive diff_us = strategy1 is much faster (time1 << time2)
    # Smallest negative diff_us = strategy2 is much faster (time2 << time1)
    print(f"Top 5 cases where {strategy1_name} is FASTER:")
    top_strategy1 = diff_df.nlargest(5, "diff_us")  # Most positive diff_us
    for idx, row in top_strategy1.iterrows():
        config = row["config"]
        print(
            f"  {config[0]} ({config[1]},{row['a_cores']}) + {config[3]} ({config[4]},{row['b_cores']}) "
            f"→ c={config[6]},{row['c_cores']}"
        )
        print(
            f"    {strategy1_name}: {row[f'{strategy1_name}_cores']:2d} cores, {row[f'{strategy1_name}_time']:6.2f} μs"
        )
        print(
            f"    {strategy2_name}: {row[f'{strategy2_name}_cores']:2d} cores, {row[f'{strategy2_name}_time']:6.2f} μs"
        )
        print(f"    → {strategy1_name} is {row['diff_us']:.2f} μs faster ({row['pct_diff']:+.1f}%)")
    print()

    print(f"Top 5 cases where {strategy2_name} is FASTER:")
    top_strategy2 = diff_df.nsmallest(5, "diff_us")  # Most negative diff_us
    for idx, row in top_strategy2.iterrows():
        config = row["config"]
        print(
            f"  {config[0]} ({config[1]},{row['a_cores']}) + {config[3]} ({config[4]},{row['b_cores']}) "
            f"→ c={config[6]},{row['c_cores']}"
        )
        print(
            f"    {strategy1_name}: {row[f'{strategy1_name}_cores']:2d} cores, {row[f'{strategy1_name}_time']:6.2f} μs"
        )
        print(
            f"    {strategy2_name}: {row[f'{strategy2_name}_cores']:2d} cores, {row[f'{strategy2_name}_time']:6.2f} μs"
        )
        print(f"    → {strategy2_name} is {-row['diff_us']:.2f} μs faster ({-row['pct_diff']:+.1f}%)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare binary_ng strategy results across op types and broadcast types",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare strategies (finds latest files)
  %(prog)s max_ab min_ab
  %(prog)s max_ab full_grid --op-type ADD

  # Compare specific CSV files by name
  %(prog)s ADD_no_broadcast_max_ab_20251115_235255 ADD_no_broadcast_half_grid_20251116_002025
  %(prog)s ADD_no_broadcast_max_ab_20251115_235255.csv ADD_no_broadcast_max_ab_20251115_225919.csv
        """,
    )
    parser.add_argument("input1", help="First input: strategy name (e.g., max_ab) or CSV filename")
    parser.add_argument("input2", help="Second input: strategy name (e.g., min_ab) or CSV filename")
    parser.add_argument("--op-type", nargs="+", help="Filter by operation type(s) - only for strategy mode")
    parser.add_argument("--broadcast-type", nargs="+", help="Filter by broadcast type(s) - only for strategy mode")
    parser.add_argument(
        "--results-dir",
        default="/workspace/tests/ttnn/benchmarks/binary_ng/results",
        help="Directory containing CSV results (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for comparison results (default: auto-generated in comparisons/)",
    )
    parser.add_argument(
        "--no-viz", action="store_true", help="Skip generating visualizations (faster, text output only)"
    )

    args = parser.parse_args()

    # Determine if inputs are filenames or strategy names
    input1_is_file = is_filename_pattern(args.input1)
    input2_is_file = is_filename_pattern(args.input2)

    # Both must be the same type
    if input1_is_file != input2_is_file:
        print("ERROR: Both inputs must be either strategy names OR CSV filenames")
        print(f"  Input 1: {args.input1} (detected as: {'filename' if input1_is_file else 'strategy'})")
        print(f"  Input 2: {args.input2} (detected as: {'filename' if input2_is_file else 'strategy'})")
        return 1

    # Mode: Direct filename comparison
    if input1_is_file:
        print(f"Mode: Direct CSV filename comparison")
        print(f"  File 1: {args.input1}")
        print(f"  File 2: {args.input2}")
        print()

        # Find specific files
        filepath1, info1 = find_csv_file_by_name(args.results_dir, args.input1)
        filepath2, info2 = find_csv_file_by_name(args.results_dir, args.input2)

        if not filepath1:
            print(f"ERROR: File not found: {args.input1}")
            print(f"  Looking in: {args.results_dir}")
            return 1

        if not filepath2:
            print(f"ERROR: File not found: {args.input2}")
            print(f"  Looking in: {args.results_dir}")
            return 1

        # Create single comparison
        comparisons = defaultdict(dict)
        op_type = info1["op_type"]
        broadcast_type = info1["broadcast_type"]
        strategy1 = info1["grid_strategy"]
        strategy2 = info2["grid_strategy"]

        comparisons[(op_type, broadcast_type)][strategy1] = filepath1
        comparisons[(op_type, broadcast_type)][strategy2] = filepath2

        # For output naming
        args.strategy1 = strategy1
        args.strategy2 = strategy2

    # Mode: Strategy comparison
    else:
        print(f"Mode: Strategy comparison (finds latest files)")
        print(f"  Strategy 1: {args.input1}")
        print(f"  Strategy 2: {args.input2}")
        print()

        args.strategy1 = args.input1
        args.strategy2 = args.input2

        # Find all relevant CSV files
        csv_files = find_csv_files(
            args.results_dir,
            strategies=[args.strategy1, args.strategy2],
            op_types=args.op_type,
            broadcast_types=args.broadcast_type,
        )

        if not csv_files:
            print(f"ERROR: No CSV files found matching criteria")
            print(f"  Strategies: {args.strategy1}, {args.strategy2}")
            print(f"  Op types: {args.op_type or 'all'}")
            print(f"  Broadcast types: {args.broadcast_type or 'all'}")
            return 1

        # Group files by (op_type, broadcast_type)
        comparisons = defaultdict(dict)
        for (op_type, broadcast_type, strategy), filepath in csv_files.items():
            comparisons[(op_type, broadcast_type)][strategy] = filepath

    # Print found files
    print(f"{'='*80}")
    print(f"FOUND CSV FILES")
    print(f"{'='*80}")
    for (op_type, broadcast_type), strategies in sorted(comparisons.items()):
        print(f"\n{op_type} / {broadcast_type}:")
        for strategy, filepath in strategies.items():
            print(f"  {strategy:12s}: {filepath.name}")
    print()

    # Create output directory
    if args.output_dir:
        output_base_dir = Path(args.output_dir)
    else:
        # Auto-generate output directory name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base_dir = (
            Path(args.results_dir).parent
            / "comparisons"
            / f"comparison_{args.strategy1}_vs_{args.strategy2}_{timestamp}"
        )

    output_base_dir.mkdir(parents=True, exist_ok=True)

    # Perform comparisons
    all_results = []
    all_comparison_stats = {}

    for (op_type, broadcast_type), strategies in sorted(comparisons.items()):
        if args.strategy1 not in strategies or args.strategy2 not in strategies:
            print(f"\nSkipping {op_type}/{broadcast_type}: missing one or both strategies")
            continue

        # Load CSVs
        df1 = load_csv(strategies[args.strategy1])
        df2 = load_csv(strategies[args.strategy2])

        # Compare
        diff_df = compare_two_strategies(df1, df2, args.strategy1, args.strategy2)

        if diff_df is not None:
            diff_df["op_type"] = op_type
            diff_df["broadcast_type"] = broadcast_type
            all_results.append(diff_df)

            # Print detailed comparison
            print_comparison_summary(diff_df, args.strategy1, args.strategy2, op_type, broadcast_type)

            # Generate visualizations and save data
            if not args.no_viz:
                # Create subdirectory for this comparison
                pair_dir = output_base_dir / f"{args.strategy1}_vs_{args.strategy2}"

                # Save scatter plot
                plot_path = create_scatter_plot(
                    diff_df, args.strategy1, args.strategy2, op_type, broadcast_type, pair_dir
                )
                if plot_path:
                    print(f"  📊 Scatter plot saved: {plot_path.relative_to(output_base_dir)}")

                # Save comparison data
                csv_path = save_comparison_data(diff_df, op_type, broadcast_type, output_base_dir)
                if csv_path:
                    print(f"  💾 Data saved: {csv_path.relative_to(output_base_dir)}")

            # Collect statistics
            faster1 = (diff_df["diff_us"] > 0).sum()  # strategy1 faster when diff_us > 0
            faster2 = (diff_df["diff_us"] < 0).sum()  # strategy2 faster when diff_us < 0
            all_comparison_stats[(op_type, broadcast_type)] = {
                "total_configs": len(diff_df),
                "faster1": int(faster1),
                "faster2": int(faster2),
                "mean_diff_us": float(diff_df["diff_us"].mean()),
                "mean_diff_pct": float(diff_df["pct_diff"].mean()),
                "median_diff_us": float(diff_df["diff_us"].median()),
                "median_diff_pct": float(diff_df["pct_diff"].median()),
            }

    # Overall summary across all op_types and broadcast_types
    if len(all_results) > 1:
        combined_df = pd.concat(all_results, ignore_index=True)

        print(f"\n{'='*80}")
        print(f"OVERALL SUMMARY ACROSS ALL OP TYPES AND BROADCAST TYPES")
        print(f"{'='*80}")
        print(f"Total configurations compared: {len(combined_df)}")
        print()

        # Group by op_type
        print("By Operation Type:")
        for op_type in sorted(combined_df["op_type"].unique()):
            subset = combined_df[combined_df["op_type"] == op_type]
            faster1 = (subset["diff_us"] > 0).sum()
            faster2 = (subset["diff_us"] < 0).sum()
            print(
                f"  {op_type:12s}: {len(subset):3d} configs, "
                f"avg {subset['pct_diff'].mean():+6.2f}%, "
                f"{args.strategy1}:{faster1:3d} {args.strategy2}:{faster2:3d}"
            )
        print()

        # Group by broadcast_type
        print("By Broadcast Type:")
        for broadcast_type in sorted(combined_df["broadcast_type"].unique()):
            subset = combined_df[combined_df["broadcast_type"] == broadcast_type]
            faster1 = (subset["diff_us"] > 0).sum()
            faster2 = (subset["diff_us"] < 0).sum()
            print(
                f"  {broadcast_type:15s}: {len(subset):3d} configs, "
                f"avg {subset['pct_diff'].mean():+6.2f}%, "
                f"{args.strategy1}:{faster1:3d} {args.strategy2}:{faster2:3d}"
            )
        print()

        # Overall winner
        overall_faster1 = (combined_df["diff_us"] > 0).sum()
        overall_faster2 = (combined_df["diff_us"] < 0).sum()
        print(f"Overall:")
        print(f"  {args.strategy1} faster: {overall_faster1} cases ({overall_faster1 / len(combined_df) * 100:.1f}%)")
        print(f"  {args.strategy2} faster: {overall_faster2} cases ({overall_faster2 / len(combined_df) * 100:.1f}%)")
        print(f"  Mean performance delta: {combined_df['pct_diff'].mean():+.2f}%")
        print(f"  Median performance delta: {combined_df['pct_diff'].median():+.2f}%")
        print()

    # Generate summary files
    if all_comparison_stats and not args.no_viz:
        print(f"\n{'='*80}")
        print(f"GENERATING SUMMARY FILES")
        print(f"{'='*80}")

        # Generate text summary
        summary_path = generate_summary_text(all_comparison_stats, args.strategy1, args.strategy2, output_base_dir)
        print(f"📄 Summary: {summary_path}")

        # Generate JSON statistics
        json_path = save_statistics_json(all_comparison_stats, args.strategy1, args.strategy2, output_base_dir)
        print(f"📊 Statistics: {json_path}")

        print(f"\n{'='*80}")
        print(f"✅ All outputs saved to: {output_base_dir}")
        print(f"{'='*80}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
