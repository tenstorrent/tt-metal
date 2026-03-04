#!/usr/bin/env python3
"""
Compare profiler results across multiple prompt lengths within a single run.

This script aggregates and compares profiler results across different prompt lengths
(128, 4k, 8k, 16k, 32k, 64k, 128k tokens) from a single profiler sweep run.
It shows how operation timings scale with input size.

Use this script to understand:
  - How total prefill/decode time scales with prompt length
  - Which operations grow most with input size
  - Percentage breakdown of time per operation at each prompt length

For comparing two different runs (baseline vs optimized), use compare_two_runs.py instead.

=============================================================================
USAGE
=============================================================================

    python scripts/compare_profiler_results.py <sweep_results_dir> [options]

=============================================================================
ARGUMENTS
=============================================================================

    sweep_results_dir   Directory containing prompt length subdirectories
                        (e.g., profiler_sweep_results/baseline/)
                        Must contain subdirectories: 128/, 4k/, 8k/, etc.
                        Each subdirectory must have prefill.csv and decode.csv

=============================================================================
OPTIONS
=============================================================================

    --output-dir DIR    Directory to write comparison files
                        Default: same as sweep_results_dir

=============================================================================
OUTPUT FILES
=============================================================================

    prefill_comparison.csv  - Aggregated per-op timing across all prompt lengths
    decode_comparison.csv   - Aggregated per-op timing across all prompt lengths

    CSV columns:
      - OP_NAME: Operation name
      - <length>_time_us: Total time for this op at each prompt length
      - <length>_count: Number of invocations at each prompt length
      - <length>_min_cores, <length>_max_cores: Core usage range

    Terminal output includes:
      - Total time by prompt length (us and ms)
      - Scaling factor relative to smallest prompt length
      - Top 15 operations by time
      - Percentage breakdown per prompt length

=============================================================================
EXAMPLES
=============================================================================

    # Compare across prompt lengths for a run named 'baseline'
    python scripts/compare_profiler_results.py profiler_sweep_results/baseline/

    # Save comparison to a different directory
    python scripts/compare_profiler_results.py profiler_sweep_results/baseline/ --output-dir analysis/

    # This is typically called automatically by run_profiler_sweep.sh
    ./scripts/run_profiler_sweep.sh --run-name baseline

=============================================================================
RELATED SCRIPTS
=============================================================================

    run_profiler_sweep.sh       - Runs profiler and calls this script automatically
    compare_two_runs.py         - Compare TWO DIFFERENT runs (baseline vs optimized)
    compare_ops_raw.py          - Raw op-by-op comparison (not aggregated)
    parse_profiler_report.py    - Parse raw profiler output
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


# Prompt length order for display
PROMPT_ORDER = ["128", "4k", "8k", "16k", "32k", "64k", "128k"]


def read_phase_csv(csv_path: Path) -> list:
    """Read a prefill.csv or decode.csv file and return list of rows."""
    if not csv_path.exists():
        return []

    data = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def compute_op_stats(data: list) -> dict:
    """Compute per-operation statistics from parsed data.

    Returns:
        Dictionary with op_name -> {total_time_us, count, min_cores, max_cores}
    """
    op_stats = defaultdict(lambda: {"total_time_us": 0.0, "count": 0, "min_cores": float("inf"), "max_cores": 0})

    for row in data:
        op_name = row.get("OP_NAME", "Unknown")
        try:
            kernel_dur = float(row.get("KERNEL_DUR_us", 0))
        except (ValueError, TypeError):
            kernel_dur = 0.0
        try:
            cores = int(row.get("CORES", 0))
        except (ValueError, TypeError):
            cores = 0

        op_stats[op_name]["total_time_us"] += kernel_dur
        op_stats[op_name]["count"] += 1
        if cores > 0:
            op_stats[op_name]["min_cores"] = min(op_stats[op_name]["min_cores"], cores)
            op_stats[op_name]["max_cores"] = max(op_stats[op_name]["max_cores"], cores)

    # Fix min_cores for ops with no core data
    for op_name in op_stats:
        if op_stats[op_name]["min_cores"] == float("inf"):
            op_stats[op_name]["min_cores"] = 0

    return dict(op_stats)


def collect_all_results(sweep_dir: Path) -> dict:
    """Collect results from all prompt length directories.

    Returns:
        {
            'prefill': {prompt_len: {op_name: stats, ...}, ...},
            'decode': {prompt_len: {op_name: stats, ...}, ...}
        }
    """
    results = {"prefill": {}, "decode": {}}

    for prompt_len in PROMPT_ORDER:
        prompt_dir = sweep_dir / prompt_len
        if not prompt_dir.exists():
            continue

        # Read prefill data
        prefill_csv = prompt_dir / "prefill.csv"
        prefill_data = read_phase_csv(prefill_csv)
        if prefill_data:
            results["prefill"][prompt_len] = compute_op_stats(prefill_data)

        # Read decode data
        decode_csv = prompt_dir / "decode.csv"
        decode_data = read_phase_csv(decode_csv)
        if decode_data:
            results["decode"][prompt_len] = compute_op_stats(decode_data)

    return results


def get_all_ops(phase_results: dict) -> list:
    """Get sorted list of all unique operation names across all prompt lengths."""
    all_ops = set()
    for prompt_len, op_stats in phase_results.items():
        all_ops.update(op_stats.keys())
    return sorted(all_ops)


def write_comparison_csv(phase_results: dict, output_path: Path, phase_name: str):
    """Write comparison CSV with ops as rows and prompt lengths as columns."""
    if not phase_results:
        print(f"No data for {phase_name} comparison")
        return

    all_ops = get_all_ops(phase_results)
    prompt_lengths = [p for p in PROMPT_ORDER if p in phase_results]

    if not prompt_lengths:
        print(f"No prompt lengths found for {phase_name}")
        return

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header row: OP_NAME, '', 128_time_us, 128_count, ..., '', 4k_time_us, 4k_count, ...
        header = ["OP_NAME"]
        for pl in prompt_lengths:
            header.append("")  # Empty separator column
            header.extend([f"{pl}_time_us", f"{pl}_count", f"{pl}_min_cores", f"{pl}_max_cores"])
        writer.writerow(header)

        # Data rows
        for op_name in all_ops:
            row = [op_name]
            for pl in prompt_lengths:
                row.append("")  # Empty separator column
                stats = phase_results.get(pl, {}).get(op_name, {})
                row.append(f"{stats.get('total_time_us', 0):.2f}")
                row.append(stats.get("count", 0))
                row.append(stats.get("min_cores", 0))
                row.append(stats.get("max_cores", 0))
            writer.writerow(row)

    print(f"Wrote {phase_name} comparison to {output_path}")


def print_comparison_summary(phase_results: dict, phase_name: str):
    """Print a summary comparison table to terminal."""
    if not phase_results:
        print(f"\n=== {phase_name} Comparison ===")
        print("No data available")
        return

    all_ops = get_all_ops(phase_results)
    prompt_lengths = [p for p in PROMPT_ORDER if p in phase_results]

    if not prompt_lengths:
        return

    # Calculate total time per prompt length
    totals = {}
    for pl in prompt_lengths:
        totals[pl] = sum(stats.get("total_time_us", 0) for stats in phase_results.get(pl, {}).values())

    print(f"\n{'='*100}")
    print(f"=== {phase_name} Comparison Summary ===")
    print(f"{'='*100}")

    # Print total times
    print("\nTotal Time by Prompt Length:")
    print("-" * 80)
    header = "Prompt Length".ljust(15)
    for pl in prompt_lengths:
        header += pl.rjust(12)
    print(header)
    print("-" * 80)

    time_row = "Time (us)".ljust(15)
    for pl in prompt_lengths:
        time_row += f"{totals[pl]:.2f}".rjust(12)
    print(time_row)

    time_ms_row = "Time (ms)".ljust(15)
    for pl in prompt_lengths:
        time_ms_row += f"{totals[pl]/1000:.2f}".rjust(12)
    print(time_ms_row)

    # Calculate scaling relative to first prompt length
    if len(prompt_lengths) > 1 and totals[prompt_lengths[0]] > 0:
        scale_row = f"Scale vs {prompt_lengths[0]}".ljust(15)
        base = totals[prompt_lengths[0]]
        for pl in prompt_lengths:
            scale = totals[pl] / base
            scale_row += f"{scale:.2f}x".rjust(12)
        print(scale_row)

    print()

    # Print per-op breakdown (top 10 by max time across all prompt lengths)
    op_max_times = {}
    for op_name in all_ops:
        max_time = max(phase_results.get(pl, {}).get(op_name, {}).get("total_time_us", 0) for pl in prompt_lengths)
        op_max_times[op_name] = max_time

    top_ops = sorted(op_max_times.items(), key=lambda x: x[1], reverse=True)[:15]

    print("\nTop 15 Operations by Time (us):")
    print("-" * 100)

    # Calculate column widths
    max_op_len = max(len(op) for op, _ in top_ops)
    max_op_len = max(max_op_len, len("Operation"))

    header = "Operation".ljust(max_op_len + 2)
    for pl in prompt_lengths:
        header += pl.rjust(12)
    header += "  Scaling"
    print(header)
    print("-" * 100)

    for op_name, _ in top_ops:
        row = op_name.ljust(max_op_len + 2)
        times = []
        for pl in prompt_lengths:
            t = phase_results.get(pl, {}).get(op_name, {}).get("total_time_us", 0)
            times.append(t)
            row += f"{t:.2f}".rjust(12)

        # Calculate scaling
        if len(times) > 1 and times[0] > 0:
            scale = times[-1] / times[0]
            row += f"  {scale:.2f}x"
        else:
            row += "  -"
        print(row)

    print()

    # Print percentage breakdown for each prompt length
    print("\nPercentage Breakdown by Prompt Length:")
    print("-" * 100)

    for pl in prompt_lengths:
        print(f"\n{pl}:")
        total = totals.get(pl, 0)
        if total == 0:
            print("  No data")
            continue

        # Get top 5 ops by percentage for this prompt length
        op_times = [(op, phase_results.get(pl, {}).get(op, {}).get("total_time_us", 0)) for op in all_ops]
        op_times.sort(key=lambda x: x[1], reverse=True)

        for op_name, time_us in op_times[:5]:
            pct = (time_us / total) * 100
            print(f"  {op_name}: {time_us:.2f} us ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Compare profiler results across prompt lengths")
    parser.add_argument("sweep_dir", help="Directory containing sweep results (with 128/, 4k/, etc. subdirs)")
    parser.add_argument("--output-dir", help="Directory to write comparison files (default: sweep_dir)")

    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.exists():
        print(f"Error: Directory not found: {sweep_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else sweep_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Collecting results from: {sweep_dir}")
    print(f"Output directory: {output_dir}")

    # Collect all results
    results = collect_all_results(sweep_dir)

    # Write comparison CSVs
    write_comparison_csv(results["prefill"], output_dir / "prefill_comparison.csv", "Prefill")
    write_comparison_csv(results["decode"], output_dir / "decode_comparison.csv", "Decode")

    # Print summaries
    print_comparison_summary(results["prefill"], "Prefill")
    print_comparison_summary(results["decode"], "Decode")


if __name__ == "__main__":
    main()
