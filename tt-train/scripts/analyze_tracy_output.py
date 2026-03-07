# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Analyze Tracy CSV profiler output for operation performance.

This script analyzes performance data from Tracy profiler CSV exports. It supports
two analysis modes:

1. MARKERS MODE
   Measures operations between profiler begin/end markers. Automatically handles
   grouped markers (10 consecutive ProfilerNoopOperation markers are treated as one
   marker group). Sums DEVICE KERNEL DURATION of all operations between markers.

   Use when: You've instrumented code with ProfilerNoopOperation markers to measure
   composite operations (e.g., composite SwiGLU = 3 matmuls + silu + mul).

2. OPERATIONS MODE
   Directly measures specific operations by name/regex pattern. Extracts DEVICE
   KERNEL DURATION from each matching operation.

   Use when: Measuring fused/single operations (e.g., fused SwiGLU kernel).

TRAINING STEP WARMUP SKIP
  --skip-first N --ops-per-step M
  Skips first N training steps by skipping N*M operations. Useful for excluding
  warmup iterations. For example, nanoGPT has 6 SwiGLUs per training step, so:
  --skip-first 1 --ops-per-step 6  # Skip first training step (6 ops)

EXAMPLES

  # Analyze composite SwiGLU (with profiler markers)
  python analyze_tracy_output.py profile.csv --mode markers \
    --begin-marker swiglu_fw_composite_start \
    --end-marker swiglu_fw_composite_end \
    --skip-first 1 --ops-per-step 6

  # Analyze fused SwiGLU (direct operation)
  python analyze_tracy_output.py profile.csv --mode operations \
    --op-name SwiGLU \
    --skip-first 1 --ops-per-step 6

  # Use regex to match multiple operation types
  python analyze_tracy_output.py profile.csv --mode operations \
    --op-name "Matmul|SwiGLU" \
    --unit us

OUTPUT
  Prints per-operation/group timings and summary statistics:
  - Mean, std dev, min/max
  - Total time across all measured operations
  - Iteration counts (total and used after skipping)
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np


def to_int(x):
    """
    Convert value to int, return None if invalid.

    Args:
        x: Value to convert (can be string, int, float, or pd.NA)

    Returns:
        int or None: Converted integer value, or None if conversion fails
    """
    try:
        if pd.isna(x):
            return None
        return int(str(x).strip())
    except Exception:
        return None


def is_marker(row, marker_name):
    """
    Check if row is a ProfilerNoopOperation marker with given name.

    Args:
        row: DataFrame row to check
        marker_name: Marker identifier to look for in ATTRIBUTES column

    Returns:
        bool: True if row is a ProfilerNoopOperation with matching marker name
    """
    attr = str(row.get("ATTRIBUTES", ""))
    opc = str(row.get("OP CODE", ""))
    return marker_name in attr and "ProfilerNoopOperation" in opc


def analyze_markers(df, begin_marker, end_marker):
    """
    Find operations between begin/end markers and sum their DEVICE KERNEL DURATION.

    Handles grouped markers: consecutive ProfilerNoopOperation markers (gap <= 5 rows)
    are treated as one marker group. Uses the LAST begin marker and FIRST end marker
    from each group to define the measurement interval.

    Args:
        df: DataFrame containing Tracy profiler CSV data
        begin_marker: Identifier for begin marker (e.g., "swiglu_fw_composite_start")
        end_marker: Identifier for end marker (e.g., "swiglu_fw_composite_end")

    Returns:
        list[int]: List of total durations in nanoseconds, one per marker group
    """
    # Find all marker positions
    begins = []
    ends = []

    for idx, row in df.iterrows():
        if is_marker(row, begin_marker):
            begins.append(idx)
        elif is_marker(row, end_marker):
            ends.append(idx)

    if not begins or not ends:
        print(f"[ERROR] Found {len(begins)} begin markers and {len(ends)} end markers")
        return []

    print(f"[INFO] Found {len(begins)} begin markers and {len(ends)} end markers")

    # Group consecutive begin markers (gap <= 5 rows)
    begin_groups = []
    i = 0
    while i < len(begins):
        group = [begins[i]]
        while i + 1 < len(begins) and begins[i + 1] - begins[i] <= 5:
            i += 1
            group.append(begins[i])
        begin_groups.append(group)
        i += 1

    print(f"[INFO] Grouped into {len(begin_groups)} marker groups")

    # For each marker group, sum device durations between last begin and first end
    durations = []

    for group_idx, begin_group in enumerate(begin_groups):
        last_begin = begin_group[-1]  # Last begin marker in group

        # Find first end marker after last begin
        end_marker_idx = None
        for end_idx in ends:
            if end_idx > last_begin:
                end_marker_idx = end_idx
                break

        if not end_marker_idx:
            continue

        # Sum DEVICE KERNEL DURATION of all ops between markers
        total_duration = 0
        op_count = 0

        for _, row in df.iloc[last_begin + 1 : end_marker_idx].iterrows():
            # Skip ProfilerNoopOperation markers
            if "ProfilerNoopOperation" in str(row.get("OP CODE", "")):
                continue

            device_duration = to_int(row.get("DEVICE KERNEL DURATION [ns]"))
            if device_duration and device_duration > 0:
                total_duration += device_duration
                op_count += 1

        if total_duration > 0:
            durations.append(total_duration)
            print(
                f"[INFO] Group {group_idx}: {len(begin_group)} markers, {op_count} ops -> {total_duration / 1e6:.3f} ms"
            )

    return durations


def analyze_operations(df, op_pattern):
    """
    Find operations matching name/regex and extract their DEVICE KERNEL DURATION.

    Args:
        df: DataFrame containing Tracy profiler CSV data
        op_pattern: Operation name or regex pattern (case-insensitive)
                   Examples: "SwiGLU", "Matmul", "SwiGLU|Matmul"

    Returns:
        list[int]: List of durations in nanoseconds, one per matching operation
    """
    # Match operations by name (case-insensitive substring or regex)
    mask = df["OP CODE"].str.contains(op_pattern, case=False, na=False, regex=True)
    ops = df[mask]

    print(f"[INFO] Found {len(ops)} operations matching '{op_pattern}'")

    durations = []
    for op_idx, (idx, row) in enumerate(ops.iterrows()):
        device_duration = to_int(row.get("DEVICE KERNEL DURATION [ns]"))
        if device_duration and device_duration > 0:
            durations.append(device_duration)
            print(f"[INFO] Op {op_idx}: {device_duration / 1e6:.3f} ms")

    return durations


def print_stats(durations, unit="ms", skip_first=0):
    """
    Print performance statistics for the collected durations.

    Args:
        durations: List of duration values in nanoseconds
        unit: Time unit for display ("ns", "us", "ms", or "s")
        skip_first: Number of operations that were skipped (for display only)
    """
    if not durations:
        print("[ERROR] No durations to analyze")
        return

    # Unit conversion
    if unit == "ns":
        scale = 1
    elif unit == "us":
        scale = 1e3
    elif unit == "ms":
        scale = 1e6
    elif unit == "s":
        scale = 1e9
    else:
        scale = 1e6

    values = [d / scale for d in durations]

    print(f"\n=== Performance Summary ===")
    print(
        f"Iterations: {len(durations) + skip_first} total, {len(durations)} used (skipped first {skip_first})"
    )
    print(f"Mean:       {np.mean(values):.3f} {unit}")
    print(f"Std dev:    {np.std(values, ddof=1):.3f} {unit}")
    print(f"Min/Max:    {np.min(values):.3f} / {np.max(values):.3f} {unit}")
    print(f"Total:      {np.sum(values):.3f} {unit}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Tracy profiler CSV output.")
    parser.add_argument("csv", help="Path to CSV file from Tracy profiler")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["markers", "operations"],
        help="Analysis mode: 'markers' for begin/end markers, 'operations' for specific ops",
    )
    parser.add_argument("--begin-marker", help="Begin marker name (for markers mode)")
    parser.add_argument("--end-marker", help="End marker name (for markers mode)")
    parser.add_argument(
        "--op-name", help="Operation name or regex pattern (for operations mode)"
    )
    parser.add_argument(
        "--skip-first", type=int, default=0, help="Skip first N training steps (warmup)"
    )
    parser.add_argument(
        "--ops-per-step",
        type=int,
        default=1,
        help="Number of operations per training step (e.g., 6 for nanoGPT with 6 SwiGLUs/step)",
    )
    parser.add_argument(
        "--unit",
        default="ms",
        choices=["ns", "us", "ms", "s"],
        help="Time unit for output",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.mode == "markers" and (not args.begin_marker or not args.end_marker):
        print("[ERROR] --begin-marker and --end-marker required for markers mode")
        sys.exit(1)

    if args.mode == "operations" and not args.op_name:
        print("[ERROR] --op-name required for operations mode")
        sys.exit(1)

    # Load CSV
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[ERROR] File not found: {csv_path}")
        sys.exit(1)

    print(f"[INFO] Loading {csv_path}...")
    try:
        df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
    except Exception as e:
        print(f"[ERROR] Failed to read CSV: {e}")
        sys.exit(1)

    # Analyze based on mode
    if args.mode == "markers":
        durations = analyze_markers(df, args.begin_marker, args.end_marker)
    else:
        durations = analyze_operations(df, args.op_name)

    if not durations:
        print("[ERROR] No measurements found")
        sys.exit(1)

    # Apply skip (skip N training steps = N * ops_per_step operations)
    skip_ops = max(0, args.skip_first * args.ops_per_step)
    if skip_ops >= len(durations):
        print(
            f"[ERROR] Cannot skip {args.skip_first} steps ({skip_ops} ops), only {len(durations)} available"
        )
        sys.exit(1)

    used_durations = durations[skip_ops:]

    # Print results
    print_stats(used_durations, args.unit, skip_ops)


if __name__ == "__main__":
    main()
