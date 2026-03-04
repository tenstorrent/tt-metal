#!/usr/bin/env python3
"""
Compare raw (non-aggregated) operation data across prompt lengths.

This script creates a side-by-side comparison of individual operation calls,
showing each op's timing at different prompt lengths. Unlike compare_profiler_results.py
which aggregates by operation name, this script preserves the execution order
and shows every single op invocation.

Use this script to:
  - See exact op-by-op execution sequence
  - Compare timing of specific op instances (not just totals)
  - Identify which specific op invocations scale poorly

=============================================================================
USAGE
=============================================================================

    python scripts/compare_ops_raw.py <sweep_results_dir> [options]

=============================================================================
ARGUMENTS
=============================================================================

    sweep_results_dir   Directory containing prompt length subdirectories
                        (e.g., profiler_sweep_results/baseline/)
                        Must contain subdirectories: 128/, 4k/, 8k/, etc.

=============================================================================
OPTIONS
=============================================================================

    --output-dir DIR    Directory to write output files
                        Default: same as sweep_results_dir

=============================================================================
OUTPUT FILES
=============================================================================

    prefill_raw_comparison.csv  - Raw op-by-op comparison for prefill phase
    decode_raw_comparison.csv   - Raw op-by-op comparison for decode phase

    CSV columns (for each prompt length):
      - ROW: Row index
      - <length>_OP: Operation name at this row for given prompt length
      - <length>_DUR: Kernel duration in microseconds
      - <length>_CORES: Number of cores used

=============================================================================
EXAMPLES
=============================================================================

    # Generate raw comparison for a run
    python scripts/compare_ops_raw.py profiler_sweep_results/baseline/

    # Save to different directory
    python scripts/compare_ops_raw.py profiler_sweep_results/baseline/ --output-dir analysis/

    # This is typically called automatically by run_profiler_sweep.sh
    ./scripts/run_profiler_sweep.sh --run-name baseline

=============================================================================
RELATED SCRIPTS
=============================================================================

    compare_profiler_results.py - Aggregated comparison (by op name)
    compare_two_runs.py         - Compare two different runs (Excel output)
    parse_profiler_report.py    - Parse raw profiler output
"""

import argparse
import csv
import sys
from pathlib import Path

# Prompt length order for display
PROMPT_ORDER = ["128", "4k", "8k", "16k", "32k", "64k", "128k"]


def read_phase_csv(csv_path: Path) -> list:
    """Read a prefill.csv or decode.csv file and return list of rows with occurrence tracking."""
    if not csv_path.exists():
        return []

    data = []
    op_occurrence = {}  # Track occurrence count for each op name

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            op_name = row.get("OP_NAME", "")

            # Track occurrence number for this op
            if op_name not in op_occurrence:
                op_occurrence[op_name] = 0
            op_occurrence[op_name] += 1
            occurrence = op_occurrence[op_name]

            # Create a unique key: OP_NAME#occurrence
            op_key = f"{op_name}#{occurrence}"

            data.append(
                {
                    "OP_NAME": op_name,
                    "OP_KEY": op_key,
                    "OCCURRENCE": occurrence,
                    "CALL_IDX": row.get("CALL_IDX", ""),
                    "KERNEL_DUR_us": row.get("KERNEL_DUR_us", "0"),
                    "CORES": row.get("CORES", "0"),
                }
            )
    return data


def create_raw_comparison(sweep_dir: Path, output_dir: Path, phase: str):
    """Create raw comparison CSV for a phase (prefill or decode).

    Matches operations by name and occurrence order (e.g., 1st MatmulDeviceOperation,
    2nd MatmulDeviceOperation, etc.) across different sequence lengths.
    """

    # Collect data from all prompt lengths
    all_data = {}
    all_op_keys = []  # Ordered list of unique op keys

    for prompt_len in PROMPT_ORDER:
        csv_path = sweep_dir / prompt_len / f"{phase}.csv"
        if csv_path.exists():
            data = read_phase_csv(csv_path)
            all_data[prompt_len] = {row["OP_KEY"]: row for row in data}

            # Collect all op keys in order (use first file's order as reference)
            if not all_op_keys:
                all_op_keys = [row["OP_KEY"] for row in data]
            else:
                # Add any new keys not seen before (in their order of appearance)
                existing_keys = set(all_op_keys)
                for row in data:
                    if row["OP_KEY"] not in existing_keys:
                        all_op_keys.append(row["OP_KEY"])
                        existing_keys.add(row["OP_KEY"])

            print(f"  {prompt_len}: {len(data)} rows")

    if not all_data:
        print(f"No {phase} data found")
        return

    prompt_lengths = [p for p in PROMPT_ORDER if p in all_data]

    # Write comparison CSV
    output_path = output_dir / f"{phase}_raw_comparison.csv"

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header: OP_NAME, OCCURRENCE, '', 128_dur, 128_cores, '', 4k_dur, 4k_cores, ...
        header = ["OP_NAME", "OCCURRENCE"]
        for i, pl in enumerate(prompt_lengths):
            header.append("")  # Empty separator column
            header.extend([f"{pl}_dur_us", f"{pl}_cores"])
        writer.writerow(header)

        # Write rows - match by op_key (OP_NAME#occurrence)
        for op_key in all_op_keys:
            # Parse op_name and occurrence from key
            op_name, occurrence = op_key.rsplit("#", 1)

            row = [op_name, occurrence]

            for pl in prompt_lengths:
                row.append("")  # Empty separator column
                op_data = all_data[pl].get(op_key)
                if op_data:
                    row.append(op_data["KERNEL_DUR_us"])
                    row.append(op_data["CORES"])
                else:
                    row.append("0")
                    row.append("0")

            writer.writerow(row)

    print(f"Wrote {phase} raw comparison ({len(all_op_keys)} rows) to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare raw operation data across prompt lengths")
    parser.add_argument("sweep_dir", help="Directory containing sweep results")
    parser.add_argument("--output-dir", help="Directory to write output files (default: sweep_dir)")

    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.exists():
        print(f"Error: Directory not found: {sweep_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else sweep_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Collecting results from: {sweep_dir}")
    print(f"Output directory: {output_dir}")

    print("\nProcessing prefill...")
    create_raw_comparison(sweep_dir, output_dir, "prefill")

    print("\nProcessing decode...")
    create_raw_comparison(sweep_dir, output_dir, "decode")


if __name__ == "__main__":
    main()
