#!/usr/bin/env python3
"""
Compare two profiler runs op-by-op and output to Excel with separate sheets per prompt length.

This script compares profiler results from two different runs (e.g., baseline vs optimized)
by matching operations using their CALL_IDX. It produces an Excel file with:
  - Summary sheet: Total times per prompt length and phase with diff percentages
  - Individual sheets: Op-by-op comparison for each prompt length (128, 4k, 8k, etc.)
    and phase (prefill, decode)

Features:
  - Matches ops by row index (execution order) - assumes same op sequence in both runs
  - Aggregates times for ops with multiple kernel executions (same CALL_IDX)
  - Shows both runs' OP_NAME and CALL_IDX so you can spot mismatches
  - STATUS column flags: OP_MISMATCH (different ops at same row), MISSING_IN_<run>
  - Data bars on TIME_us columns for visual comparison
  - Color-coded DIFF_%: green (improvement) / red (regression) when |diff| > 50us AND |%| > 5%

Prerequisites:
  - Both runs must have been parsed with parse_profiler_report.py (prefill.csv, decode.csv)
  - Requires xlsxwriter Python package (available in python_env)

Usage:
    source python_env/bin/activate
    python scripts/compare_two_runs.py <run1_name> <run2_name> [options]

Arguments:
    run1_name           First run name (baseline) - directory name under output-dir
    run2_name           Second run name (to compare) - directory name under output-dir

Options:
    --output-dir DIR    Base directory containing runs (default: profiler_sweep_results)
    --output-file FILE  Output Excel filename (default: comparison_<run1>_vs_<run2>.xlsx)
    -h, --help          Show this help message

Output:
    Excel file with sheets:
      - Summary: Overview of total times and differences per prompt length/phase
      - prefill_128, decode_128, prefill_4k, decode_4k, ... : Op-by-op comparisons

    Columns in op-by-op sheets:
      - ROW: Row index (execution order)
      - <run1>_CALL_IDX, <run2>_CALL_IDX: Global call count from each run
      - <run1>_OP, <run2>_OP: Operation names from each run (check for mismatches)
      - STATUS: Empty, OP_MISMATCH (different ops), or MISSING_IN_<run>
      - <run1>_TIME_us: Kernel duration in run1 (with data bar)
      - <run2>_TIME_us: Kernel duration in run2 (with data bar)
      - DIFF_us: Absolute difference (run2 - run1)
      - DIFF_%: Percentage difference (color-coded if significant)
      - <run1>_CORES, <run2>_CORES: Core counts
      - IN0_SHAPE, OUT0_SHAPE: Tensor shapes

Examples:
    # Compare baseline vs optimized run
    python scripts/compare_two_runs.py baseline 20260129_022518

    # Compare with custom output filename
    python scripts/compare_two_runs.py baseline optimized --output-file perf_comparison.xlsx

    # Compare runs in a different directory
    python scripts/compare_two_runs.py run1 run2 --output-dir /path/to/results

    # Via shell script
    ./scripts/run_profiler_sweep.sh --compare-runs baseline 20260129_022518
"""

import argparse
import csv
import sys
from pathlib import Path

# Prompt length order
PROMPT_ORDER = ["128", "4k", "8k", "16k", "32k", "64k", "128k"]


def read_csv(csv_path: Path) -> list:
    """Read a CSV file and return list of row dicts."""
    if not csv_path.exists():
        return []

    data = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def aggregate_by_call_idx(data: list) -> list:
    """Aggregate data by CALL_IDX, summing times for ops with same CALL_IDX.

    Returns list of dicts sorted by CALL_IDX, each containing:
    {call_idx, total_time_us, op_name, cores, in0_shape, out0_shape, count}
    """
    aggregated = {}
    for row in data:
        call_idx = row.get("CALL_IDX", "")
        if not call_idx:
            continue

        try:
            time_us = float(row.get("KERNEL_DUR_us", 0) or 0)
        except (ValueError, TypeError):
            time_us = 0.0

        if call_idx not in aggregated:
            aggregated[call_idx] = {
                "call_idx": call_idx,
                "total_time_us": 0.0,
                "op_name": row.get("OP_NAME", ""),
                "cores": row.get("CORES", 0),
                "in0_shape": row.get("IN0_SHAPE", ""),
                "out0_shape": row.get("OUT0_SHAPE", ""),
                "count": 0,
            }

        aggregated[call_idx]["total_time_us"] += time_us
        aggregated[call_idx]["count"] += 1

    # Sort by CALL_IDX and return as list
    try:
        sorted_items = sorted(
            aggregated.values(), key=lambda x: int(x["call_idx"]) if x["call_idx"].isdigit() else float("inf")
        )
    except (ValueError, TypeError):
        sorted_items = sorted(aggregated.values(), key=lambda x: x["call_idx"])

    return sorted_items


def compare_ops(run1_data: list, run2_data: list, run1_name: str, run2_name: str) -> list:
    """Compare two lists of ops by row index.

    Aggregates times for ops with the same CALL_IDX (multiple kernel executions),
    then compares by row index (execution order). This assumes both runs execute
    ops in the same order.

    Returns list of comparison dicts with columns from both runs and diff calculations.
    """
    # Aggregate by CALL_IDX (sum times for same op), then convert to sorted list
    run1_agg = aggregate_by_call_idx(run1_data)
    run2_agg = aggregate_by_call_idx(run2_data)

    # Compare by row index (execution order)
    max_len = max(len(run1_agg), len(run2_agg))

    comparisons = []
    for i in range(max_len):
        agg1 = run1_agg[i] if i < len(run1_agg) else {}
        agg2 = run2_agg[i] if i < len(run2_agg) else {}

        # Get aggregated timing values (0 if row doesn't exist in that run)
        time1 = agg1.get("total_time_us", 0.0)
        time2 = agg2.get("total_time_us", 0.0)

        # Get op names from both runs
        op1 = agg1.get("op_name", "")
        op2 = agg2.get("op_name", "")

        # Determine display op name (prefer run1, fall back to run2)
        op_name = op1 or op2 or "N/A"

        # Mark if ops don't match or row is missing
        status = ""
        if not agg1:
            status = f"MISSING_IN_{run1_name}"
        elif not agg2:
            status = f"MISSING_IN_{run2_name}"
        elif op1 != op2:
            status = "OP_MISMATCH"

        # Calculate difference
        diff = time2 - time1
        pct_diff = ((time2 - time1) / time1 * 100) if time1 > 0 else (100.0 if time2 > 0 else 0.0)

        comparison = {
            "ROW": i + 1,
            f"{run1_name}_CALL_IDX": agg1.get("call_idx", ""),
            f"{run2_name}_CALL_IDX": agg2.get("call_idx", ""),
            f"{run1_name}_OP": op1,
            f"{run2_name}_OP": op2,
            "STATUS": status,
            f"{run1_name}_TIME_us": time1,
            f"{run2_name}_TIME_us": time2,
            "DIFF_us": diff,
            "DIFF_%": pct_diff,
            f"{run1_name}_CORES": agg1.get("cores", 0),
            f"{run2_name}_CORES": agg2.get("cores", 0),
            "IN0_SHAPE": agg1.get("in0_shape") or agg2.get("in0_shape", ""),
            "OUT0_SHAPE": agg1.get("out0_shape") or agg2.get("out0_shape", ""),
        }
        comparisons.append(comparison)

    return comparisons


def write_excel_with_xlsxwriter(output_path: Path, all_comparisons: dict, run1_name: str, run2_name: str):
    """Write comparisons to Excel using xlsxwriter."""
    import xlsxwriter

    workbook = xlsxwriter.Workbook(str(output_path))

    # Define formats
    header_format = workbook.add_format({"bold": True, "bg_color": "#4472C4", "font_color": "white", "border": 1})
    number_format = workbook.add_format({"num_format": "0.00", "border": 1})
    int_format = workbook.add_format({"num_format": "0", "border": 1})
    pct_format = workbook.add_format({"num_format": "0.00", "border": 1})
    green_format = workbook.add_format({"num_format": "0.00", "bg_color": "#C6EFCE", "border": 1})
    red_format = workbook.add_format({"num_format": "0.00", "bg_color": "#FFC7CE", "border": 1})
    text_format = workbook.add_format({"border": 1})

    # Summary sheet first
    summary_ws = workbook.add_worksheet("Summary")
    summary_headers = [
        "Prompt_Length",
        "Phase",
        f"{run1_name}_Total_ms",
        f"{run2_name}_Total_ms",
        "Diff_ms",
        "Diff_%",
        "Op_Count",
    ]

    for col, header in enumerate(summary_headers):
        summary_ws.write(0, col, header, header_format)
        summary_ws.set_column(col, col, 18)

    row = 1
    for prompt_len in PROMPT_ORDER:
        for phase in ["prefill", "decode"]:
            key = f"{phase}_{prompt_len}"
            if key in all_comparisons and all_comparisons[key]:
                data = all_comparisons[key]
                total_run1 = sum(r[f"{run1_name}_TIME_us"] for r in data)
                total_run2 = sum(r[f"{run2_name}_TIME_us"] for r in data)
                diff = total_run2 - total_run1
                pct = (diff / total_run1 * 100) if total_run1 > 0 else 0

                summary_ws.write(row, 0, prompt_len, text_format)
                summary_ws.write(row, 1, phase, text_format)
                summary_ws.write(row, 2, round(total_run1 / 1000, 2), number_format)
                summary_ws.write(row, 3, round(total_run2 / 1000, 2), number_format)
                summary_ws.write(row, 4, round(diff / 1000, 2), number_format)

                # Color code diff percentage
                if pct < -1:
                    summary_ws.write(row, 5, round(pct, 2), green_format)
                elif pct > 1:
                    summary_ws.write(row, 5, round(pct, 2), red_format)
                else:
                    summary_ws.write(row, 5, round(pct, 2), pct_format)

                summary_ws.write(row, 6, len(data), int_format)
                row += 1

    # Individual sheets for each prompt length and phase
    for prompt_len in PROMPT_ORDER:
        for phase in ["prefill", "decode"]:
            key = f"{phase}_{prompt_len}"
            if key not in all_comparisons or not all_comparisons[key]:
                continue

            data = all_comparisons[key]
            sheet_name = f"{phase}_{prompt_len}"[:31]
            ws = workbook.add_worksheet(sheet_name)

            if data:
                headers = list(data[0].keys())

                # Write headers
                for col, header in enumerate(headers):
                    ws.write(0, col, header, header_format)
                    # Set column width
                    if "OP_NAME" in header:
                        ws.set_column(col, col, 40)
                    elif "SHAPE" in header:
                        ws.set_column(col, col, 30)
                    else:
                        ws.set_column(col, col, 15)

                # Track columns for gradient formatting
                time_columns = []

                # Write data
                for row_idx, row_data in enumerate(data, start=1):
                    for col_idx, header in enumerate(headers):
                        value = row_data.get(header, "")

                        # Track TIME_us columns for gradient formatting
                        if row_idx == 1 and "TIME_us" in header:
                            time_columns.append(col_idx)

                        if header == "DIFF_%":
                            if isinstance(value, (int, float)):
                                # Get absolute diff to check significance
                                abs_diff = abs(row_data.get("DIFF_us", 0))
                                pct_diff = value
                                # Only color if both: abs diff > 50us AND pct diff > 5%
                                if abs_diff > 50 and pct_diff < -5:
                                    ws.write(row_idx, col_idx, value, green_format)
                                elif abs_diff > 50 and pct_diff > 5:
                                    ws.write(row_idx, col_idx, value, red_format)
                                else:
                                    ws.write(row_idx, col_idx, value, pct_format)
                            else:
                                ws.write(row_idx, col_idx, value, text_format)
                        elif isinstance(value, float):
                            ws.write(row_idx, col_idx, value, number_format)
                        elif isinstance(value, int):
                            ws.write(row_idx, col_idx, value, int_format)
                        else:
                            ws.write(row_idx, col_idx, value, text_format)

                # Apply data bars to TIME_us columns
                # Bar length represents the value relative to max in column
                num_rows = len(data)
                for col_idx in time_columns:
                    ws.conditional_format(
                        1,
                        col_idx,
                        num_rows,
                        col_idx,
                        {
                            "type": "data_bar",
                            "min_type": "num",
                            "max_type": "max",
                            "min_value": 0,
                            "bar_color": "#638EC6",  # Blue bar
                            "bar_solid": True,
                            "bar_direction": "left",
                        },
                    )

    workbook.close()
    print(f"Wrote Excel file: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare two profiler runs op-by-op and generate Excel report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/compare_two_runs.py baseline 20260129_022518
  python scripts/compare_two_runs.py baseline optimized --output-file comparison.xlsx
  python scripts/compare_two_runs.py run1 run2 --output-dir /path/to/results

Output:
  Excel file with Summary sheet and per-prompt-length sheets (prefill_128, decode_128, etc.)
  Each sheet shows op-by-op comparison with timing differences and data bars.
        """,
    )
    parser.add_argument("run1", metavar="RUN1", help="First run name (baseline) - directory name under output-dir")
    parser.add_argument("run2", metavar="RUN2", help="Second run name (to compare) - directory name under output-dir")
    parser.add_argument(
        "--output-dir",
        default="profiler_sweep_results",
        help="Base directory containing runs (default: profiler_sweep_results)",
    )
    parser.add_argument(
        "--output-file", metavar="FILE", help="Output Excel filename (default: comparison_<run1>_vs_<run2>.xlsx)"
    )

    args = parser.parse_args()

    base_dir = Path(args.output_dir)
    run1_dir = base_dir / args.run1
    run2_dir = base_dir / args.run2

    # Validate directories exist
    if not run1_dir.exists():
        print(f"Error: Run directory not found: {run1_dir}")
        sys.exit(1)
    if not run2_dir.exists():
        print(f"Error: Run directory not found: {run2_dir}")
        sys.exit(1)

    print(f"Comparing runs:")
    print(f"  Run 1 (baseline): {args.run1}")
    print(f"  Run 2 (compare):  {args.run2}")
    print()

    # Collect all comparisons
    all_comparisons = {}

    for prompt_len in PROMPT_ORDER:
        run1_prompt_dir = run1_dir / prompt_len
        run2_prompt_dir = run2_dir / prompt_len

        if not run1_prompt_dir.exists() and not run2_prompt_dir.exists():
            continue

        print(f"Processing {prompt_len}...")

        # Compare prefill
        prefill1 = read_csv(run1_prompt_dir / "prefill.csv")
        prefill2 = read_csv(run2_prompt_dir / "prefill.csv")
        if prefill1 or prefill2:
            key = f"prefill_{prompt_len}"
            all_comparisons[key] = compare_ops(prefill1, prefill2, args.run1, args.run2)
            print(f"  prefill: {len(prefill1)} vs {len(prefill2)} ops")

        # Compare decode
        decode1 = read_csv(run1_prompt_dir / "decode.csv")
        decode2 = read_csv(run2_prompt_dir / "decode.csv")
        if decode1 or decode2:
            key = f"decode_{prompt_len}"
            all_comparisons[key] = compare_ops(decode1, decode2, args.run1, args.run2)
            print(f"  decode: {len(decode1)} vs {len(decode2)} ops")

    if not all_comparisons:
        print("No data found to compare!")
        sys.exit(1)

    # Output file
    if args.output_file:
        output_path = base_dir / args.output_file
    else:
        output_path = base_dir / f"comparison_{args.run1}_vs_{args.run2}.xlsx"

    # Use xlsxwriter
    write_excel_with_xlsxwriter(output_path, all_comparisons, args.run1, args.run2)

    print()
    print("Summary:")
    print("-" * 80)
    for prompt_len in PROMPT_ORDER:
        for phase in ["prefill", "decode"]:
            key = f"{phase}_{prompt_len}"
            if key in all_comparisons and all_comparisons[key]:
                data = all_comparisons[key]
                total_run1 = sum(row[f"{args.run1}_TIME_us"] for row in data)
                total_run2 = sum(row[f"{args.run2}_TIME_us"] for row in data)
                diff = total_run2 - total_run1
                pct = (diff / total_run1 * 100) if total_run1 > 0 else 0
                status = "FASTER" if pct < 0 else "SLOWER" if pct > 0 else "SAME"
                print(
                    f"  {prompt_len:>5} {phase:>7}: {args.run1}={total_run1/1000:.2f}ms, "
                    f"{args.run2}={total_run2/1000:.2f}ms, diff={pct:+.1f}% ({status})"
                )


if __name__ == "__main__":
    main()
