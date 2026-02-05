#!/usr/bin/env python3
"""
Script to analyze ops performance reports from generated/profiler/reports/
Aggregates DEVICE KERNEL DURATION across iterations and devices for each operation.
No external dependencies required - uses only Python standard library.
"""

import csv
import statistics
from collections import defaultdict
from pathlib import Path


def find_latest_report(base_dir="generated/profiler/reports"):
    """Find the latest ops performance report directory."""
    reports_dir = Path(base_dir)

    if not reports_dir.exists():
        raise FileNotFoundError(f"Reports directory not found: {reports_dir}")

    # Find all subdirectories with date pattern
    report_dirs = [d for d in reports_dir.iterdir() if d.is_dir()]

    if not report_dirs:
        raise FileNotFoundError(f"No report directories found in {reports_dir}")

    # Sort by directory name (which contains timestamps)
    latest_dir = max(report_dirs, key=lambda d: d.name)

    # Find the CSV file in the directory
    csv_files = list(latest_dir.glob("ops_perf_results_*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {latest_dir}")

    return csv_files[0]


def analyze_report(csv_path):
    """Analyze the ops performance report and aggregate results."""

    print(f"Reading report from: {csv_path}")

    # Columns to keep for context (input shapes, etc.)
    context_cols = [
        "INPUT_0_W_PAD[LOGICAL]",
        "INPUT_0_Z_PAD[LOGICAL]",
        "INPUT_0_Y_PAD[LOGICAL]",
        "INPUT_0_X_PAD[LOGICAL]",
        "INPUT_1_W_PAD[LOGICAL]",
        "INPUT_1_Z_PAD[LOGICAL]",
        "INPUT_1_Y_PAD[LOGICAL]",
        "INPUT_1_Z_PAD[LOGICAL]",
        "OUTPUT_0_W_PAD[LOGICAL]",
        "OUTPUT_0_Z_PAD[LOGICAL]",
        "OUTPUT_0_Y_PAD[LOGICAL]",
        "OUTPUT_0_X_PAD[LOGICAL]",
        "CORE COUNT",
    ]

    # Data structure to hold aggregated results
    # key: (op_code, shape_identifier), value: list of durations
    op_data = defaultdict(lambda: {"durations": [], "context": {}, "op_code": None, "order": None})

    order_counter = 0
    found_matmul = False

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames

        # Check required columns
        if "OP CODE" not in headers:
            raise ValueError("'OP CODE' column not found in CSV")
        if "DEVICE KERNEL DURATION [ns]" not in headers:
            raise ValueError("'DEVICE KERNEL DURATION [ns]' column not found in CSV")

        # Find existing context columns
        existing_context_cols = [col for col in context_cols if col in headers]

        print(f"Total columns: {len(headers)}")
        print(f"Context columns found: {len(existing_context_cols)}")

        row_count = 0
        skipped_count = 0
        for row in reader:
            row_count += 1

            # Skip rows until we find the first Matmul operation
            if not found_matmul:
                op_code = row["OP CODE"]
                if "matmul::MatmulDeviceOperation" in op_code or "Matmul" in op_code:
                    found_matmul = True
                    print(f"Found first Matmul operation at row {row_count}")
                else:
                    skipped_count += 1
                    continue

            op_code = row["OP CODE"]
            duration_str = row["DEVICE KERNEL DURATION [ns]"]

            # Skip if duration is empty or invalid
            if not duration_str or duration_str == "":
                continue

            try:
                duration = float(duration_str)
            except ValueError:
                continue

            # Create a shape identifier from input/output dimensions
            shape_parts = []
            for col in [
                "INPUT_0_Y_PAD[LOGICAL]",
                "INPUT_0_X_PAD[LOGICAL]",
                "INPUT_1_Y_PAD[LOGICAL]",
                "INPUT_1_X_PAD[LOGICAL]",
                "OUTPUT_0_Y_PAD[LOGICAL]",
                "OUTPUT_0_X_PAD[LOGICAL]",
            ]:
                if col in row:
                    shape_parts.append(row[col])

            shape_id = "_".join(shape_parts)
            op_key = (op_code, shape_id)

            # Store data
            if op_data[op_key]["order"] is None:
                op_data[op_key]["order"] = order_counter
                order_counter += 1

            op_data[op_key]["durations"].append(duration)
            op_data[op_key]["op_code"] = op_code

            # Store context (first occurrence)
            if not op_data[op_key]["context"]:
                for col in existing_context_cols:
                    if col in row:
                        op_data[op_key]["context"][col] = row[col]

        print(f"Total rows processed: {row_count}")
        print(f"Rows skipped (before first Matmul): {skipped_count}")

    # Calculate statistics for each operation
    results = []
    for op_key, data in op_data.items():
        durations = data["durations"]
        if not durations:
            continue

        avg_duration = statistics.mean(durations)
        min_duration = min(durations)
        max_duration = max(durations)
        std_duration = statistics.stdev(durations) if len(durations) > 1 else 0
        count = len(durations)

        result = {
            "order": data["order"],
            "OP CODE": data["op_code"],
            "Avg Duration (ns)": avg_duration,
            "Min Duration (ns)": min_duration,
            "Max Duration (ns)": max_duration,
            "Std Duration (ns)": std_duration,
            "Sample Count": count,
            "Avg Duration (μs)": avg_duration / 1000,
        }

        # Add context
        result.update(data["context"])

        results.append(result)

    # Sort by original order
    results.sort(key=lambda x: x["order"])

    # Add a total row with sum of all average durations
    if results:
        total_avg_duration = sum(r["Avg Duration (ns)"] for r in results)
        total_row = {
            "order": len(results) + 1,
            "OP CODE": "=== TOTAL ===",
            "Avg Duration (ns)": total_avg_duration,
            "Min Duration (ns)": "",
            "Max Duration (ns)": "",
            "Std Duration (ns)": "",
            "Sample Count": sum(r["Sample Count"] for r in results),
            "Avg Duration (μs)": total_avg_duration / 1000,
        }
        results.append(total_row)

    return results


def format_table(results):
    """Format results as a text table."""
    if not results:
        return "No results to display"

    # Define columns to display
    display_cols = [
        "OP CODE",
        "Avg Duration (μs)",
        "Min Duration (ns)",
        "Max Duration (ns)",
        "Std Duration (ns)",
        "Sample Count",
        "INPUT_0_Y_PAD[LOGICAL]",
        "INPUT_0_X_PAD[LOGICAL]",
        "INPUT_1_Y_PAD[LOGICAL]",
        "INPUT_1_X_PAD[LOGICAL]",
        "OUTPUT_0_Y_PAD[LOGICAL]",
        "OUTPUT_0_X_PAD[LOGICAL]",
    ]

    # Filter to only columns that exist in results
    display_cols = [col for col in display_cols if col in results[0]]

    # Calculate column widths
    col_widths = {}
    for col in display_cols:
        max_width = len(col)
        for row in results:
            if col in row:
                value = str(row[col])
                if col == "OP CODE" and len(value) > 60:
                    value = value[:57] + "..."
                max_width = max(max_width, len(value))
        col_widths[col] = min(max_width, 60)

    # Build header
    header = " | ".join(col.ljust(col_widths[col]) for col in display_cols)
    separator = "-+-".join("-" * col_widths[col] for col in display_cols)

    lines = [header, separator]

    # Build rows
    for row in results:
        values = []
        for col in display_cols:
            value = str(row.get(col, ""))
            if col == "OP CODE" and len(value) > 60:
                value = value[:57] + "..."
            elif col == "Avg Duration (μs)":
                try:
                    value = f"{float(value):.3f}"
                except:
                    pass
            elif "Duration" in col and col != "OP CODE":
                try:
                    value = f"{float(value):.1f}"
                except:
                    pass
            values.append(value.ljust(col_widths[col]))
        lines.append(" | ".join(values))

    return "\n".join(lines)


def save_csv(results, output_path):
    """Save results to CSV file."""
    if not results:
        return

    # Get all keys from all results
    all_keys = set()
    for result in results:
        all_keys.update(result.keys())

    # Remove 'order' as it's internal
    all_keys.discard("order")

    # Preferred column order
    preferred_order = [
        "OP CODE",
        "Avg Duration (ns)",
        "Avg Duration (μs)",
        "Min Duration (ns)",
        "Max Duration (ns)",
        "Std Duration (ns)",
        "Sample Count",
    ]

    # Add remaining keys
    fieldnames = [k for k in preferred_order if k in all_keys]
    fieldnames.extend(sorted(all_keys - set(fieldnames)))

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({k: result.get(k, "") for k in fieldnames})


def main():
    try:
        # Find the latest report
        csv_path = find_latest_report()

        # Analyze the report
        results = analyze_report(csv_path)

        # Display results
        print("\n" + "=" * 80)
        print("AGGREGATED OPERATION PERFORMANCE")
        print("=" * 80)
        print(format_table(results))

        # Save to CSV with date from report path
        report_date = csv_path.parent.name  # Extract date from path like "2026_01_28_15_25_17"
        output_file = Path(f"aggregated_ops_performance_{report_date}.csv")
        save_csv(results, output_file)
        print(f"\n\nResults saved to: {output_file.absolute()}")

        # Print summary statistics
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        print(f"Total unique operations: {len(results) - 1}")  # Exclude the TOTAL row
        total_samples = sum(r["Sample Count"] for r in results if r["OP CODE"] != "=== TOTAL ===")
        print(f"Total samples analyzed: {total_samples}")

        # Show total duration
        total_result = next((r for r in results if r["OP CODE"] == "=== TOTAL ==="), None)
        if total_result:
            print(
                f"\nTotal aggregated duration: {total_result['Avg Duration (μs)']:.3f} μs ({total_result['Avg Duration (ns)']:.1f} ns)"
            )

        # Top 10 slowest operations (excluding TOTAL)
        print(f"\nTop 10 slowest operations (by average duration):")
        sorted_results = sorted(
            [r for r in results if r["OP CODE"] != "=== TOTAL ==="], key=lambda x: x["Avg Duration (ns)"], reverse=True
        )[:10]

        for i, result in enumerate(sorted_results, 1):
            op_code = result["OP CODE"]
            if len(op_code) > 70:
                op_code = op_code[:67] + "..."
            print(
                f"{i:2d}. {op_code:70s} | {result['Avg Duration (μs)']:8.3f} μs | {result['Sample Count']:3d} samples"
            )

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
