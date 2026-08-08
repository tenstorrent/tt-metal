#!/usr/bin/env python3
"""
Script to extract device kernel durations from existing Binary Power profiler reports
and regenerate the CSV file.
"""

import csv
import shutil
from pathlib import Path
from datetime import datetime

# Test configurations (same as main script)
CONFIGS = [
    ("[32, 32]", "L1_interleaved"),
    ("[64, 128]", "L1_interleaved"),
    ("[512, 512]", "L1_block_sharded"),
    ("[25600, 128]", "L1_height_sharded"),
    ("[1, 16, 320, 320]", "height_sharded"),
]

PROFILER_REPORTS_DIR = Path("/home/ubuntu/tt-metal/generated/profiler/reports")
OUTPUT_CSV = Path("/home/ubuntu/tt-metal/fmod_perf_reports/perf_results.csv")
RAW_CSV_DIR = Path("/home/ubuntu/tt-metal/fmod_perf_reports/profiler_csvs")


def extract_duration_from_csv(test_num):
    """Extract and SUM all DEVICE KERNEL DURATION [ns] from profiler CSV

    Returns: (total_duration, source_csv_path) or (None, None)
    """

    test_name = f"fmod_test_{test_num}"
    test_dir = PROFILER_REPORTS_DIR / test_name

    if not test_dir.exists():
        print(f"  ✗ Directory not found: {test_dir}")
        return None, None

    # Find most recent timestamp directory
    timestamp_dirs = sorted([d for d in test_dir.iterdir() if d.is_dir()], reverse=True)
    if not timestamp_dirs:
        print(f"  ✗ No timestamp directories in {test_dir}")
        return None, None

    latest_timestamp = timestamp_dirs[0]

    # Find CSV file
    csv_files = list(latest_timestamp.glob("ops_perf_results_*.csv"))
    if not csv_files:
        print(f"  ✗ No CSV file in {latest_timestamp}")
        return None, None

    csv_file = csv_files[0]

    try:
        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)

            if len(rows) < 2:
                print(f"  ✗ CSV has insufficient data")
                return None, None

            # Parse header
            header = rows[0]

            # Find DEVICE KERNEL DURATION [ns] column index
            duration_col_idx = None
            for idx, col in enumerate(header):
                if "DEVICE KERNEL DURATION" in col and "[ns]" in col:
                    duration_col_idx = idx
                    break

            if duration_col_idx is None:
                print(f"  ✗ Column 'DEVICE KERNEL DURATION [ns]' not found")
                return None, None

            # Sum durations from ALL data rows
            total_duration = 0.0
            row_count = 0
            for row in rows[1:]:
                if not row:
                    continue

                if duration_col_idx >= len(row):
                    continue

                duration_str = row[duration_col_idx].strip()
                if duration_str:
                    try:
                        total_duration += float(duration_str)
                        row_count += 1
                    except ValueError:
                        continue

            if row_count == 0:
                print(f"  ✗ No valid duration values found")
                return None, None

            print(f"  ✓ Total Duration: {total_duration} ns (sum of {row_count} ops, from {csv_file.name})")
            return total_duration, csv_file

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None, None


def main():
    """Extract durations and regenerate CSV"""

    print("=" * 80)
    print("EXTRACTING FMOD DEVICE KERNEL DURATIONS FROM PROFILER REPORTS")
    print("=" * 80)
    print(f"Profiler reports: {PROFILER_REPORTS_DIR}")
    print(f"Output CSV: {OUTPUT_CSV}")
    print(f"Raw CSVs: {RAW_CSV_DIR}")
    print("=" * 80)
    print()

    # Create raw CSV directory
    RAW_CSV_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    csv_copy_list = []  # Track CSVs to copy

    for idx, (shape, memory_config) in enumerate(CONFIGS):
        test_num = idx + 1

        print(f"Test {test_num}: {shape} / {memory_config}")

        duration, source_csv = extract_duration_from_csv(test_num)

        results.append(
            {
                "shape": shape,
                "memory_config_type": memory_config,
                "device_kernel_duration_ns": duration if duration is not None else "N/A",
            }
        )

        # Track source CSV for copying
        if source_csv:
            csv_copy_list.append(
                {
                    "test_num": test_num,
                    "source": source_csv,
                    "memory_config": memory_config,
                    "shape": shape,
                }
            )

    # Write consolidated CSV
    print()
    print("=" * 80)
    print("GENERATING CONSOLIDATED CSV")
    print("=" * 80)

    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["shape", "memory_config_type", "device_kernel_duration_ns"])

        for r in results:
            writer.writerow([r["shape"], r["memory_config_type"], r["device_kernel_duration_ns"]])

    print(f"✓ CSV saved: {OUTPUT_CSV}")

    # Copy raw profiler CSVs with descriptive names
    print()
    print("=" * 80)
    print("COPYING RAW PROFILER CSVs")
    print("=" * 80)

    copy_success = 0
    for item in csv_copy_list:
        dest_name = f"perf_{item['test_num']:02d}_{item['memory_config']}.csv"
        dest_path = RAW_CSV_DIR / dest_name

        try:
            shutil.copy2(item["source"], dest_path)
            print(f"  ✓ Copied: {dest_name}")
            copy_success += 1
        except Exception as e:
            print(f"  ✗ Failed to copy {item['source'].name}: {e}")

    print(f"\n✓ Copied {copy_success}/{len(csv_copy_list)} raw CSV files to {RAW_CSV_DIR}")

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total = len(results)
    with_duration = sum(1 for r in results if r["device_kernel_duration_ns"] != "N/A")
    without_duration = total - with_duration

    print(f"Total tests: {total}")
    print(f"With duration: {with_duration}")
    print(f"Without duration: {without_duration}")

    if with_duration > 0:
        print()
        print("Results:")
        print("-" * 80)
        print(f"{'Shape':<25} {'Memory Config':<20} {'Duration (ns)':<20}")
        print("-" * 80)
        for r in results:
            print(f"{r['shape']:<25} {r['memory_config_type']:<20} {str(r['device_kernel_duration_ns']):<20}")
        print("-" * 80)

    print()
    print("OUTPUT FILES:")
    print(f"  Consolidated CSV: {OUTPUT_CSV}")
    print(f"  Raw profiler CSVs: {RAW_CSV_DIR}/")
    print()


if __name__ == "__main__":
    main()
