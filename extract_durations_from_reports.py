#!/usr/bin/env python3
"""
Script to extract device kernel durations from existing profiler reports
and regenerate the CSV file.
"""

import csv
from pathlib import Path
from datetime import datetime

# Test configurations (same as main script)
CONFIGS = [
    ("[32, 32]", "L1_interleaved", 0.0),
    ("[32, 32]", "L1_interleaved", 1.0),
    ("[32, 32]", "L1_interleaved", 2.0),
    ("[32, 32]", "L1_interleaved", 3.56),
    ("[32, 32]", "L1_interleaved", 3.0),
    ("[32, 32]", "L1_interleaved", 0.65),
    ("[64, 128]", "L1_interleaved", 0.0),
    ("[64, 128]", "L1_interleaved", 1.0),
    ("[64, 128]", "L1_interleaved", 2.0),
    ("[64, 128]", "L1_interleaved", 3.56),
    ("[64, 128]", "L1_interleaved", 3.0),
    ("[64, 128]", "L1_interleaved", 0.65),
    ("[512, 512]", "L1_block_sharded", 0.0),
    ("[512, 512]", "L1_block_sharded", 1.0),
    ("[512, 512]", "L1_block_sharded", 2.0),
    ("[512, 512]", "L1_block_sharded", 3.56),
    ("[512, 512]", "L1_block_sharded", 3.0),
    ("[512, 512]", "L1_block_sharded", 0.65),
    ("[25600, 128]", "L1_height_sharded", 0.0),
    ("[25600, 128]", "L1_height_sharded", 1.0),
    ("[25600, 128]", "L1_height_sharded", 2.0),
    ("[25600, 128]", "L1_height_sharded", 3.56),
    ("[25600, 128]", "L1_height_sharded", 3.0),
    ("[25600, 128]", "L1_height_sharded", 0.65),
    ("[1, 16, 320, 320]", "height_sharded", 0.0),
    ("[1, 16, 320, 320]", "height_sharded", 1.0),
    ("[1, 16, 320, 320]", "height_sharded", 2.0),
    ("[1, 16, 320, 320]", "height_sharded", 3.56),
    ("[1, 16, 320, 320]", "height_sharded", 3.0),
    ("[1, 16, 320, 320]", "height_sharded", 0.65),
]

PROFILER_REPORTS_DIR = Path("/home/ubuntu/tt-metal/generated/profiler/reports")
OUTPUT_CSV = Path("/home/ubuntu/tt-metal/pow_perf_reports/perf_results.csv")


def extract_duration_from_csv(test_num):
    """Extract DEVICE KERNEL DURATION [ns] from profiler CSV"""

    test_name = f"pow_test_{test_num}"
    test_dir = PROFILER_REPORTS_DIR / test_name

    if not test_dir.exists():
        print(f"  ✗ Directory not found: {test_dir}")
        return None

    # Find most recent timestamp directory
    timestamp_dirs = sorted([d for d in test_dir.iterdir() if d.is_dir()], reverse=True)
    if not timestamp_dirs:
        print(f"  ✗ No timestamp directories in {test_dir}")
        return None

    latest_timestamp = timestamp_dirs[0]

    # Find CSV file
    csv_files = list(latest_timestamp.glob("ops_perf_results_*.csv"))
    if not csv_files:
        print(f"  ✗ No CSV file in {latest_timestamp}")
        return None

    csv_file = csv_files[0]

    try:
        with open(csv_file, "r") as f:
            lines = f.readlines()
            if len(lines) < 2:
                print(f"  ✗ CSV has insufficient data")
                return None

            # Parse header
            header = lines[0].strip().split(",")

            # Find DEVICE KERNEL DURATION [ns] column
            duration_col_idx = None
            for idx, col in enumerate(header):
                if "DEVICE KERNEL DURATION" in col and "[ns]" in col:
                    duration_col_idx = idx
                    break

            if duration_col_idx is None:
                print(f"  ✗ Column 'DEVICE KERNEL DURATION [ns]' not found")
                return None

            # Parse data row
            data_line = lines[1].strip().split(",")
            if duration_col_idx >= len(data_line):
                print(f"  ✗ Duration column index out of range")
                return None

            duration_str = data_line[duration_col_idx].strip()
            if not duration_str:
                print(f"  ✗ Duration value is empty")
                return None

            duration = float(duration_str)
            print(f"  ✓ Duration: {duration} ns (from {csv_file.name})")
            return duration

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None


def main():
    """Extract durations and regenerate CSV"""

    print("=" * 80)
    print("EXTRACTING DEVICE KERNEL DURATIONS FROM PROFILER REPORTS")
    print("=" * 80)
    print(f"Profiler reports: {PROFILER_REPORTS_DIR}")
    print(f"Output CSV: {OUTPUT_CSV}")
    print("=" * 80)
    print()

    results = []

    for idx, (shape, memory_config, exponent) in enumerate(CONFIGS):
        test_num = idx + 1

        print(f"Test {test_num}: {shape} / {memory_config} / exp={exponent}")

        duration = extract_duration_from_csv(test_num)

        results.append(
            {
                "exponent": exponent,
                "memory_config_type": memory_config,
                "device_kernel_duration_ns": duration if duration is not None else "N/A",
            }
        )

    # Write CSV
    print()
    print("=" * 80)
    print("GENERATING CSV")
    print("=" * 80)

    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["exponent", "memory_config_type", "device_kernel_duration_ns"])

        for r in results:
            writer.writerow([r["exponent"], r["memory_config_type"], r["device_kernel_duration_ns"]])

    print(f"✓ CSV saved: {OUTPUT_CSV}")

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
        print("Sample results:")
        print("-" * 80)
        print(f"{'Exp':<8} {'Memory Config':<20} {'Duration (ns)':<20}")
        print("-" * 80)
        for r in results[:10]:
            print(f"{r['exponent']:<8} {r['memory_config_type']:<20} {str(r['device_kernel_duration_ns']):<20}")
        if len(results) > 10:
            print("...")
        print("-" * 80)

    print()
    print(f"✓ CSV file ready: {OUTPUT_CSV}")
    print()


if __name__ == "__main__":
    main()
