#!/usr/bin/env python3
"""
Script to organize all generated Binary Power profiler CSV reports into a single folder
with descriptive filenames matching the test configurations.
"""

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
OUTPUT_DIR = Path("/home/ubuntu/tt-metal/binary_pow_perf_reports/profiler_csvs")


def sanitize_filename(text):
    """Create safe filename from text"""
    import re

    return re.sub(r"[^\w\-.]", "_", str(text)).strip("_")


def copy_profiler_csv(test_num, shape, memory_config):
    """Copy profiler CSV to organized location with descriptive name"""

    test_name = f"binary_pow_test_{test_num}"
    test_dir = PROFILER_REPORTS_DIR / test_name

    if not test_dir.exists():
        print(f"  ✗ Directory not found: {test_dir}")
        return False

    # Find most recent timestamp directory
    timestamp_dirs = sorted([d for d in test_dir.iterdir() if d.is_dir()], reverse=True)
    if not timestamp_dirs:
        print(f"  ✗ No timestamp directories in {test_dir}")
        return False

    latest_timestamp = timestamp_dirs[0]

    # Find CSV file
    csv_files = list(latest_timestamp.glob("ops_perf_results_*.csv"))
    if not csv_files:
        print(f"  ✗ No CSV file in {latest_timestamp}")
        return False

    source_csv = csv_files[0]

    # Create descriptive filename
    shape_safe = sanitize_filename(shape)
    mem_safe = sanitize_filename(memory_config)

    dest_filename = f"profiler_{test_num:02d}_{mem_safe}_shape{shape_safe}.csv"
    dest_path = OUTPUT_DIR / dest_filename

    # Copy file
    try:
        shutil.copy2(source_csv, dest_path)
        print(f"  ✓ Copied: {dest_filename}")
        return True
    except Exception as e:
        print(f"  ✗ Error copying: {e}")
        return False


def main():
    """Main execution"""

    print("=" * 80)
    print("ORGANIZING BINARY POWER PROFILER CSV REPORTS")
    print("=" * 80)
    print(f"Source: {PROFILER_REPORTS_DIR}")
    print(f"Destination: {OUTPUT_DIR}")
    print("=" * 80)
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    success_count = 0
    fail_count = 0

    for idx, (shape, memory_config) in enumerate(CONFIGS):
        test_num = idx + 1

        print(f"[{test_num:2d}/{len(CONFIGS)}] {shape:<20} {memory_config:<20}")

        if copy_profiler_csv(test_num, shape, memory_config):
            success_count += 1
        else:
            fail_count += 1

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files: {len(CONFIGS)}")
    print(f"Successfully copied: {success_count}")
    print(f"Failed: {fail_count}")
    print()
    print(f"All profiler CSV files organized in:")
    print(f"  {OUTPUT_DIR}")
    print()

    if success_count > 0:
        print("File naming format:")
        print("  profiler_NN_<memory_config>_shape<shape>.csv")
        print()
        print("Examples:")
        files = sorted(OUTPUT_DIR.glob("*.csv"))[:3]
        for f in files:
            print(f"  {f.name}")
        if len(files) >= 3:
            print("  ...")

    print("=" * 80)


if __name__ == "__main__":
    main()
