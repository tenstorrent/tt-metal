#!/usr/bin/env python3
"""
Ultimate solution: Creates individual test files for each configuration
and runs them separately with the perf tool. This ensures each test is
completely isolated.
"""

import subprocess
import os
import csv
import re
import time
import shutil
from pathlib import Path
from datetime import datetime

# Test configurations
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

# Shape to torch.Size mapping
SHAPE_MAPPING = {
    "[32, 32]": "torch.Size([32, 32])",
    "[64, 128]": "torch.Size([64, 128])",
    "[512, 512]": "torch.Size([512, 512])",
    "[25600, 128]": "torch.Size([25600, 128])",
    "[1, 16, 320, 320]": "torch.Size([1, 16, 320, 320])",
}

# Paths
ORIGINAL_TEST = "/home/ubuntu/tt-metal/tests/ttnn/unit_tests/operations/eltwise/test_unary_pow.py"
TEMP_TEST_DIR = Path("/home/ubuntu/tt-metal/tests/temp_pow_tests")
PERF_TOOL = "./tools/tracy/profile_this.py"
OUTPUT_DIR = Path("/home/ubuntu/tt-metal/pow_perf_reports")
CSV_OUTPUT = OUTPUT_DIR / "perf_results.csv"
LOG_FILE = OUTPUT_DIR / "execution.log"
RAW_CSV_DIR = OUTPUT_DIR / "profiler_csvs"
PROFILER_REPORTS_DIR = Path("/home/ubuntu/tt-metal/generated/profiler/reports")

# Create directories
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_TEST_DIR.mkdir(exist_ok=True)


def log(msg):
    """Log message"""
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")


def create_single_test_file(shape, memory_config, exponent, test_num):
    """Create a test file for a single configuration"""

    torch_shape = SHAPE_MAPPING[shape]

    test_content = f'''import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

def get_memory_config(memory_config_type, shape, device):
    """Get memory configuration based on type and shape"""
    if memory_config_type == "L1_interleaved":
        return ttnn.L1_MEMORY_CONFIG
    elif memory_config_type == "L1_block_sharded":
        return ttnn.create_sharded_memory_config(
            shape,
            ttnn.CoreGrid(x=8, y=8),
            ttnn.ShardStrategy.BLOCK,
        )
    elif memory_config_type == "L1_height_sharded":
        core_grid = ttnn.CoreRangeSet(
            {{
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5)),
                ttnn.CoreRange(ttnn.CoreCoord(0, 6), ttnn.CoreCoord(1, 6)),
            }}
        )
        return ttnn.create_sharded_memory_config(
            (512, 128),
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
    elif memory_config_type == "height_sharded":
        core_grid = ttnn.CoreRangeSet({{ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 4))}})
        return ttnn.create_sharded_memory_config(
            (128, 320),
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
    else:
        return ttnn.DRAM_MEMORY_CONFIG

def test_pow_single(device):
    """Test single pow configuration"""
    torch.manual_seed(0)

    # Configuration
    input_shape = {torch_shape}
    memory_config_type = "{memory_config}"
    exponent = {exponent}

    # Create input
    in_data = torch.empty(input_shape, dtype=torch.bfloat16).uniform_(-10, 10)

    # Get memory config
    memory_config = get_memory_config(memory_config_type, list(input_shape), device)

    # Create input tensor
    input_tensor = ttnn.from_torch(
        in_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=memory_config
    )

    # Run operation
    output_tensor = ttnn.pow(input_tensor, exponent)

    # Verify correctness
    golden_function = ttnn.get_golden_function(ttnn.pow)
    golden_tensor = golden_function(in_data, exponent, device=device)
    assert_with_pcc(ttnn.to_torch(output_tensor), golden_tensor, pcc=0.95)

    print(f"Test passed: shape={{input_shape}}, memory={{memory_config_type}}, exp={{exponent}}")
'''

    test_file = TEMP_TEST_DIR / f"test_pow_{test_num:02d}.py"
    with open(test_file, "w") as f:
        f.write(test_content)

    return test_file


def run_perf_test(test_file, test_num, output_file):
    """Run perf test on a single test file"""

    cmd = [PERF_TOOL, "-n", f"pow_test_{test_num}", "-c", f"pytest {test_file} -v"]

    log(f"  Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, cwd="/home/ubuntu/tt-metal", capture_output=True, text=True, timeout=600)

        # Save output
        with open(output_file, "w") as f:
            f.write(f"Test {test_num}\n")
            f.write(f"Test file: {test_file}\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write("=" * 80 + "\n\n")
            f.write("STDOUT:\n")
            f.write(result.stdout)
            f.write("\n\nSTDERR:\n")
            f.write(result.stderr)
            f.write(f"\n\nReturn code: {result.returncode}\n")

        return result.returncode == 0, result.stdout

    except subprocess.TimeoutExpired:
        log(f"  ✗ Timeout")
        return False, ""
    except Exception as e:
        log(f"  ✗ Error: {e}")
        return False, ""


def extract_duration(output_file, stdout_content, test_num):
    """Extract and SUM all device kernel durations from generated profiler CSV

    Returns: (total_duration, source_csv_path) or (None, None)
    """

    test_name = f"pow_test_{test_num}"

    try:
        # Find the test directory
        test_dir = PROFILER_REPORTS_DIR / test_name
        if not test_dir.exists():
            return None, None

        # Find the most recent timestamp directory
        timestamp_dirs = sorted([d for d in test_dir.iterdir() if d.is_dir()], reverse=True)
        if not timestamp_dirs:
            return None, None

        latest_timestamp = timestamp_dirs[0]

        # Find the CSV file
        csv_files = list(latest_timestamp.glob("ops_perf_results_*.csv"))
        if not csv_files:
            return None, None

        csv_file = csv_files[0]

        # Read the CSV using proper CSV parsing (handles quoted fields with commas)
        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)

            if len(rows) < 2:
                return None, None

            # Parse header
            header = rows[0]

            # Find the column index for "DEVICE KERNEL DURATION [ns]"
            duration_col_idx = None
            for idx, col in enumerate(header):
                if "DEVICE KERNEL DURATION" in col and "[ns]" in col:
                    duration_col_idx = idx
                    break

            if duration_col_idx is None:
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
                return None, None

            log(f"  Extracted duration: {total_duration} ns (sum of {row_count} ops)")
            return total_duration, csv_file

    except Exception as e:
        log(f"  Error extracting duration: {e}")
        return None, None


def cleanup():
    """Clean up temporary files"""
    if TEMP_TEST_DIR.exists():
        shutil.rmtree(TEMP_TEST_DIR)
    TEMP_TEST_DIR.mkdir(exist_ok=True)


def main():
    """Main execution"""

    # Initialize
    with open(LOG_FILE, "w") as f:
        f.write(f"Started: {datetime.now()}\n\n")

    # Create directories
    RAW_CSV_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("POW UNARY PERFORMANCE TEST - INDIVIDUAL EXECUTION")
    print("=" * 80)
    print(f"Total tests: {len(CONFIGS)}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 80 + "\n")

    results = []
    csv_copy_list = []  # Track CSVs to copy

    for idx, (shape, memory_config, exponent) in enumerate(CONFIGS):
        test_num = idx + 1

        print(f"\n[Test {test_num}/{len(CONFIGS)}]")
        print(f"  Shape: {shape}")
        print(f"  Memory: {memory_config}")
        print(f"  Exponent: {exponent}")

        log(f"Test {test_num}: shape={shape}, memory={memory_config}, exp={exponent}")

        # Create test file
        test_file = create_single_test_file(shape, memory_config, exponent, test_num)
        log(f"  Created: {test_file}")

        # Output file for perf report
        output_file = OUTPUT_DIR / f"perf_{test_num:02d}_{memory_config}_exp{exponent}.txt"

        # Run test
        success, stdout = run_perf_test(test_file, test_num, output_file)

        if success:
            print(f"  ✓ Test passed")
            log(f"  ✓ Success")
        else:
            print(f"  ✗ Test failed")
            log(f"  ✗ Failed")

        # Extract duration
        print(f"  Extracting duration...")
        duration, source_csv = extract_duration(output_file, stdout, test_num)
        if duration:
            print(f"  ✓ Duration: {duration} ns")
            log(f"  Duration: {duration} ns")
        else:
            print(f"  ⚠ Duration: N/A")
            log(f"  Duration: N/A")

        results.append(
            {
                "exponent": exponent,
                "memory_config_type": memory_config,
                "device_kernel_duration_ns": duration if duration else "N/A",
                "shape": shape,
                "success": success,
            }
        )

        # Track source CSV for copying
        if source_csv:
            csv_copy_list.append(
                {
                    "test_num": test_num,
                    "source": source_csv,
                    "memory_config": memory_config,
                    "exponent": exponent,
                }
            )

        time.sleep(1)

    # Generate consolidated CSV
    print(f"\n{'=' * 80}")
    print("GENERATING CONSOLIDATED CSV")
    print("=" * 80)

    with open(CSV_OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["exponent", "memory_config_type", "device_kernel_duration_ns"])
        for r in results:
            writer.writerow([r["exponent"], r["memory_config_type"], r["device_kernel_duration_ns"]])

    print(f"✓ CSV: {CSV_OUTPUT}")

    # Copy raw profiler CSVs with descriptive names
    print(f"\n{'=' * 80}")
    print("COPYING RAW PROFILER CSVs")
    print("=" * 80)

    copy_success = 0
    for item in csv_copy_list:
        dest_name = f"perf_{item['test_num']:02d}_{item['memory_config']}_exp{item['exponent']}.csv"
        dest_path = RAW_CSV_DIR / dest_name
        try:
            shutil.copy2(item["source"], dest_path)
            print(f"  ✓ Copied: {dest_name}")
            copy_success += 1
        except Exception as e:
            print(f"  ✗ Failed to copy: {e}")

    print(f"\n✓ Copied {copy_success}/{len(csv_copy_list)} raw CSV files")

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total: {len(results)}")
    print(f"Passed: {sum(1 for r in results if r['success'])}")
    print(f"Failed: {sum(1 for r in results if not r['success'])}")
    print(f"With duration: {sum(1 for r in results if r['device_kernel_duration_ns'] != 'N/A')}")

    # Cleanup temp files
    print(f"\nCleaning up temporary test files...")
    cleanup()
    print(f"✓ Done")

    # Print output folder paths
    print(f"\n{'=' * 80}")
    print("OUTPUT FILES")
    print("=" * 80)
    print(f"  Consolidated CSV:  {CSV_OUTPUT}")
    print(f"  Raw Profiler CSVs: {RAW_CSV_DIR}/")
    print(f"  Execution Logs:    {OUTPUT_DIR}/perf_*.txt")
    print(f"  Log File:          {LOG_FILE}")
    print("=" * 80)

    print(f"\n{'=' * 80}")
    print("COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
