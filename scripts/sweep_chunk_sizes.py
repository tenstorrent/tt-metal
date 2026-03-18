#!/usr/bin/env python3
"""
Performance sweep script for q_chunk_size and k_chunk_size optimization.

This script sweeps through valid chunk size combinations for test_mla_sdpa_bh_galaxy
to find the optimal parameters that minimize the maximum device kernel execution time.
"""

import os
import sys
import csv
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

# Configuration
CHUNK_SIZES = [32, 64, 128, 256, 512, 1024]
TEST_FILE = "models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_ring_joint_mla.py"
TEST_NAME = "test_mla_sdpa_bh_galaxy"
BACKUP_FILE = f"{TEST_FILE}.sweep_backup"

# Detect repository root (current working directory should be tt-metal)
REPO_ROOT = os.getcwd()
TRACY_CSV_PATTERN = f"{REPO_ROOT}/generated/profiler/reports/*/ops_perf_results_*.csv"

# Line number and pattern to modify
PARAM_LINE_NUM = 577
PARAM_PATTERN = r'\(1, 128, 1, 128, 128 \* 1024, 576, 576, 128, True, \d+, \d+\)'


class ChunkSizeSweeper:
    """Manages the chunk size parameter sweep."""

    def __init__(self):
        self.results = []
        self.start_time = datetime.now()
        self.timestamp = self.start_time.strftime("%Y_%m_%d_%H_%M_%S")
        self.results_file = f"sweep_results_{self.timestamp}.csv"
        self.progress_file = f"sweep_progress_{self.timestamp}.json"
        self.log_file = f"sweep_log_{self.timestamp}.txt"
        self.original_content = None

    def log(self, message: str, to_console: bool = True):
        """Log message to file and optionally console."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"

        with open(self.log_file, 'a') as f:
            f.write(log_msg + "\n")

        if to_console:
            print(message)

    def backup_test_file(self):
        """Create backup of original test file."""
        self.log(f"Creating backup: {BACKUP_FILE}")
        shutil.copy2(TEST_FILE, BACKUP_FILE)

        # Store original content
        with open(TEST_FILE, 'r') as f:
            self.original_content = f.read()

    def restore_test_file(self):
        """Restore original test file from backup."""
        if os.path.exists(BACKUP_FILE):
            self.log(f"Restoring original test file from backup")
            shutil.copy2(BACKUP_FILE, TEST_FILE)
            os.remove(BACKUP_FILE)

    def modify_test_parameters(self, q_chunk: int, k_chunk: int) -> bool:
        """
        Modify the test parameters in the test file.

        Args:
            q_chunk: q_chunk_size value
            k_chunk: k_chunk_size value

        Returns:
            True if modification successful, False otherwise
        """
        try:
            with open(TEST_FILE, 'r') as f:
                lines = f.readlines()

            # Modify line 615 (0-indexed: 614)
            if len(lines) < PARAM_LINE_NUM:
                self.log(f"Error: File has fewer than {PARAM_LINE_NUM} lines", to_console=True)
                return False

            # Create new parameter tuple
            new_params = f"        (1, 128, 1, 128, 128 * 1024, 576, 576, 128, True, {q_chunk}, {k_chunk}, ttnn.MathFidelity.LoFi),\n"

            # Replace the line
            lines[PARAM_LINE_NUM - 1] = new_params

            # Write back
            with open(TEST_FILE, 'w') as f:
                f.writelines(lines)

            self.log(f"Modified test file: q_chunk_size={q_chunk}, k_chunk_size={k_chunk}", to_console=False)
            return True

        except Exception as e:
            self.log(f"Error modifying test file: {e}", to_console=True)
            return False

    def run_tracy_test(self) -> Tuple[str, Optional[str]]:
        """
        Run the test with tracy profiler.

        Returns:
            Tuple of (status, csv_path or None)
            status can be: 'SUCCESS', 'OOM', 'FAILED'
        """
        # Clean up old profiler reports before running
        reports_dir = Path(f"{REPO_ROOT}/generated/profiler/reports")
        if reports_dir.exists():
            # Get list of existing reports
            existing_reports = list(reports_dir.glob("*/ops_perf_results_*.csv"))
        else:
            existing_reports = []

        # Run the test inside docker container with python_env activated
        cmd = f"docker exec metal-dev bash -c \"cd /tt-metal && source python_env/bin/activate && python -m tracy -r -m pytest {TEST_FILE}::{TEST_NAME} -v\""

        self.log(f"Running: {cmd}", to_console=False)

        try:
            result = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                shell=True,
                executable='/bin/bash',
                timeout=1800  # 30 minute timeout
            )

            # Log output
            with open(self.log_file, 'a') as f:
                f.write("\n--- STDOUT ---\n")
                f.write(result.stdout)
                f.write("\n--- STDERR ---\n")
                f.write(result.stderr)
                f.write("\n--- END OUTPUT ---\n\n")

            # Check for OOM errors REGARDLESS of return code (test framework might catch exceptions)
            output_combined = (result.stdout + result.stderr).lower()

            # Check for various OOM patterns
            oom_patterns = [
                "beyond max l1 size",
                "l1 size",
                "tt_throw",
                "out of memory",
                "oom",
                "statically allocated circular buffers"
            ]

            if any(pattern in output_combined for pattern in oom_patterns):
                self.log("  L1 memory overflow (OOM) detected", to_console=False)
                return 'OOM', None

            if result.returncode != 0:
                self.log(f"  Test failed with return code {result.returncode}", to_console=False)
                return 'FAILED', None

            # Find the new CSV file
            new_reports = []
            if reports_dir.exists():
                new_reports = [
                    csv_file for csv_file in reports_dir.glob("*/ops_perf_results_*.csv")
                    if csv_file not in existing_reports
                ]

            if not new_reports:
                self.log("  Warning: No new CSV file found", to_console=True)
                # Try to find the most recent one anyway
                all_csvs = list(reports_dir.glob("*/ops_perf_results_*.csv")) if reports_dir.exists() else []
                if all_csvs:
                    csv_path = max(all_csvs, key=lambda p: p.stat().st_mtime)
                    self.log(f"  Using most recent CSV: {csv_path}", to_console=False)
                    return 'SUCCESS', str(csv_path)
                return 'FAILED', None

            # Get the most recent new report
            csv_path = max(new_reports, key=lambda p: p.stat().st_mtime)
            self.log(f"  CSV generated: {csv_path}", to_console=False)

            return 'SUCCESS', str(csv_path)

        except subprocess.TimeoutExpired:
            self.log("  Test timed out after 30 minutes", to_console=True)
            return 'TIMEOUT', None
        except Exception as e:
            self.log(f"  Error running test: {e}", to_console=True)
            return 'ERROR', None

    def parse_tracy_csv(self, csv_path: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Parse tracy CSV to extract device kernel durations.

        Args:
            csv_path: Path to the CSV file

        Returns:
            Tuple of (max_duration_ns, min_duration_ns) or (None, None) on error
        """
        try:
            durations = []

            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    # Get the DEVICE KERNEL DURATION column
                    duration_str = row.get('DEVICE KERNEL DURATION [ns]', '').strip()

                    # Skip empty or non-numeric values
                    if not duration_str or duration_str == '':
                        continue

                    try:
                        duration = float(duration_str)
                        durations.append(duration)
                    except ValueError:
                        continue

            if not durations:
                self.log(f"  Warning: No valid durations found in {csv_path}", to_console=True)
                return None, None

            max_duration = max(durations)
            min_duration = min(durations)

            self.log(f"  Found {len(durations)} device durations", to_console=False)

            return max_duration, min_duration

        except Exception as e:
            self.log(f"  Error parsing CSV: {e}", to_console=True)
            return None, None

    def save_results(self):
        """Save results to CSV file."""
        if not self.results:
            self.log("No results to save", to_console=True)
            return

        fieldnames = ['q_chunk_size', 'k_chunk_size', 'max_time_ns', 'min_time_ns', 'status']

        with open(self.results_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)

        self.log(f"\nResults saved to: {self.results_file}")

    def print_optimal_config(self):
        """Find and print the optimal configuration."""
        # Filter successful results
        successful = [r for r in self.results if r['status'] == 'SUCCESS' and r['max_time_ns'] is not None]

        if not successful:
            self.log("\nNo successful runs to analyze", to_console=True)
            return

        # Find optimal (minimum max_time)
        optimal = min(successful, key=lambda r: r['max_time_ns'])

        # Find baseline (32, 32)
        baseline = next((r for r in successful if r['q_chunk_size'] == 32 and r['k_chunk_size'] == 32), None)

        self.log("\n" + "="*60)
        self.log("=== OPTIMAL CONFIGURATION ===")
        self.log("="*60)
        self.log(f"q_chunk_size: {optimal['q_chunk_size']}")
        self.log(f"k_chunk_size: {optimal['k_chunk_size']}")
        self.log(f"MAX time: {optimal['max_time_ns']/1e9:.3f}s")
        self.log(f"MIN time: {optimal['min_time_ns']/1e9:.3f}s")

        if baseline and baseline != optimal:
            speedup = baseline['max_time_ns'] / optimal['max_time_ns']
            self.log(f"Speedup vs baseline (32,32): {speedup:.2f}x")

        self.log("="*60)

        # Print all configurations sorted by performance
        self.log("\n=== ALL CONFIGURATIONS (SORTED BY PERFORMANCE) ===")
        sorted_configs = sorted(successful, key=lambda r: r['max_time_ns'])

        for i, config in enumerate(sorted_configs, 1):
            self.log(f"{i}. q={config['q_chunk_size']}, k={config['k_chunk_size']}: "
                    f"MAX={config['max_time_ns']/1e9:.3f}s, MIN={config['min_time_ns']/1e9:.3f}s")

        self.log("\n")

    def run_sweep(self):
        """Execute the full parameter sweep."""
        self.log("="*60)
        self.log("Starting chunk size sweep...")
        self.log(f"Valid chunk sizes (q and k): {CHUNK_SIZES}")
        self.log(f"Total combinations to test: {len(CHUNK_SIZES) * len(CHUNK_SIZES)}")
        self.log(f"Test file: {TEST_FILE}")
        self.log(f"Test name: {TEST_NAME}")
        self.log("="*60 + "\n")

        try:
            # Backup original file
            self.backup_test_file()

            # Iterate through valid q_chunk sizes
            for q_chunk in CHUNK_SIZES:
                self.log(f"\n--- Testing q_chunk_size = {q_chunk} ---")

                # For each q_chunk, sweep k_chunk from 32 to 2048
                for k_chunk in CHUNK_SIZES:
                    self.log(f"\nTesting q_chunk_size={q_chunk}, k_chunk_size={k_chunk}")

                    # Modify test parameters
                    if not self.modify_test_parameters(q_chunk, k_chunk):
                        self.log("  Failed to modify test file, skipping")
                        continue

                    # Run test with tracy
                    status, csv_path = self.run_tracy_test()

                    # Handle different failure statuses
                    if status in ['OOM', 'FAILED', 'TIMEOUT', 'ERROR']:
                        # Map status to user-friendly message
                        status_msg = {
                            'OOM': 'OOM (L1 memory overflow)',
                            'FAILED': 'FAILED',
                            'TIMEOUT': 'TIMEOUT',
                            'ERROR': 'ERROR'
                        }.get(status, status)

                        self.results.append({
                            'q_chunk_size': q_chunk,
                            'k_chunk_size': k_chunk,
                            'max_time_ns': None,
                            'min_time_ns': None,
                            'status': status_msg
                        })
                        self.log(f"  {status_msg} - stopping sweep for q_chunk_size={q_chunk}, moving to next q_chunk")

                        # Save progress after each test
                        self.save_results()
                        break  # Stop this q_chunk sweep, move to next q_chunk value

                    # Parse results for successful tests
                    max_time, min_time = self.parse_tracy_csv(csv_path)

                    if max_time is None:
                        # Test passed but CSV parsing failed (empty device-only report)
                        self.results.append({
                            'q_chunk_size': q_chunk,
                            'k_chunk_size': k_chunk,
                            'max_time_ns': None,
                            'min_time_ns': None,
                            'status': 'PARSE_ERROR (device-only CSV)'
                        })
                        self.log("  CSV parse error (likely device-only report)")
                        # Don't stop sweep for parse errors - might just be a profiler issue
                    else:
                        self.results.append({
                            'q_chunk_size': q_chunk,
                            'k_chunk_size': k_chunk,
                            'max_time_ns': max_time,
                            'min_time_ns': min_time,
                            'status': 'SUCCESS'
                        })
                        self.log(f"  ✓ MAX: {max_time/1e9:.3f}s, MIN: {min_time/1e9:.3f}s")

                    # Save progress after each test
                    self.save_results()

            # Final save and summary
            self.save_results()
            self.print_optimal_config()

            # Calculate total time
            elapsed = datetime.now() - self.start_time
            self.log(f"Total sweep time: {elapsed}")

        except KeyboardInterrupt:
            self.log("\n\nSweep interrupted by user")
            self.save_results()
            self.print_optimal_config()

        except Exception as e:
            self.log(f"\n\nUnexpected error: {e}")
            import traceback
            self.log(traceback.format_exc())

        finally:
            # Always restore original file
            self.restore_test_file()
            self.log("\nTest file restored to original state")


def main():
    """Main entry point."""
    if not os.path.exists(TEST_FILE):
        print(f"Error: Test file not found: {TEST_FILE}")
        print("Please run this script from /tt-metal directory")
        sys.exit(1)

    sweeper = ChunkSizeSweeper()
    sweeper.run_sweep()


if __name__ == "__main__":
    main()
