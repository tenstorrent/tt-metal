#!/usr/bin/env python3
"""
Grid Sample Performance Testing Script

This script runs performance tests for grid sample with different split reader settings
and generates a comprehensive comparison table.
"""

import subprocess
import os
import sys
import pandas as pd
import re
from pathlib import Path
import tempfile
import json
from typing import Dict, List, Tuple, Optional


class GridSamplePerfTester:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.results = []

    def get_split_reader_status(self) -> bool:
        """Check current split reader setting"""
        program_factory_path = (
            self.base_dir / "ttnn/cpp/ttnn/operations/pool/grid_sample/device/grid_sample_program_factory.cpp"
        )
        with open(program_factory_path) as f:
            content = f.read()
            # Look for the enable_split_reader line
            match = re.search(r"constexpr bool enable_split_reader = (true|false);", content)
            if match:
                return match.group(1) == "true"
        return False

    def set_split_reader(self, enabled: bool):
        """Enable/disable split reader in the code"""
        program_factory_path = (
            self.base_dir / "ttnn/cpp/ttnn/operations/pool/grid_sample/device/grid_sample_program_factory.cpp"
        )
        with open(program_factory_path) as f:
            content = f.read()

        # Replace the split reader setting
        new_value = "true" if enabled else "false"
        new_content = re.sub(
            r"constexpr bool enable_split_reader = (true|false);",
            f"constexpr bool enable_split_reader = {new_value};",
            content,
        )

        with open(program_factory_path, "w") as f:
            f.write(new_content)

        print(f"Split reader {'enabled' if enabled else 'disabled'}")

    def run_test_with_tracy(self, test_function: str) -> Optional[str]:
        """Run a specific test with Tracy profiler and return CSV file path"""
        cmd = [
            "TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=4,3",
            "python",
            "-m",
            "tracy",
            "-r",
            "-m",
            "pytest",
            f"test_grid_sample_sharding_debug.py::{test_function}",
            "-v",
        ]

        # Run the command in shell to handle environment variable
        shell_cmd = " ".join(cmd)
        try:
            result = subprocess.run(
                shell_cmd, shell=True, cwd=self.base_dir, capture_output=True, text=True, timeout=300
            )

            # Look for generated CSV file in the output
            # Tracy typically generates ops_perf_results_*.csv files
            csv_files = list(self.base_dir.glob("ops_perf_results_*.csv"))
            if csv_files:
                return str(csv_files[-1])  # Return the most recent one
            return None

        except subprocess.TimeoutExpired:
            print(f"Test {test_function} timed out")
            return None
        except Exception as e:
            print(f"Error running test {test_function}: {e}")
            return None

    def extract_performance_data(self, csv_file: str) -> List[Dict]:
        """Extract performance data from Tracy CSV file"""
        if not csv_file or not os.path.exists(csv_file):
            return []

        try:
            df = pd.read_csv(csv_file)
            # Look for grid_sample operations
            grid_sample_ops = df[df["op_name"].str.contains("grid_sample", case=False, na=False)]

            results = []
            for _, row in grid_sample_ops.iterrows():
                results.append(
                    {
                        "device_kernel_duration": row.get("device_kernel_duration", 0),
                        "op_name": row.get("op_name", ""),
                        "timestamp": row.get("timestamp", ""),
                    }
                )
            return results
        except Exception as e:
            print(f"Error reading CSV {csv_file}: {e}")
            return []

    def run_test_suite(self, split_reader_enabled: bool) -> Dict[str, List[Dict]]:
        """Run all test combinations for a specific split reader setting"""
        print(f"\n=== Running tests with split reader {'ON' if split_reader_enabled else 'OFF'} ===")

        # Test functions to run
        test_functions = ["test_grid_sample_sharded_channels", "test_grid_sample_sharded_grid_batching"]

        results = {}

        for test_func in test_functions:
            print(f"Running {test_func}...")
            csv_file = self.run_test_with_tracy(test_func)
            perf_data = self.extract_performance_data(csv_file)
            results[test_func] = perf_data

            # Clean up CSV file
            if csv_file and os.path.exists(csv_file):
                os.remove(csv_file)

        return results

    def parse_test_parameters(self, test_output: str) -> List[Dict]:
        """Parse test parameters from pytest output"""
        # This would parse the specific parameters used in each test run
        # For now, return the known parameter combinations

        # From test_grid_sample_sharded_channels
        channels_tests = []
        for channels in [32, 64, 96, 128, 160, 192, 224, 256]:
            for use_precomputed in [False, True]:
                channels_tests.append(
                    {
                        "test_type": "sharded_channels",
                        "channels": channels,
                        "use_precomputed_grid": use_precomputed,
                        "batch_output_channels": None,
                        "grid_batching_factor": None,
                    }
                )

        # From test_grid_sample_sharded_grid_batching
        batching_tests = []
        for channels in [32, 64, 96, 128, 160, 192, 224, 256]:
            for use_precomputed in [True, False]:
                for batch_output_channels in [True, False]:
                    batching_tests.append(
                        {
                            "test_type": "sharded_grid_batching",
                            "channels": channels,
                            "use_precomputed_grid": use_precomputed,
                            "batch_output_channels": batch_output_channels,
                            "grid_batching_factor": 8,
                        }
                    )

        return channels_tests + batching_tests

    def create_comparison_table(self, results_off: Dict, results_on: Dict) -> pd.DataFrame:
        """Create a comprehensive comparison table"""

        # Get all parameter combinations
        param_combinations = self.parse_test_parameters("")

        table_data = []

        for params in param_combinations:
            row = {
                "Channels": params["channels"],
                "Use Precomputed Grid": params["use_precomputed_grid"],
                "Test Type": params["test_type"],
                "Batch Output Channels": params.get("batch_output_channels", "N/A"),
                "Grid Batching Factor": params.get("grid_batching_factor", "N/A"),
                "Split Reader OFF (μs)": "N/A",
                "Split Reader ON (μs)": "N/A",
                "Improvement (%)": "N/A",
            }

            # This is a simplified approach - in reality, you'd need to match
            # the performance data to the specific parameter combinations
            # based on the test execution order or additional metadata

            table_data.append(row)

        df = pd.DataFrame(table_data)
        return df

    def run_full_comparison(self):
        """Run the complete performance comparison"""
        print("Starting Grid Sample Performance Comparison")
        print("=" * 50)

        # Step 1: Test with split reader OFF
        self.set_split_reader(False)
        results_off = self.run_test_suite(False)

        # Step 2: Test with split reader ON
        self.set_split_reader(True)
        results_on = self.run_test_suite(True)

        # Step 3: Create comparison table
        comparison_df = self.create_comparison_table(results_off, results_on)

        # Step 4: Save results
        output_file = self.base_dir / "grid_sample_performance_comparison.csv"
        comparison_df.to_csv(output_file, index=False)

        print(f"\nComparison table saved to: {output_file}")
        print("\nPreview of results:")
        print(comparison_df.to_string(index=False))

        return comparison_df


if __name__ == "__main__":
    tester = GridSamplePerfTester()
    tester.run_full_comparison()
