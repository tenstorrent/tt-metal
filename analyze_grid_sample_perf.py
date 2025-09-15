#!/usr/bin/env python3
"""
Grid Sample Performance Analysis Script

This script extracts performance data from Tracy CSV files and creates structured tables.
"""

import pandas as pd
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class GridSamplePerformanceAnalyzer:
    def __init__(self):
        self.base_dir = Path(__file__).parent

    def extract_channels_from_attributes(self, attributes_str: str) -> Optional[int]:
        """Extract channel information from attributes string"""
        # Look for input tensor dimensions in the attributes
        # The format is usually INPUT_0_W_PAD[LOGICAL] etc.
        return None

    def extract_channels_from_tensor_info(self, row: pd.Series) -> Optional[int]:
        """Extract channel count from tensor dimension columns"""
        # Look for INPUT_0_X_PAD[LOGICAL] which represents channels
        input_channels = row.get("INPUT_0_X_PAD[LOGICAL]", None)
        if pd.notna(input_channels):
            # Parse the format like "32[32]" to extract 32
            match = re.search(r"(\d+)", str(input_channels))
            if match:
                return int(match.group(1))
        return None

    def extract_precomputed_grid_info(self, attributes_str: str) -> bool:
        """Extract use_precomputed_grid information from attributes"""
        if pd.isna(attributes_str):
            return False
        return "'use_precomputed_grid': 'true'" in str(attributes_str)

    def extract_batch_output_channels_info(self, attributes_str: str) -> Optional[bool]:
        """Extract batch_output_channels information from attributes"""
        if pd.isna(attributes_str):
            return None
        if "'batch_output_channels': 'true'" in str(attributes_str):
            return True
        elif "'batch_output_channels': 'false'" in str(attributes_str):
            return False
        return None

    def parse_csv_file(self, csv_path: str, split_reader_enabled: bool) -> List[Dict]:
        """Parse a single CSV file and extract grid sample performance data"""
        print(f"Parsing CSV file: {csv_path}")

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Error reading CSV {csv_path}: {e}")
            return []

        # Filter for GridSample operations only
        print(f"Available OP TYPEs: {df['OP TYPE'].unique()}")
        grid_sample_ops = df[df["OP CODE"] == "GridSample"].copy()

        if grid_sample_ops.empty:
            print("No GridSample operations found in CSV")
            return []

        results = []

        for _, row in grid_sample_ops.iterrows():
            channels = self.extract_channels_from_tensor_info(row)
            use_precomputed_grid = self.extract_precomputed_grid_info(row["ATTRIBUTES"])
            batch_output_channels = self.extract_batch_output_channels_info(row["ATTRIBUTES"])

            # Determine test type based on batch_output_channels presence
            if batch_output_channels is not None:
                test_type = "sharded_grid_batching"
            else:
                test_type = "sharded_channels"
                batch_output_channels = "N/A"

            device_kernel_duration = row.get("DEVICE KERNEL DURATION [ns]", 0)

            result = {
                "channels": channels,
                "use_precomputed_grid": use_precomputed_grid,
                "test_type": test_type,
                "batch_output_channels": batch_output_channels,
                "split_reader_enabled": split_reader_enabled,
                "device_kernel_duration_ns": device_kernel_duration,
                "device_kernel_duration_us": device_kernel_duration / 1000.0,
                "op_code": row.get("OP CODE", ""),
                "attributes": row.get("ATTRIBUTES", ""),
            }

            results.append(result)

        print(f"Extracted {len(results)} GridSample operations")
        return results

    def analyze_split_reader_off_data(self, csv_files: List[str]) -> List[Dict]:
        """Analyze performance data with split reader OFF"""
        all_results = []

        for csv_file in csv_files:
            results = self.parse_csv_file(csv_file, split_reader_enabled=False)
            all_results.extend(results)

        return all_results

    def create_performance_table(
        self, split_reader_off_data: List[Dict], split_reader_on_data: List[Dict] = None
    ) -> pd.DataFrame:
        """Create a comprehensive performance comparison table"""

        # Create a comprehensive list of all parameter combinations
        all_combinations = set()

        for data in split_reader_off_data:
            if data["channels"] is not None:
                combination = (
                    data["channels"],
                    data["use_precomputed_grid"],
                    data["test_type"],
                    data["batch_output_channels"],
                )
                all_combinations.add(combination)

        if split_reader_on_data:
            for data in split_reader_on_data:
                if data["channels"] is not None:
                    combination = (
                        data["channels"],
                        data["use_precomputed_grid"],
                        data["test_type"],
                        data["batch_output_channels"],
                    )
                    all_combinations.add(combination)

        # Create lookup dictionaries
        off_lookup = {}
        for data in split_reader_off_data:
            if data["channels"] is not None:
                key = (data["channels"], data["use_precomputed_grid"], data["test_type"], data["batch_output_channels"])
                off_lookup[key] = data["device_kernel_duration_us"]

        on_lookup = {}
        if split_reader_on_data:
            for data in split_reader_on_data:
                if data["channels"] is not None:
                    key = (
                        data["channels"],
                        data["use_precomputed_grid"],
                        data["test_type"],
                        data["batch_output_channels"],
                    )
                    on_lookup[key] = data["device_kernel_duration_us"]

        # Build the table
        table_data = []

        for channels, use_precomputed, test_type, batch_output_channels in sorted(all_combinations):
            key = (channels, use_precomputed, test_type, batch_output_channels)

            off_time = off_lookup.get(key, "N/A")
            on_time = on_lookup.get(key, "N/A") if split_reader_on_data else "N/A"

            # Calculate improvement percentage
            improvement = "N/A"
            if off_time != "N/A" and on_time != "N/A" and off_time > 0:
                improvement_pct = ((off_time - on_time) / off_time) * 100
                improvement = f"{improvement_pct:.1f}%"

            row = {
                "Channels": channels,
                "Use Precomputed Grid": use_precomputed,
                "Test Type": test_type,
                "Batch Output Channels": batch_output_channels,
                "Split Reader OFF (μs)": f"{off_time:.1f}" if off_time != "N/A" else "N/A",
                "Split Reader ON (μs)": f"{on_time:.1f}" if on_time != "N/A" else "N/A",
                "Improvement": improvement,
            }

            table_data.append(row)

        return pd.DataFrame(table_data)

    def save_results(self, results_df: pd.DataFrame, filename: str):
        """Save results to CSV and display summary"""
        output_file = self.base_dir / filename
        results_df.to_csv(output_file, index=False)

        print(f"\nResults saved to: {output_file}")
        print("\n" + "=" * 80)
        print("GRID SAMPLE PERFORMANCE COMPARISON - SPLIT READER OFF")
        print("=" * 80)
        print(results_df.to_string(index=False))

        # Print some summary statistics
        print(f"\nSummary:")
        print(f"Total test configurations: {len(results_df)}")

        if len(results_df) > 0 and "Test Type" in results_df.columns:
            # Group by test type
            test_type_counts = results_df["Test Type"].value_counts()
            for test_type, count in test_type_counts.items():
                print(f"  {test_type}: {count} configurations")

        return output_file


if __name__ == "__main__":
    analyzer = GridSamplePerformanceAnalyzer()

    # Find the recent CSV files for split reader OFF
    csv_files = [
        "./generated/profiler/reports/2025_09_15_12_39_30/ops_perf_results_2025_09_15_12_39_30.csv",
        "./generated/profiler/reports/2025_09_15_12_42_20/ops_perf_results_2025_09_15_12_42_20.csv",
    ]

    # Analyze split reader OFF data
    split_reader_off_data = analyzer.analyze_split_reader_off_data(csv_files)

    # Create performance table
    performance_table = analyzer.create_performance_table(split_reader_off_data)

    # Save results
    analyzer.save_results(performance_table, "grid_sample_performance_split_reader_off.csv")

    # Save raw data for later use
    raw_data_file = analyzer.base_dir / "grid_sample_raw_data_split_reader_off.json"
    with open(raw_data_file, "w") as f:
        json.dump(split_reader_off_data, f, indent=2)

    print(f"\nRaw data saved to: {raw_data_file}")
