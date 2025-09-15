#!/usr/bin/env python3
"""
Final Grid Sample Performance Comparison Script

This script creates the comprehensive side-by-side comparison table
showing split reader ON vs OFF performance as requested by the user.
"""

import pandas as pd
import json
from pathlib import Path
from analyze_grid_sample_perf import GridSamplePerformanceAnalyzer


def create_final_comparison_table():
    analyzer = GridSamplePerformanceAnalyzer()

    # CSV files for split reader OFF (from earlier tests)
    csv_files_off = [
        "./generated/profiler/reports/2025_09_15_12_39_30/ops_perf_results_2025_09_15_12_39_30.csv",
        "./generated/profiler/reports/2025_09_15_12_42_20/ops_perf_results_2025_09_15_12_42_20.csv",
    ]

    # CSV file for split reader ON (most recent test)
    csv_files_on = ["./generated/profiler/reports/2025_09_15_12_49_03/ops_perf_results_2025_09_15_12_49_03.csv"]

    print("Analyzing Split Reader OFF data...")
    split_reader_off_data = analyzer.analyze_split_reader_off_data(csv_files_off)

    print("Analyzing Split Reader ON data...")
    split_reader_on_data = []
    for csv_file in csv_files_on:
        results = analyzer.parse_csv_file(csv_file, split_reader_enabled=True)
        split_reader_on_data.extend(results)

    # Create comprehensive comparison table
    performance_table = analyzer.create_performance_table(split_reader_off_data, split_reader_on_data)

    # Enhance the table formatting for better readability
    enhanced_table = performance_table.copy()

    # Sort by channels first, then by precomputed grid, then by batch output channels
    enhanced_table = enhanced_table.sort_values(["Channels", "Use Precomputed Grid", "Batch Output Channels"])

    # Save the final comparison table
    output_file = Path("./grid_sample_split_reader_comparison_final.csv")
    enhanced_table.to_csv(output_file, index=False)

    print(f"\n" + "=" * 120)
    print("GRID SAMPLE PERFORMANCE COMPARISON - SPLIT READER ON vs OFF")
    print("=" * 120)
    print("Key Metrics: Device Kernel Duration (microseconds)")
    print("Channels: 32, 64, 96, 128, 160, 192, 224, 256")
    print("=" * 120)

    # Display the table with better formatting
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    print(enhanced_table.to_string(index=False))

    # Print analysis summary
    print(f"\n" + "=" * 120)
    print("PERFORMANCE ANALYSIS SUMMARY")
    print("=" * 120)

    # Calculate some key statistics
    valid_improvements = enhanced_table[enhanced_table["Improvement"] != "N/A"]["Improvement"]
    if len(valid_improvements) > 0:
        # Convert improvement percentages to float for analysis
        improvement_values = []
        for imp in valid_improvements:
            if imp != "N/A":
                try:
                    val = float(imp.replace("%", ""))
                    improvement_values.append(val)
                except:
                    pass

        if improvement_values:
            avg_improvement = sum(improvement_values) / len(improvement_values)
            max_improvement = max(improvement_values)
            min_improvement = min(improvement_values)

            print(f"Split Reader Performance Impact:")
            print(f"  Average improvement: {avg_improvement:.1f}%")
            print(f"  Maximum improvement: {max_improvement:.1f}%")
            print(f"  Minimum improvement: {min_improvement:.1f}%")

    # Group by precomputed grid setting
    precomputed_false = enhanced_table[enhanced_table["Use Precomputed Grid"] == False]
    precomputed_true = enhanced_table[enhanced_table["Use Precomputed Grid"] == True]

    print(f"\nTest Configuration Summary:")
    print(f"  Total configurations tested: {len(enhanced_table)}")
    print(f"  Regular grid (precomputed=False): {len(precomputed_false)} configurations")
    print(f"  Precomputed grid (precomputed=True): {len(precomputed_true)} configurations")

    # Channel range analysis
    channels_tested = sorted(enhanced_table["Channels"].unique())
    print(f"  Channels tested: {channels_tested}")

    print(f"\nOutput saved to: {output_file}")

    return enhanced_table


if __name__ == "__main__":
    final_table = create_final_comparison_table()
