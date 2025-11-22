#!/usr/bin/env python3
"""
Script to extract specific performance metrics columns from AI benchmark reports.

This script processes CSV files containing performance data and extracts key columns:
- Op code, global call count, core count, NOC UTIL (%), DRAM BW UTIL (%),
- NPE CONG IMPACT (%), PM COMPUTE [ns], PM FPU UTIL (%),
- DEVICE KERNEL DURATION [ns],
- Packet Size Min, Q1, Median, Q3, Max
- SFPU utilization metrics (Min, Median, Max, Avg on full grid)
- FPU utilization metrics (Min, Median, Max, Avg on full grid) 
- MATH utilization metrics (Min, Median, Max, Avg on full grid)

Additionally, it calculates and adds:
1. "% of Total Cycles" column showing each operation's DEVICE KERNEL DURATION as a
   percentage of the total across all operations.
2. Logical size columns for all inputs and output:
   - "INPUT0_LOGICAL_SIZE", "INPUT1_LOGICAL_SIZE", "INPUT2_LOGICAL_SIZE", "OUTPUT0_LOGICAL_SIZE"
   combining padding dimensions into format [W, Z, Y, X].
3. Memory config columns for all inputs and output:
   - "INPUT_0_MEM_CONFIG", "INPUT_1_MEM_CONFIG", "INPUT_2_MEM_CONFIG",
   - "INPUT_3_MEM_CONFIG", "OUTPUT_0_MEM_CONFIG"
   combining layout, datatype, and memory into format LAYOUT-DATATYPE-MEMORY.

The script also supports an optional cleanup feature that removes histogram files 
(packet_size_hist_*.png) that don't correspond to any GLOBAL CALL COUNT values in the data.

Usage:
    python extract_performance_metrics.py <input_csv_file> [cleanup_folder] [output_csv_file]

Examples:
    python extract_performance_metrics.py ops_perf_results_yolov8s_cleaned.csv
    python extract_performance_metrics.py ops_perf_results_yolov8s_cleaned.csv /path/to/histograms
    python extract_performance_metrics.py ops_perf_results_yolov8s_cleaned.csv /path/to/histograms extracted_metrics.csv
    
The cleanup_folder parameter is optional and used to remove histogram files (packet_size_hist_*.png)
that don't correspond to any GLOBAL CALL COUNT values in the processed data.
"""

import pandas as pd
import sys
import os
import re
import glob
from pathlib import Path


def extract_performance_metrics(input_file):
    """
    Extract specific performance metrics from CSV file.

    Args:
        input_file (str): Path to the input CSV file

    Returns:
        pd.DataFrame: DataFrame containing extracted columns
    """

    # Define the columns we want to extract
    target_columns = [
        "OP CODE",
        "GLOBAL CALL COUNT",
        "CORE COUNT",
        "NOC UTIL (%)",
        "DRAM BW UTIL (%)",
        "NPE CONG IMPACT (%)",
        "PM COMPUTE [ns]",
        "PM FPU UTIL (%)",
        "DEVICE KERNEL DURATION [ns]",
        "Packet Size Min",
        "Packet Size Q1",
        "Packet Size Median",
        "Packet Size Q3",
        "Packet Size Max",
        # Additional utilization columns
        "SFPU Util Min (%)",
        "SFPU Util Median (%)",
        "SFPU Util Max (%)",
        "Avg SFPU util on full grid (%)",
        "FPU Util Min (%)",
        "FPU Util Median (%)",
        "FPU Util Max (%)",
        "Avg FPU util on full grid (%)",
        "MATH Util Min (%)",
        "MATH Util Median (%)",
        "MATH Util Max (%)",
        "Avg Math util on full grid (%)",
    ]

    try:
        # Read the CSV file
        print(f"Reading input file: {input_file}")
        df = pd.read_csv(input_file)

        print(f"Original CSV shape: {df.shape}")
        print(f"Available columns: {len(df.columns)}")

        # Check which target columns exist in the DataFrame
        existing_columns = []
        missing_columns = []

        for col in target_columns:
            if col in df.columns:
                existing_columns.append(col)
            else:
                missing_columns.append(col)

        if missing_columns:
            print(f"Warning: The following columns were not found in the CSV:")
            for col in missing_columns:
                print(f"  - {col}")

        if not existing_columns:
            raise ValueError("None of the target columns were found in the input CSV!")

        # Extract the existing columns
        extracted_df = df[existing_columns].copy()

        # Add "% of Total Cycles" column if DEVICE KERNEL DURATION exists
        if "DEVICE KERNEL DURATION [ns]" in existing_columns:
            total_kernel_duration = extracted_df["DEVICE KERNEL DURATION [ns]"].sum()
            if total_kernel_duration > 0:
                extracted_df["% of Total Cycles"] = (extracted_df["DEVICE KERNEL DURATION [ns]"] / total_kernel_duration) * 100
                print(f"Added '% of Total Cycles' column (total KERNEL duration: {total_kernel_duration:,.0f} ns)")
            else:
                print("Warning: Total DEVICE KERNEL DURATION is 0, cannot calculate percentages")
        else:
            print("Warning: DEVICE KERNEL DURATION [ns] not found, cannot calculate % of Total Cycles")

        # Add logical size and memory config columns for INPUT_1, INPUT_2, INPUT_3, and OUTPUT_0
        def create_input_columns(prefix, display_name):
            """Create logical size and memory config columns for a given input/output prefix"""
            # Create logical size column
            pad_columns = [
                f"{prefix}_W_PAD[LOGICAL]",
                f"{prefix}_Z_PAD[LOGICAL]",
                f"{prefix}_Y_PAD[LOGICAL]",
                f"{prefix}_X_PAD[LOGICAL]",
            ]

            existing_pad_cols = [col for col in pad_columns if col in df.columns]
            if len(existing_pad_cols) == 4:

                def create_logical_size(row):
                    """Create [W, Z, Y, X] format from padding dimensions"""

                    def extract_logical_value(value):
                        """Extract logical value from format like '640[640]' -> 640"""
                        if pd.isna(value) or value == "":
                            raise ValueError("")

                        if isinstance(value, str) and "[" in value and "]" in value:
                            logical_part = value.split("[")[1].split("]")[0]
                            return int(logical_part)
                        else:
                            return int(float(value))

                    try:
                        w = extract_logical_value(row[f"{prefix}_W_PAD[LOGICAL]"])
                        z = extract_logical_value(row[f"{prefix}_Z_PAD[LOGICAL]"])
                        y = extract_logical_value(row[f"{prefix}_Y_PAD[LOGICAL]"])
                        x = extract_logical_value(row[f"{prefix}_X_PAD[LOGICAL]"])
                        return f"[{w}, {z}, {y}, {x}]"
                    except:
                        return "N/A"

                logical_col_name = f'{prefix.replace("_", "")}_LOGICAL_SIZE'
                extracted_df[logical_col_name] = df.apply(create_logical_size, axis=1)
                print(f"Added '{logical_col_name}' column combining [W, Z, Y, X] dimensions")
            else:
                print(
                    f"Warning: Not all {prefix} padding columns found ({len(existing_pad_cols)}/4), cannot create {prefix.replace('_', '')}_LOGICAL_SIZE"
                )

            # Create memory config column
            mem_columns = [f"{prefix}_LAYOUT", f"{prefix}_DATATYPE", f"{prefix}_MEMORY"]

            existing_mem_cols = [col for col in mem_columns if col in df.columns]
            if len(existing_mem_cols) == 3:

                def create_mem_config(row):
                    """Create layout-datatype-memory format from memory config columns"""
                    try:
                        if (
                            pd.isna(row[f"{prefix}_LAYOUT"])
                            or pd.isna(row[f"{prefix}_DATATYPE"])
                            or pd.isna(row[f"{prefix}_MEMORY"])
                            or row[f"{prefix}_LAYOUT"] == ""
                            or row[f"{prefix}_DATATYPE"] == ""
                            or row[f"{prefix}_MEMORY"] == ""
                        ):
                            raise ValueError("")
                        layout = str(row[f"{prefix}_LAYOUT"])
                        datatype = str(row[f"{prefix}_DATATYPE"])
                        memory = str(row[f"{prefix}_MEMORY"])
                        return f"{layout}-{datatype}-{memory}"
                    except:
                        return "N/A"

                mem_col_name = f"{prefix}_MEM_CONFIG"
                extracted_df[mem_col_name] = df.apply(create_mem_config, axis=1)
                print(f"Added '{mem_col_name}' column combining LAYOUT-DATATYPE-MEMORY")
            else:
                print(
                    f"Warning: Not all {prefix} memory config columns found ({len(existing_mem_cols)}/3), cannot create {prefix}_MEM_CONFIG"
                )

        # Create columns for each input and output
        create_input_columns("INPUT_0", "INPUT_0")
        create_input_columns("INPUT_1", "INPUT_1")
        create_input_columns("INPUT_2", "INPUT_2")
        create_input_columns("OUTPUT_0", "OUTPUT_0")

        print(f"Extracted {len(existing_columns)} columns:")
        for col in existing_columns:
            print(f"  - {col}")

        # Show all calculated columns
        calculated_cols = [
            "% of Total Cycles",
            "INPUT0_LOGICAL_SIZE",
            "INPUT_0_MEM_CONFIG",
            "INPUT1_LOGICAL_SIZE",
            "INPUT_1_MEM_CONFIG",
            "INPUT2_LOGICAL_SIZE",
            "INPUT_2_MEM_CONFIG",
            "OUTPUT0_LOGICAL_SIZE",
            "OUTPUT_0_MEM_CONFIG",
        ]

        for col in calculated_cols:
            if col in extracted_df.columns:
                print(f"  - {col} (calculated)")

        print(f"Extracted DataFrame shape: {extracted_df.shape}")

        # Apply explicit column ordering
        preferred_column_order = [
            "OP CODE",
            "GLOBAL CALL COUNT", 
            "CORE COUNT",
            "% of Total Cycles",
            "DEVICE KERNEL DURATION [ns]",
            "Avg SFPU util on full grid (%)",
            "Avg FPU util on full grid (%)",
            "Avg Math util on full grid (%)",
            "NOC UTIL (%)",
            "DRAM BW UTIL (%)",
            "NPE CONG IMPACT (%)",
            "PM COMPUTE [ns]",
            "PM FPU UTIL (%)",
            "Packet Size Min",
            "Packet Size Q1", 
            "Packet Size Median",
            "Packet Size Q3",
            "Packet Size Max",
            "SFPU Util Min (%)",
            "SFPU Util Median (%)",
            "SFPU Util Max (%)",
            "FPU Util Min (%)",
            "FPU Util Median (%)",
            "FPU Util Max (%)",
            "MATH Util Min (%)",
            "MATH Util Median (%)",
            "MATH Util Max (%)",
        ]
        
        # Build final column order: preferred columns first, then any remaining columns
        current_cols = list(extracted_df.columns)
        final_column_order = []
        
        # Add preferred columns that exist in the dataframe
        for col in preferred_column_order:
            if col in current_cols:
                final_column_order.append(col)
        
        # Add any remaining columns that weren't in the preferred list
        for col in current_cols:
            if col not in final_column_order:
                final_column_order.append(col)
        
        # Reorder the DataFrame
        extracted_df = extracted_df[final_column_order]
        print("Applied explicit column ordering")

        # Format all percentage columns to 3 decimal places
        percentage_columns = [col for col in extracted_df.columns if '(%)' in col or col == '% of Total Cycles']
        for col in percentage_columns:
            if col in extracted_df.columns and extracted_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                extracted_df[col] = extracted_df[col].round(3)
        
        if percentage_columns:
            print(f"Formatted {len(percentage_columns)} percentage columns to 3 decimal places: {', '.join(percentage_columns)}")

        # Display some statistics
        print("\n=== DATA SUMMARY ===")
        print(f"Total rows: {len(extracted_df)}")

        # Show unique values for categorical columns
        if "OP CODE" in extracted_df.columns:
            unique_ops = extracted_df["OP CODE"].nunique()
            print(f"Unique operation codes: {unique_ops}")
            print("Top 5 most frequent operations:")
            print(extracted_df["OP CODE"].value_counts().head())

        # Show top operations by cycle percentage if available
        if "% of Total Cycles" in extracted_df.columns:
            print(f"\nTop 10 operations by % of Total Cycles:")
            top_ops = extracted_df.nlargest(10, "% of Total Cycles")[
                ["OP CODE", "DEVICE KERNEL DURATION [ns]", "% of Total Cycles"]
            ]
            for _, row in top_ops.iterrows():
                print(
                    f"  {row['OP CODE']:<30} {row['DEVICE KERNEL DURATION [ns]']:>12,.0f} ns ({row['% of Total Cycles']:>6.2f}%)"
                )

        # Show sample logical sizes for all inputs and output
        logical_size_cols = [
            "INPUT0_LOGICAL_SIZE",
            "INPUT1_LOGICAL_SIZE",
            "INPUT2_LOGICAL_SIZE",
            "OUTPUT0_LOGICAL_SIZE",
        ]
        available_logical_cols = [col for col in logical_size_cols if col in extracted_df.columns]

        if available_logical_cols:
            print(f"\nSample Logical Sizes (first 3 unique values for each available):")
            for col in available_logical_cols:
                unique_sizes = extracted_df[col].drop_duplicates().head(3)
                non_na_sizes = [size for size in unique_sizes if size != "[N/A, N/A, N/A, N/A]"][:3]
                if non_na_sizes:
                    print(f"  {col}: {', '.join(non_na_sizes)}")

        # Show sample memory configs for all inputs and output
        mem_config_cols = [
            "INPUT_0_MEM_CONFIG",
            "INPUT_1_MEM_CONFIG",
            "INPUT_2_MEM_CONFIG",
            "INPUT_3_MEM_CONFIG",
            "OUTPUT_0_MEM_CONFIG",
        ]
        available_mem_cols = [col for col in mem_config_cols if col in extracted_df.columns]

        if available_mem_cols:
            print(f"\nSample Memory Configs (first 2 unique values for each available):")
            for col in available_mem_cols:
                unique_configs = extracted_df[col].drop_duplicates().head(2)
                non_na_configs = [config for config in unique_configs if config != "N/A-N/A-N/A"][:2]
                if non_na_configs:
                    print(f"  {col}: {', '.join(non_na_configs)}")

        # Show statistics for numerical columns
        numerical_cols = extracted_df.select_dtypes(include=["number"]).columns
        if len(numerical_cols) > 0:
            print(f"\nNumerical column statistics:")
            print(extracted_df[numerical_cols].describe())

        return extracted_df

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found!")
        return None
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None


def cleanup_histogram_files(cleanup_folder, extracted_df):
    """
    Clean up histogram files that don't correspond to any GLOBAL CALL COUNT values.
    
    Args:
        cleanup_folder (str): Path to the folder containing histogram files
        extracted_df (pd.DataFrame): DataFrame with the extracted data
    """
    if not cleanup_folder or not os.path.exists(cleanup_folder):
        print(f"Cleanup folder '{cleanup_folder}' does not exist or not provided, skipping cleanup")
        return
    
    if "GLOBAL CALL COUNT" not in extracted_df.columns:
        print("GLOBAL CALL COUNT column not found, skipping cleanup")
        return
    
    # Get all unique GLOBAL CALL COUNT values from the data
    valid_call_counts = set(extracted_df["GLOBAL CALL COUNT"].unique())
    
    # Pattern to match histogram files like "packet_size_hist_1024.png"
    histogram_pattern = os.path.join(cleanup_folder, "packet_size_hist_*.png")
    histogram_files = glob.glob(histogram_pattern)
    
    if not histogram_files:
        print(f"No histogram files found matching pattern in '{cleanup_folder}'")
        return
    
    print(f"\nChecking {len(histogram_files)} histogram files for cleanup...")
    print(f"Valid GLOBAL CALL COUNT values: {sorted(valid_call_counts)}")
    
    files_removed = 0
    files_kept = 0
    
    for file_path in histogram_files:
        filename = os.path.basename(file_path)
        
        # Extract the number from filename like "packet_size_hist_1024.png"
        match = re.search(r'packet_size_hist_(\d+)\.png$', filename)
        if match:
            call_count = int(match.group(1))
            
            if call_count not in valid_call_counts:
                try:
                    os.remove(file_path)
                    print(f"  ✗ Removed: {filename} (call count {call_count} not in data)")
                    files_removed += 1
                except Exception as e:
                    print(f"  ❌ Failed to remove {filename}: {e}")
            else:
                print(f"  ✓ Kept: {filename} (call count {call_count} exists in data)")
                files_kept += 1
        else:
            print(f"  ⚠ Skipped: {filename} (doesn't match expected pattern)")
    
    print(f"\nCleanup complete: {files_removed} files removed, {files_kept} files kept")


def main():
    """Main function to handle command line arguments and execute extraction."""

    if len(sys.argv) < 2:
        print("Usage: python precess_report.py <input_csv_file> [cleanup_folder] [output_csv_file]")
        print("\nExamples:")
        print("  python precess_report.py ops_perf_results_yolov8s_cleaned.csv")
        print("  python precess_report.py ops_perf_results_yolov8s_cleaned.csv /path/to/histograms")
        print("  python precess_report.py ops_perf_results_yolov8s_cleaned.csv /path/to/histograms extracted_metrics.csv")
        print("\nThe cleanup_folder parameter is optional and used to remove histogram files")
        print("that don't correspond to any GLOBAL CALL COUNT values in the data.")
        sys.exit(1)

    input_file = sys.argv[1]
    cleanup_folder = sys.argv[2] if len(sys.argv) > 2 else None
    output_file = sys.argv[3] if len(sys.argv) > 3 else None

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist!")
        sys.exit(1)

    # Extract the metrics
    result_df = extract_performance_metrics(input_file)

    if result_df is not None:
        # Generate output filename if not provided
        if output_file is None:
            input_path = Path(input_file)
            output_file = input_path.parent / f"extracted_metrics_{input_path.stem}.csv"

        # Save to CSV
        result_df.to_csv(output_file, index=False)
        print(f"Extracted data saved to: {output_file}")
        
        print("\n✅ Successfully extracted performance metrics!")
        
        # Perform cleanup if cleanup folder is provided
        if cleanup_folder:
            cleanup_histogram_files(cleanup_folder, result_df)
    else:
        print("\n❌ Failed to extract performance metrics!")
        sys.exit(1)


if __name__ == "__main__":
    main()
