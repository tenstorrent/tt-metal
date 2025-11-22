#!/usr/bin/env python3
"""
AI Ubench Report Generator

This script automates the process of generating comprehensive AI benchmark reports
by running profiling commands twice and merging the results.

Usage: gen_ai_ubench_report -o <output_dir> -c <command>

The script:
1. Runs profile_this.py twice:
   - Once with NOC traces (--collect-noc-traces) -> perf_report_with_npe_metrics/
   - Once with perf counters (--profiler-capture-perf-counters=fpu) -> perf_report/
2. Copies CSV files from subdirectories to root output directory
3. Processes the data:
   - Cleans up the regular perf report data
   - Extracts NPE data from the NOC traces report
4. Merges the dataframes using GLOBAL CALL COUNT
5. Extracts final performance metrics
"""

import argparse
import os
import sys
import subprocess
import glob
import shutil
import pandas as pd
from pathlib import Path

# Import our processing functions
from cleanup_report import process_cleanup_data
from extract_npe_data import process_npe_data
from extract_performance_metrics import extract_performance_metrics


def run_profile_command(command, output_dir, subdir, profile_options):
    """
    Run profile_this.py with specified options.
    
    Args:
        command (str): The test command to profile
        output_dir (str): Base output directory
        subdir (str): Subdirectory name for this profiling run
        profile_options (list): Additional options for profile_this.py
    
    Returns:
        str: Path to the subdirectory where results were saved
    """
    full_output_path = os.path.join(output_dir, subdir)
    
    # Build the profile_this.py command
    profile_cmd = [
        sys.executable,
        "tools/tracy/profile_this.py",
        "-o", full_output_path,
        "-c", command
    ] + profile_options
    
    print(f"Running: {' '.join(profile_cmd)}")
    print(f"Output will be saved to: {full_output_path}")
    
    try:
        result = subprocess.run(profile_cmd, check=True, capture_output=True, text=True)
        print(f"Profile command completed successfully")
        if result.stdout:
            print("STDOUT:", result.stdout)
        return full_output_path
    except subprocess.CalledProcessError as e:
        print(f"Error running profile command: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        raise


def find_csv_files(directory):
    """
    Find ops_perf_results_*.csv files in subdirectories with timestamp format.
    Looks for pattern: subdir/<yyyy_mm_dd_hh_mm_ss>/ops_perf_results_*.csv
    Returns the latest file based on timestamp in directory name.
    
    Args:
        directory (str): Directory to search in
        
    Returns:
        list: List with the latest CSV file path, or empty list if none found
    """
    import re
    from datetime import datetime
    
    # Find all CSV files in timestamp-named subdirectories
    pattern = os.path.join(directory, "**/ops_perf_results_*.csv")
    all_csv_files = glob.glob(pattern, recursive=True)
    
    # Filter files that match the timestamp directory pattern
    timestamp_files = []
    timestamp_pattern = re.compile(r'(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})')
    
    for csv_file in all_csv_files:
        # Extract the directory path and look for timestamp pattern
        dir_path = os.path.dirname(csv_file)
        dir_name = os.path.basename(dir_path)
        
        match = timestamp_pattern.search(dir_name)
        if match:
            timestamp_str = match.group(1)
            try:
                # Parse timestamp to datetime for sorting
                timestamp = datetime.strptime(timestamp_str, '%Y_%m_%d_%H_%M_%S')
                timestamp_files.append((timestamp, csv_file))
            except ValueError:
                # Skip files with invalid timestamp format
                continue
    
    if not timestamp_files:
        # Fallback to old behavior if no timestamp directories found
        return all_csv_files
    
    # Sort by timestamp (latest first) and return the latest file
    timestamp_files.sort(key=lambda x: x[0], reverse=True)
    latest_file = timestamp_files[0][1]
    
    print(f"Found {len(timestamp_files)} timestamped CSV files, using latest: {os.path.basename(latest_file)}")
    
    return [latest_file]


def copy_csv_to_root(csv_file, output_dir, suffix=""):
    """
    Copy CSV file to the root output directory with optional suffix.
    
    Args:
        csv_file (str): Source CSV file path
        output_dir (str): Destination directory
        suffix (str): Optional suffix to add to filename
        
    Returns:
        str: Path to the copied file
    """
    csv_path = Path(csv_file)
    if suffix:
        new_name = f"{csv_path.stem}_{suffix}{csv_path.suffix}"
    else:
        new_name = csv_path.name
    
    dest_path = os.path.join(output_dir, new_name)
    shutil.copy2(csv_file, dest_path)
    print(f"Copied {csv_file} -> {dest_path}")
    return dest_path


def merge_dataframes(cleaned_df, npe_df):
    """
    Merge cleaned dataframe with NPE dataframe using GLOBAL CALL COUNT.
    
    Enforces perfect 1:1 matching - will error out if merge is not perfect.
    Takes only NOC UTIL (%), DRAM BW UTIL (%), NPE CONG IMPACT (%) columns
    from NPE data and the rest from cleaned data.
    
    Args:
        cleaned_df (pd.DataFrame): Cleaned performance data
        npe_df (pd.DataFrame): NPE data
        
    Returns:
        pd.DataFrame: Merged dataframe
        
    Raises:
        ValueError: If merge is not perfect 1:1 match
    """
    print(f"Merging dataframes with strict 1:1 validation:")
    print(f"  - Cleaned data shape: {cleaned_df.shape}")
    print(f"  - NPE data shape: {npe_df.shape}")
    
    # Define the columns we want from NPE data
    npe_columns_to_merge = [
        'GLOBAL CALL COUNT',  # Join key
        'NOC UTIL (%)',
        'DRAM BW UTIL (%)', 
        'NPE CONG IMPACT (%)'
    ]
    
    # Check which NPE columns actually exist
    existing_npe_columns = [col for col in npe_columns_to_merge if col in npe_df.columns]
    missing_npe_columns = [col for col in npe_columns_to_merge if col not in npe_df.columns]
    
    if missing_npe_columns:
        print(f"Warning: Missing NPE columns: {missing_npe_columns}")
    
    if 'GLOBAL CALL COUNT' not in existing_npe_columns:
        raise ValueError("Cannot merge: GLOBAL CALL COUNT not found in NPE data")
    
    if 'GLOBAL CALL COUNT' not in cleaned_df.columns:
        raise ValueError("Cannot merge: GLOBAL CALL COUNT not found in cleaned data")
    
    # STRICT VALIDATION: Ensure perfect 1:1 matching
    print("Performing 1:1 merge validation...")
    
    # Get unique GLOBAL CALL COUNT values from both dataframes
    cleaned_call_counts = set(cleaned_df['GLOBAL CALL COUNT'].unique())
    npe_call_counts = set(npe_df['GLOBAL CALL COUNT'].unique())
    
    print(f"  - Unique GLOBAL CALL COUNT values in cleaned data: {len(cleaned_call_counts)}")
    print(f"  - Unique GLOBAL CALL COUNT values in NPE data: {len(npe_call_counts)}")
    
    # Check for duplicates
    if cleaned_df['GLOBAL CALL COUNT'].duplicated().any():
        raise ValueError("Perfect 1:1 merge failed: Duplicate GLOBAL CALL COUNT values in cleaned data")
    
    if npe_df['GLOBAL CALL COUNT'].duplicated().any():
        raise ValueError("Perfect 1:1 merge failed: Duplicate GLOBAL CALL COUNT values in NPE data")
    
    # Check that every GLOBAL CALL COUNT in one is in the other
    if cleaned_call_counts != npe_call_counts:
        missing_in_npe = cleaned_call_counts - npe_call_counts
        missing_in_cleaned = npe_call_counts - cleaned_call_counts
        
        error_msg = "Perfect 1:1 merge failed: GLOBAL CALL COUNT mismatch"
        if missing_in_npe:
            error_msg += f"\n  Missing in NPE data: {len(missing_in_npe)} values"
        if missing_in_cleaned:
            error_msg += f"\n  Missing in cleaned data: {len(missing_in_cleaned)} values"
        raise ValueError(error_msg)
    
    print("✓ 1:1 merge validation passed")
    
    # Extract NPE data with only the columns we need
    npe_subset = npe_df[existing_npe_columns].copy()
    
    # Start with cleaned dataframe and replace NPE columns with NPE data
    merged_df = cleaned_df.copy()
    
    # Replace existing columns with NPE data where available
    # First, create a mapping from GLOBAL CALL COUNT to NPE values
    npe_data_columns = [col for col in existing_npe_columns if col != 'GLOBAL CALL COUNT']
    npe_dict = {}
    for col in npe_data_columns:
        npe_dict[col] = dict(zip(npe_subset['GLOBAL CALL COUNT'], npe_subset[col]))
    
    # Update cleaned dataframe with NPE values
    for col in npe_data_columns:
        if col in merged_df.columns:
            # Replace existing column values with NPE data
            merged_df[col] = merged_df['GLOBAL CALL COUNT'].map(npe_dict[col]).fillna(merged_df[col])
        else:
            # Add new column if it doesn't exist
            merged_df[col] = merged_df['GLOBAL CALL COUNT'].map(npe_dict[col])
    
    print(f"Merged dataframe shape: {merged_df.shape}")
    
    # Final validation: Ensure merge preserved all rows
    if len(merged_df) != len(cleaned_df):
        raise ValueError(f"Merge operation lost rows: expected {len(cleaned_df)}, got {len(merged_df)}")
    
    print("✓ Perfect 1:1 merge completed successfully")
    print(f"  - Total rows: {len(merged_df)}")
    print(f"  - NPE columns added: {len([col for col in existing_npe_columns if col != 'GLOBAL CALL COUNT'])}")
    
    return merged_df


def main():
    """Main function to orchestrate the AI benchmark report generation."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive AI benchmark reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script automates the generation of AI benchmark reports by:
1. Running profiling commands twice (with and without NPE metrics)
2. Processing and cleaning the resulting data
3. Merging the datasets to create a comprehensive report
4. Extracting final performance metrics

Examples:
  python gen_ai_ubench_report.py -o ./results -c "python my_benchmark.py"
  python gen_ai_ubench_report.py -o /path/to/output -c "./run_model.sh --model=vit"
        """
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        required=True,
        help='Output directory for all generated files and reports'
    )
    
    parser.add_argument(
        '-c', '--command',
        required=True,
        help='Command to profile (will be run twice with different profiling options)'
    )
    
    parser.add_argument(
        '--skip-profiling',
        action='store_true',
        help='Skip profiling steps and only process existing CSV files in output directory'
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    output_dir = os.path.abspath(args.output_dir)
    
    print(f"=== AI Ubench Report Generator ===")
    print(f"Command: {args.command}")
    print(f"Output directory: {output_dir}")
    print()
    
    try:
        if not args.skip_profiling:
            # Step 1: Run profile_this.py twice with different configurations
            print("Step 1: Running profiling commands...")
            
            # Run with NPE metrics (NOC traces)
            print("\n--- Running with NPE metrics (NOC traces) ---")
            npe_dir = run_profile_command(
                args.command,
                output_dir,
                "perf_report_with_npe_metrics",
                ["--collect-noc-traces"]
            )
            
            # Run with performance counters
            print("\n--- Running with performance counters ---")
            perf_dir = run_profile_command(
                args.command,
                output_dir,
                "perf_report",
                ["--profiler-capture-perf-counters=fpu"]
            )
            
            print("\nStep 1 completed: Both profiling runs finished successfully")
        else:
            print("Step 1: Skipping profiling (using existing files)")
            npe_dir = os.path.join(output_dir, "perf_report_with_npe_metrics")
            perf_dir = os.path.join(output_dir, "perf_report")
        
        # Step 2: Find and copy CSV files to root directory
        print(f"\nStep 2: Finding and copying CSV files...")
        
        # Find CSV files in both directories
        npe_csv_files = find_csv_files(npe_dir)
        perf_csv_files = find_csv_files(perf_dir)
        
        print(f"Found {len(npe_csv_files)} NPE CSV files")
        print(f"Found {len(perf_csv_files)} perf CSV files")
        
        if not npe_csv_files:
            raise ValueError(f"No NPE CSV files found in {npe_dir}")
        if not perf_csv_files:
            raise ValueError(f"No perf CSV files found in {perf_dir}")
        
        # Use the first CSV file from each directory (assuming single benchmark run)
        npe_csv = npe_csv_files[0]
        perf_csv = perf_csv_files[0]
        
        # Copy files to root directory
        npe_csv_root = copy_csv_to_root(npe_csv, output_dir, "npe_raw")
        perf_csv_root = copy_csv_to_root(perf_csv, output_dir, "perf_raw")
        
        # Step 3: Process the data
        print(f"\nStep 3: Processing data...")
        
        # Load the CSV files
        print("Loading NPE CSV data...")
        npe_df = pd.read_csv(npe_csv_root)
        print(f"NPE data shape: {npe_df.shape}")
        
        print("Loading perf CSV data...")
        perf_df = pd.read_csv(perf_csv_root)
        print(f"Perf data shape: {perf_df.shape}")
        
        # Process cleanup data on perf report
        print("\n--- Processing cleanup data (perf report) ---")
        cleaned_df, cleanup_stats = process_cleanup_data(perf_df)
        
        if cleaned_df is None:
            raise ValueError("Failed to process cleanup data")
        
        print(f"Cleanup successful: {cleanup_stats['final_rows']} rows retained")
        
        # Process NPE data on NPE report
        print("\n--- Processing NPE data (NPE report) ---")
        npe_processed_df, npe_stats = process_npe_data(npe_df)
        
        if npe_processed_df is None:
            raise ValueError("Failed to process NPE data")
        
        print(f"NPE processing successful: {npe_stats['npe_rows']} rows extracted")
        
        # Step 4: Merge the dataframes
        print(f"\nStep 4: Merging dataframes...")
        merged_df = merge_dataframes(cleaned_df, npe_processed_df)
        
        # Save intermediate merged result
        merged_csv_path = os.path.join(output_dir, "merged_data.csv")
        merged_df.to_csv(merged_csv_path, index=False)
        print(f"Saved merged data to: {merged_csv_path}")
        
        # Step 5: Extract performance metrics
        print(f"\nStep 5: Extracting performance metrics...")
        
        # Since extract_performance_metrics expects a file path, save merged data temporarily
        final_df = extract_performance_metrics(merged_csv_path)
        
        if final_df is None:
            raise ValueError("Failed to extract performance metrics")
        
        # Save the final processed report
        final_report_path = os.path.join(output_dir, "ai_ubench_report.csv")
        final_df.to_csv(final_report_path, index=False)
        
        print(f"\n=== Report Generation Complete ===")
        print(f"Final report saved to: {final_report_path}")
        print(f"Final report shape: {final_df.shape}")
        print(f"\nGenerated files:")
        print(f"  - {npe_csv_root} (raw NPE data)")
        print(f"  - {perf_csv_root} (raw perf data)")
        print(f"  - {merged_csv_path} (merged intermediate data)")
        print(f"  - {final_report_path} (final AI ubench report)")
        
        # Summary statistics
        print(f"\nSummary Statistics:")
        print(f"  - Original perf data rows: {len(perf_df)}")
        print(f"  - Cleaned perf data rows: {len(cleaned_df)}")
        print(f"  - NPE data rows: {len(npe_processed_df)}")
        print(f"  - Final merged rows: {len(merged_df)}")
        print(f"  - Final report rows: {len(final_df)}")
        
        return final_report_path
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
