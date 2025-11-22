#!/usr/bin/env python3
"""
Extract NPE data script for performance counter reports.

This script provides both a command-line interface and reusable functions for extracting
NPE data from performance counter reports:

Core function:
- process_npe_data(df): Takes a DataFrame and returns NPE data with statistics

Command-line functionality:
1. Finds the highest METAL TRACE REPLAY SESSION ID
2. Gets rows with this highest session ID value
3. Extracts the GLOBAL CALL COUNT values from those rows
4. Finds rows with matching GLOBAL CALL COUNT but empty METAL TRACE REPLAY SESSION ID
5. Outputs these rows to a CSV suffixed with "cleaned_npe_data"
"""

import pandas as pd
import argparse
import sys
import os
from pathlib import Path


def process_npe_data(df):
    """
    Extract NPE data from the performance report dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe to process
    
    Returns:
        tuple: (npe_dataframe, stats_dict) or (None, None) on error
    """
    try:
        original_rows = len(df)
        
        # Check if required columns exist
        required_columns = ['METAL TRACE REPLAY SESSION ID', 'GLOBAL CALL COUNT']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in dataframe")
        
        session_id_col = 'METAL TRACE REPLAY SESSION ID'
        global_call_count_col = 'GLOBAL CALL COUNT'
        
        # Step 1: Find the highest METAL TRACE REPLAY SESSION ID
        # First, get rows that have valid (non-empty) session IDs
        valid_session_df = df[df[session_id_col].notna()]
        valid_session_df = valid_session_df[valid_session_df[session_id_col] != '']
        valid_session_df = valid_session_df[valid_session_df[session_id_col] != ' ']
        
        if len(valid_session_df) == 0:
            return None, None
        
        # Convert to numeric
        try:
            valid_session_df[session_id_col] = pd.to_numeric(valid_session_df[session_id_col], errors='coerce')
            valid_session_df = valid_session_df[valid_session_df[session_id_col].notna()]
            
            if len(valid_session_df) == 0:
                return None, None
                
        except Exception as e:
            raise Exception(f"Error converting session IDs to numeric: {e}")
        
        # Find the maximum session ID
        max_session_id = valid_session_df[session_id_col].max()
        
        # Check for multiple traced runs
        if max_session_id > 1:
            print("\033[93m⚠️  Warning: Found more than one traced run, npe data may be incorrect\033[0m")
        
        # Step 2: Get rows with the highest session ID
        highest_session_rows = valid_session_df[valid_session_df[session_id_col] == max_session_id]
        
        if len(highest_session_rows) == 0:
            return None, None
        
        # Step 3: Extract GLOBAL CALL COUNT values from those rows
        try:
            # Convert GLOBAL CALL COUNT to numeric
            highest_session_rows[global_call_count_col] = pd.to_numeric(
                highest_session_rows[global_call_count_col], errors='coerce'
            )
            highest_session_rows = highest_session_rows[highest_session_rows[global_call_count_col].notna()]
            
            if len(highest_session_rows) == 0:
                return None, None
                
            target_call_counts = set(highest_session_rows[global_call_count_col].unique())
            
        except Exception as e:
            raise Exception(f"Error processing GLOBAL CALL COUNT values: {e}")
        
        # Step 4: Find rows with matching GLOBAL CALL COUNT but empty METAL TRACE REPLAY SESSION ID
        # First, identify rows with empty/null session IDs
        empty_session_df = df[
            df[session_id_col].isna() | 
            (df[session_id_col] == '') | 
            (df[session_id_col] == ' ')
        ].copy()
        
        if len(empty_session_df) == 0:
            npe_data = pd.DataFrame()  # Create empty dataframe
        else:
            # Convert GLOBAL CALL COUNT to numeric for comparison
            try:
                empty_session_df[global_call_count_col] = pd.to_numeric(
                    empty_session_df[global_call_count_col], errors='coerce'
                )
                empty_session_df = empty_session_df[empty_session_df[global_call_count_col].notna()]
                
                # Find rows where GLOBAL CALL COUNT matches our target values
                npe_data = empty_session_df[empty_session_df[global_call_count_col].isin(target_call_counts)]
                
            except Exception as e:
                raise Exception(f"Error processing empty session rows: {e}")
        
        # Calculate statistics
        npe_call_counts = []
        missing_counts = []
        
        if len(npe_data) > 0:
            npe_call_counts = sorted(npe_data[global_call_count_col].unique())
            found_counts = set(npe_call_counts)
            missing_counts = sorted(target_call_counts - found_counts)
        
        # Prepare statistics
        stats = {
            'original_rows': original_rows,
            'valid_session_rows': len(valid_session_df),
            'max_session_id': max_session_id,
            'highest_session_rows': len(highest_session_rows),
            'target_call_counts': sorted(target_call_counts),
            'target_count': len(target_call_counts),
            'empty_session_rows': len(empty_session_df),
            'npe_rows': len(npe_data),
            'npe_call_counts': npe_call_counts,
            'missing_counts': missing_counts
        }
        
        return npe_data, stats
        
    except Exception as e:
        raise Exception(f"Error processing dataframe: {e}")


def extract_npe_data(input_file, output_file=None):
    """
    Extract NPE data from the performance report CSV file.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file (optional)
    
    Returns:
        str: Path to the NPE data CSV file
    """
    try:
        # Read the CSV file
        print(f"Reading CSV file: {input_file}")
        df = pd.read_csv(input_file)
        
        print(f"Total number of rows: {len(df)}")
        
        # Process the data using the core function
        npe_data, stats = process_npe_data(df)
        
        if npe_data is None:
            print("Error: Failed to extract NPE data!")
            return None
        
        # Print progress messages
        print(f"Highest METAL TRACE REPLAY SESSION ID found: {stats['max_session_id']}")
        print(f"Number of rows with highest session ID: {stats['highest_session_rows']}")
        print(f"Found {stats['target_count']} unique GLOBAL CALL COUNT values: {stats['target_call_counts']}")
        print(f"Number of rows with empty/null session ID: {stats['empty_session_rows']}")
        print(f"Number of NPE data rows found: {stats['npe_rows']}")
        
        if len(npe_data) > 0:
            print(f"NPE data GLOBAL CALL COUNT values: {stats['npe_call_counts']}")
            
            if stats['missing_counts']:
                print(f"Target call counts not found in NPE data: {stats['missing_counts']}")
        
        # Generate output filename if not provided
        if output_file is None:
            input_path = Path(input_file)
            output_file = input_path.parent / f"{input_path.stem}_cleaned_npe_data{input_path.suffix}"
        
        # Save the NPE data
        print(f"Saving NPE data to: {output_file}")
        npe_data.to_csv(output_file, index=False)
        
        print("NPE data extraction completed successfully!")
        print(f"Summary:")
        print(f"  - Total original rows: {stats['original_rows']}")
        print(f"  - Rows with valid session IDs: {stats['valid_session_rows']}")
        print(f"  - Highest session ID: {stats['max_session_id']}")
        print(f"  - Rows with highest session ID: {stats['highest_session_rows']}")
        print(f"  - Unique GLOBAL CALL COUNT values from highest session: {stats['target_count']}")
        print(f"  - Rows with empty session IDs: {stats['empty_session_rows']}")
        print(f"  - Final NPE data rows: {stats['npe_rows']}")
        
        return str(output_file)
        
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract NPE data from performance counter report CSV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script extracts NPE (Network Processing Engine) data by:
1. Finding the highest METAL TRACE REPLAY SESSION ID
2. Getting GLOBAL CALL COUNT values from those rows
3. Finding rows with matching GLOBAL CALL COUNT but empty session IDs
4. Outputting those rows as NPE data

Examples:
  python extract_npe_data.py input.csv
  python extract_npe_data.py input.csv -o npe_output.csv
  python extract_npe_data.py /path/to/report.csv --output /path/to/npe_data.csv
        """
    )
    
    parser.add_argument(
        'input_file',
        help='Path to the input CSV file'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Path to the output CSV file (default: adds _cleaned_npe_data suffix to input filename)'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist!")
        sys.exit(1)
    
    # Process the file
    result = extract_npe_data(args.input_file, args.output)
    
    if result:
        print(f"\nNPE data saved as: {result}")
        sys.exit(0)
    else:
        print("Failed to process the file!")
        sys.exit(1)


if __name__ == "__main__":
    main()
