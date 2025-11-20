#!/usr/bin/env python3
"""
Cleanup script for performance counter reports.

This script:
1. Removes all rows with op type "signpost"
2. Finds the highest METAL TRACE REPLAY SESSION ID
3. Keeps only rows with the highest session ID
4. Removes rows without a value for METAL TRACE REPLAY SESSION ID
"""

import pandas as pd
import argparse
import sys
import os
from pathlib import Path


def cleanup_report(input_file, output_file=None):
    """
    Clean up the performance report CSV file.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file (optional)
    
    Returns:
        str: Path to the cleaned CSV file
    """
    try:
        # Read the CSV file
        print(f"Reading CSV file: {input_file}")
        df = pd.read_csv(input_file)
        
        print(f"Original number of rows: {len(df)}")
        
        # Check if required columns exist
        if 'OP TYPE' not in df.columns:
            raise ValueError("Column 'OP TYPE' not found in CSV file")
        
        if 'METAL TRACE REPLAY SESSION ID' not in df.columns:
            raise ValueError("Column 'METAL TRACE REPLAY SESSION ID' not found in CSV file")
        
        # Step 1: Remove rows with op type "signpost"
        signpost_count = len(df[df['OP TYPE'] == 'signpost'])
        print(f"Removing {signpost_count} rows with OP TYPE = 'signpost'")
        df = df[df['OP TYPE'] != 'signpost']
        print(f"Rows after removing signpost: {len(df)}")
        
        # Step 2: Remove rows that don't have a value for METAL TRACE REPLAY SESSION ID
        # This includes NaN, empty strings, and None values
        session_id_col = 'METAL TRACE REPLAY SESSION ID'
        before_filter = len(df)
        
        # Filter out rows with empty/null session IDs
        df = df[df[session_id_col].notna()]  # Remove NaN values
        df = df[df[session_id_col] != '']    # Remove empty strings
        df = df[df[session_id_col] != ' ']   # Remove whitespace-only strings
        
        empty_session_count = before_filter - len(df)
        print(f"Removing {empty_session_count} rows with empty/null METAL TRACE REPLAY SESSION ID")
        print(f"Rows after removing empty session IDs: {len(df)}")
        
        if len(df) == 0:
            print("Warning: No rows remaining after filtering!")
            return None
        
        # Step 3: Find the highest METAL TRACE REPLAY SESSION ID and keep only those rows
        try:
            # Convert to numeric, handling any string values that might exist
            df[session_id_col] = pd.to_numeric(df[session_id_col], errors='coerce')
            
            # Remove any rows where conversion failed (became NaN)
            df = df[df[session_id_col].notna()]
            
            if len(df) == 0:
                print("Warning: No valid numeric session IDs found!")
                return None
            
            # Find the maximum session ID
            max_session_id = df[session_id_col].max()
            print(f"Highest METAL TRACE REPLAY SESSION ID found: {max_session_id}")
            
            # Keep only rows with the highest session ID
            df_filtered = df[df[session_id_col] == max_session_id]
            
            other_session_count = len(df) - len(df_filtered)
            print(f"Removing {other_session_count} rows with lower session IDs")
            print(f"Final number of rows: {len(df_filtered)}")
            
        except Exception as e:
            print(f"Error processing session IDs: {e}")
            return None
        
        # Generate output filename if not provided
        if output_file is None:
            input_path = Path(input_file)
            output_file = input_path.parent / f"{input_path.stem}_cleaned{input_path.suffix}"
        
        # Save the cleaned data
        print(f"Saving cleaned data to: {output_file}")
        df_filtered.to_csv(output_file, index=False)
        
        print("Cleanup completed successfully!")
        print(f"Summary:")
        print(f"  - Original rows: {pd.read_csv(input_file).shape[0]}")
        print(f"  - Signpost rows removed: {signpost_count}")
        print(f"  - Empty session ID rows removed: {empty_session_count}")
        print(f"  - Lower session ID rows removed: {other_session_count}")
        print(f"  - Final rows: {len(df_filtered)}")
        print(f"  - Highest session ID kept: {max_session_id}")
        
        return str(output_file)
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return None


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(
        description="Clean up performance counter report CSV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cleanup_report.py input.csv
  python cleanup_report.py input.csv -o cleaned_output.csv
  python cleanup_report.py /path/to/report.csv --output /path/to/cleaned.csv
        """
    )
    
    parser.add_argument(
        'input_file',
        help='Path to the input CSV file'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Path to the output CSV file (default: adds _cleaned suffix to input filename)'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist!")
        sys.exit(1)
    
    # Process the file
    result = cleanup_report(args.input_file, args.output)
    
    if result:
        print(f"\nCleaned file saved as: {result}")
        sys.exit(0)
    else:
        print("Failed to process the file!")
        sys.exit(1)


if __name__ == "__main__":
    main()

