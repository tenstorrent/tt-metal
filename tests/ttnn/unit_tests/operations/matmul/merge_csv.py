#!/usr/bin/env python3
"""
Script to merge all CSV files in the current directory into a single CSV file.
Adds a 'source_file' column to identify which CSV each row came from.
"""

import pandas as pd
import glob
import os
import csv
from pathlib import Path


def merge_csv_files():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent

    # Find all CSV files in the directory
    csv_files = sorted(glob.glob(str(script_dir / "*.csv")))

    # Exclude the output file if it already exists
    output_file = script_dir / "all_numeric_results_merged.csv"
    csv_files = [f for f in csv_files if os.path.basename(f) != output_file.name]

    if not csv_files:
        print("No CSV files found in the directory.")
        return

    print(f"Found {len(csv_files)} CSV files to merge:")

    all_dataframes = []

    for csv_file in csv_files:
        try:
            # Read CSV file - handle inconsistent column counts (old vs new format)
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)
                if not rows:
                    continue
                # Find max columns across all rows
                max_cols = max(len(row) for row in rows)
                # Pad header and rows to max_cols
                header = rows[0] + [f"col_{i}" for i in range(len(rows[0]), max_cols)]
                data_rows = [row + [None] * (max_cols - len(row)) for row in rows[1:]]
                df = pd.DataFrame(data_rows, columns=header[:max_cols])

            # Add source file column
            source_name = os.path.splitext(os.path.basename(csv_file))[0]
            df.insert(0, "source_file", source_name)

            all_dataframes.append(df)
            print(f"  ✓ Loaded {os.path.basename(csv_file)} ({len(df)} rows)")

        except Exception as e:
            print(f"  ✗ Error processing {os.path.basename(csv_file)}: {e}")

    if not all_dataframes:
        print("No data to merge.")
        return

    # Concatenate all dataframes
    merged_df = pd.concat(all_dataframes, ignore_index=True)

    # Write to output CSV
    merged_df.to_csv(output_file, index=False)

    print(f"\n✓ Successfully created: {output_file}")
    print(f"  Total rows: {len(merged_df)}")
    print(f"  Total columns: {len(merged_df.columns)}")
    print(f"  Source files: {len(csv_files)}")


if __name__ == "__main__":
    merge_csv_files()
