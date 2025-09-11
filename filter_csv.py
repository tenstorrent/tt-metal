#!/usr/bin/env python3
"""
Filter CSV file to keep only rows with DEVICE ID = 0
"""

import csv
import sys


def filter_csv_by_device_id(input_file, output_file, target_device_id=0):
    """Filter CSV to keep only rows with specified DEVICE ID."""

    rows_kept = 0
    rows_total = 0

    with open(input_file, "r") as infile, open(output_file, "w", newline="") as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Read and write header row
        header = next(reader)
        writer.writerow(header)

        # Find DEVICE ID column index
        try:
            device_id_index = header.index("DEVICE ID")
            print(f"Found 'DEVICE ID' column at index {device_id_index}")
        except ValueError:
            print("Error: 'DEVICE ID' column not found in CSV header")
            return

        # Process data rows
        for row in reader:
            rows_total += 1

            if len(row) > device_id_index:
                try:
                    device_id = int(row[device_id_index])
                    if device_id == target_device_id:
                        writer.writerow(row)
                        rows_kept += 1
                except (ValueError, IndexError):
                    # Skip rows with invalid device ID
                    continue

    print(f"Filtering complete:")
    print(f"  Total rows processed: {rows_total}")
    print(f"  Rows kept (DEVICE ID = {target_device_id}): {rows_kept}")
    print(f"  Rows filtered out: {rows_total - rows_kept}")
    print(f"  Output written to: {output_file}")


if __name__ == "__main__":
    input_file = "generated/profiler/reports/2025_09_12_09_31_59/llama_layer_prefill_all_devices.csv"
    output_file = "generated/profiler/reports/2025_09_12_09_31_59/llama_layer_prefill_device_0_only.csv"

    filter_csv_by_device_id(input_file, output_file, target_device_id=0)
