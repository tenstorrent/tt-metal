#!/usr/bin/env python3
"""
Process profiler ops performance results CSV file.
For each operation (32 consecutive rows, one per device), find the device with maximum DEVICE KERNEL DURATION.
"""

import argparse
import csv
import sys
from pathlib import Path


def process_profiler_csv(input_file: str, output_file: str = None, num_devices: int = 32, output_min: bool = False):
    """
    Process profiler CSV file and extract max duration per operation.

    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (default: input_file with _max_duration suffix)
        num_devices: Number of devices (default: 32)
        output_min: Whether to add a column with minimum duration (default: False)
    """
    input_path = Path(input_file)

    if not input_path.exists():
        print(f"Error: Input file '{input_file}' not found", file=sys.stderr)
        sys.exit(1)

    # Generate output filename if not provided
    if output_file is None:
        output_file = input_path.parent / f"{input_path.stem}_max_duration{input_path.suffix}"

    print(f"Reading from: {input_file}")
    print(f"Writing to: {output_file}")
    print(f"Number of devices per operation: {num_devices}")

    # Read the CSV file
    with open(input_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        headers = list(reader.fieldnames)

        # Check if required column exists
        duration_col = 'DEVICE KERNEL DURATION [ns]'
        if duration_col not in headers:
            print(f"Error: Column '{duration_col}' not found in CSV", file=sys.stderr)
            print(f"Available columns: {', '.join(headers)}", file=sys.stderr)
            sys.exit(1)

        all_rows = list(reader)

    # Add min duration column if requested
    min_duration_col = 'MIN DEVICE KERNEL DURATION [ns]'
    if output_min and min_duration_col not in headers:
        # Insert after the max duration column
        duration_col_idx = headers.index(duration_col)
        headers.insert(duration_col_idx + 1, min_duration_col)

    print(f"Total rows read: {len(all_rows)}")

    # Process rows in groups of num_devices
    max_duration_rows = []
    num_operations = len(all_rows) // num_devices

    if len(all_rows) % num_devices != 0:
        print(f"Warning: Total rows ({len(all_rows)}) is not divisible by {num_devices}", file=sys.stderr)
        print(f"Processing {num_operations} complete operations", file=sys.stderr)

    for i in range(num_operations):
        start_idx = i * num_devices
        end_idx = start_idx + num_devices
        operation_rows = all_rows[start_idx:end_idx]

        # Find the row with maximum DEVICE KERNEL DURATION
        max_row = None
        max_duration = -1
        min_duration = float('inf')

        for row in operation_rows:
            try:
                duration_str = row[duration_col].strip()
                if duration_str:
                    duration = float(duration_str)
                    if duration > max_duration:
                        max_duration = duration
                        max_row = row
                    if duration < min_duration:
                        min_duration = duration
            except (ValueError, KeyError) as e:
                print(f"Warning: Could not parse duration for row {start_idx + operation_rows.index(row)}: {e}", file=sys.stderr)
                continue

        if max_row is not None:
            # Add min duration column if requested
            if output_min:
                max_row[min_duration_col] = str(int(min_duration)) if min_duration != float('inf') else ''

            max_duration_rows.append(max_row)
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{num_operations} operations...")

    # Write output CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(max_duration_rows)

    print(f"\nProcessing complete!")
    print(f"Operations processed: {len(max_duration_rows)}")
    print(f"Output written to: {output_file}")

    # Print some statistics
    if max_duration_rows:
        print("\n--- Statistics ---")
        durations = [float(row[duration_col]) for row in max_duration_rows if row[duration_col].strip()]
        if durations:
            print(f"Total operations: {len(durations)}")
            print(f"Max DEVICE KERNEL DURATION: {max(durations):,.0f} ns")
            print(f"Min DEVICE KERNEL DURATION: {min(durations):,.0f} ns")
            print(f"Avg DEVICE KERNEL DURATION: {sum(durations)/len(durations):,.0f} ns")


def main():
    parser = argparse.ArgumentParser(
        description='Process profiler ops performance results CSV file. '
                    'For each operation (consecutive rows across devices), extract the device with maximum DEVICE KERNEL DURATION.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process with default settings (32 devices)
  python process_profiler_ops.py ops_perf_results.csv

  # Specify output file
  python process_profiler_ops.py ops_perf_results.csv -o max_durations.csv

  # Include minimum duration column
  python process_profiler_ops.py ops_perf_results.csv --output-min

  # Use different number of devices
  python process_profiler_ops.py ops_perf_results.csv -n 16
        """
    )

    parser.add_argument('input_file', help='Input CSV file path')
    parser.add_argument('-o', '--output', help='Output CSV file path (default: input_file_max_duration.csv)')
    parser.add_argument('-n', '--num-devices', type=int, default=32,
                        help='Number of devices per operation (default: 32)')
    parser.add_argument('--output-min', action='store_true',
                        help='Add a column with minimum device kernel duration')

    args = parser.parse_args()

    process_profiler_csv(args.input_file, args.output, args.num_devices, args.output_min)


if __name__ == '__main__':
    main()
