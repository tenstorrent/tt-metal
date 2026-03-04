#!/usr/bin/env python3
"""
Convert profiler comparison CSV files to Excel with gradient formatting.

Applies color gradient scaling to duration columns (_dur_us) where:
- Longer durations = darker/more filled cells (red)
- Shorter durations = lighter cells (green)
"""

import argparse
import csv
import sys
from pathlib import Path

XLSXWRITER_AVAILABLE = False
try:
    import xlsxwriter

    XLSXWRITER_AVAILABLE = True
except ImportError:
    pass


def csv_to_xlsx_with_gradient(csv_path: str, xlsx_path: str = None):
    """Convert CSV to Excel with gradient formatting on duration columns.

    Args:
        csv_path: Path to input CSV file
        xlsx_path: Path to output Excel file (default: same name with .xlsx)
    """
    csv_path = Path(csv_path)
    if xlsx_path is None:
        xlsx_path = csv_path.with_suffix(".xlsx")
    else:
        xlsx_path = Path(xlsx_path)

    # Read CSV data
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        print(f"Error: Empty CSV file: {csv_path}")
        return None

    headers = rows[0]
    data_rows = rows[1:]

    # Create workbook
    wb = xlsxwriter.Workbook(str(xlsx_path))
    ws = wb.add_worksheet(csv_path.stem[:31])  # Sheet name max 31 chars

    # Style definitions
    header_format = wb.add_format(
        {"bold": True, "bg_color": "#DAEEF3", "align": "center", "valign": "vcenter", "text_wrap": True, "border": 1}
    )

    cell_format = wb.add_format({"border": 1, "align": "left"})

    number_format = wb.add_format({"border": 1, "align": "right", "num_format": "0.00"})

    int_format = wb.add_format({"border": 1, "align": "right", "num_format": "0"})

    # Find duration columns (contain "_dur_us")
    dur_columns = []
    for col_idx, header in enumerate(headers):
        if "_dur_us" in header.lower():
            dur_columns.append(col_idx)

    # Write headers
    for col_idx, header in enumerate(headers):
        ws.write(0, col_idx, header, header_format)

    # Track column widths
    col_widths = [len(str(h)) for h in headers]

    # Write data rows
    for row_idx, row in enumerate(data_rows, 1):
        for col_idx, value in enumerate(row):
            # Update column width tracking
            col_widths[col_idx] = max(col_widths[col_idx], len(str(value)))

            # Try to convert numeric values
            try:
                if "." in value:
                    float_val = float(value)
                    ws.write(row_idx, col_idx, float_val, number_format)
                elif value.lstrip("-").isdigit():
                    int_val = int(value)
                    ws.write(row_idx, col_idx, int_val, int_format)
                else:
                    ws.write(row_idx, col_idx, value, cell_format)
            except (ValueError, AttributeError):
                ws.write(row_idx, col_idx, value, cell_format)

    # Set column widths
    for col_idx, width in enumerate(col_widths):
        ws.set_column(col_idx, col_idx, min(width + 2, 50))

    # Apply data bar (cell fill gradient) to each duration column
    num_data_rows = len(data_rows)
    if num_data_rows > 0 and dur_columns:
        for col_idx in dur_columns:
            # xlsxwriter uses 0-based indexing
            # Data starts at row 1 (row 0 is header)
            ws.conditional_format(
                1,
                col_idx,  # first_row, first_col
                num_data_rows,
                col_idx,  # last_row, last_col
                {
                    "type": "data_bar",
                    "bar_color": "#F8696B",  # Red fill
                    "bar_solid": True,  # Solid fill (not gradient bar)
                    "bar_direction": "left",  # Fill from left
                },
            )

    # Freeze the header row and first two columns (OP_NAME, OCCURRENCE)
    ws.freeze_panes(1, 2)

    # Close workbook
    wb.close()
    print(f"Created: {xlsx_path}")
    return xlsx_path


def process_directory(dir_path: str):
    """Process all comparison CSV files in a directory."""
    dir_path = Path(dir_path)

    csv_files = [
        "prefill_raw_comparison.csv",
        "decode_raw_comparison.csv",
        "prefill_comparison.csv",
        "decode_comparison.csv",
    ]

    created = []
    for csv_file in csv_files:
        csv_path = dir_path / csv_file
        if csv_path.exists():
            xlsx_path = csv_to_xlsx_with_gradient(csv_path)
            if xlsx_path:
                created.append(xlsx_path)

    return created


def main():
    parser = argparse.ArgumentParser(description="Convert profiler CSV files to Excel with gradient formatting")
    parser.add_argument("input", help="CSV file or directory containing comparison CSV files")
    parser.add_argument("--output", "-o", help="Output Excel file (only for single file input)")
    parser.add_argument(
        "--skip-if-unavailable",
        action="store_true",
        help="Silently skip if xlsxwriter is not available (for automated scripts)",
    )

    args = parser.parse_args()

    # Check if xlsxwriter is available
    if not XLSXWRITER_AVAILABLE:
        if args.skip_if_unavailable:
            print("Skipping xlsx generation (xlsxwriter not available)")
            sys.exit(0)
        else:
            print("Error: xlsxwriter not installed. Install with: pip install xlsxwriter")
            print("Use --skip-if-unavailable to silently skip xlsx generation")
            sys.exit(1)

    input_path = Path(args.input)

    if input_path.is_dir():
        created = process_directory(input_path)
        print(f"\nCreated {len(created)} Excel files")
    elif input_path.is_file():
        csv_to_xlsx_with_gradient(args.input, args.output)
    else:
        print(f"Error: Path not found: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
