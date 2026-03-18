#!/usr/bin/env python3
"""
Convert profiler comparison CSVs to Excel with gradient formatting on duration columns.

Reads prefill_comparison.csv and prefill_raw_comparison.csv (and decode equivalents)
from a sweep results directory and writes a single .xlsx with one sheet per CSV.
Duration/time columns are colored with a green→yellow→red gradient so hot spots
are immediately visible.

=============================================================================
USAGE
=============================================================================

 python scripts/csv_to_xlsx_with_gradient.py <sweep_results_dir> [options]

=============================================================================
OPTIONS
=============================================================================

 --output FILE        Output .xlsx path (default: <sweep_dir>/profiler_comparison.xlsx)
 --skip-if-unavailable  Exit 0 silently if xlsxwriter is not installed

=============================================================================
EXAMPLES
=============================================================================

 python scripts/csv_to_xlsx_with_gradient.py profiler_sweep_results/olmo_session12/
 python scripts/csv_to_xlsx_with_gradient.py profiler_sweep_results/olmo_session12/ \
     --output analysis/olmo_session12.xlsx
"""

import argparse
import csv
import sys
from pathlib import Path

try:
    import xlsxwriter
    import xlsxwriter.utility
except ImportError:
    if "--skip-if-unavailable" in sys.argv:
        sys.exit(0)
    print("xlsxwriter not installed. Run: pip install xlsxwriter")
    sys.exit(1)


# ── CSV reading ───────────────────────────────────────────────────────────────


def read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    """Return (headers, rows) from a CSV file."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = list(reader)
    return headers, rows


# ── duration column detection ─────────────────────────────────────────────────


def duration_col_indices(headers: list[str]) -> list[int]:
    """Return column indices whose header ends with _time_us or _dur_us."""
    return [i for i, h in enumerate(headers) if h.endswith("_time_us") or h.endswith("_dur_us")]


# ── sheet writer ──────────────────────────────────────────────────────────────


def write_sheet(
    workbook: "xlsxwriter.Workbook",
    sheet_name: str,
    headers: list[str],
    rows: list[list[str]],
):
    ws = workbook.add_worksheet(sheet_name[:31])  # Excel sheet name limit

    # Formats
    header_fmt = workbook.add_format(
        {
            "bold": True,
            "bg_color": "#2F5496",
            "font_color": "#FFFFFF",
            "border": 1,
            "text_wrap": True,
            "align": "center",
            "valign": "vcenter",
        }
    )
    sep_fmt = workbook.add_format({"bg_color": "#D9D9D9"})
    base_fmt = workbook.add_format({"border": 1, "num_format": "0.00"})
    text_fmt = workbook.add_format({"border": 1})
    int_fmt = workbook.add_format({"border": 1, "num_format": "0"})

    dur_cols = duration_col_indices(headers)
    num_rows = len(rows)

    # Write header row
    for ci, h in enumerate(headers):
        ws.write(0, ci, "" if h == "" else h, sep_fmt if h == "" else header_fmt)

    # Write data rows
    for ri, row in enumerate(rows, start=1):
        for ci in range(len(headers)):
            val = row[ci] if ci < len(row) else ""

            if headers[ci] == "":
                ws.write(ri, ci, "", sep_fmt)
                continue

            if ci in dur_cols:
                try:
                    ws.write_number(ri, ci, float(val), base_fmt)
                except ValueError:
                    ws.write(ri, ci, val, base_fmt)
                continue

            header_lc = headers[ci].lower()
            if any(header_lc.endswith(s) for s in ("_count", "_cores", "_min_cores", "_max_cores", "occurrence")):
                try:
                    ws.write_number(ri, ci, int(float(val)), int_fmt)
                except ValueError:
                    ws.write(ri, ci, val, text_fmt)
                continue

            ws.write(ri, ci, val, text_fmt)

    # Apply data bar conditional formatting to each duration column
    if num_rows > 0:
        for ci in dur_cols:
            col_letter_start = xlsxwriter.utility.xl_col_to_name(ci)
            data_range = f"{col_letter_start}2:{col_letter_start}{num_rows + 1}"
            ws.conditional_format(
                data_range,
                {
                    "type": "data_bar",
                    "bar_color": "#4472C4",  # solid blue bar
                    "bar_border_color": "#2F5496",
                    "bar_solid": True,
                    "data_bar_2010": True,  # use newer gradient-less solid bar
                    "bar_only": False,  # show number + bar
                },
            )

    # Auto-fit column widths
    for ci, h in enumerate(headers):
        if h == "":
            ws.set_column(ci, ci, 1)
        elif h == "OP_NAME":
            ws.set_column(ci, ci, 42)
        elif h == "OCCURRENCE":
            ws.set_column(ci, ci, 12)
        elif any(h.endswith(s) for s in ("_time_us", "_dur_us")):
            ws.set_column(ci, ci, 16)
        else:
            ws.set_column(ci, ci, 10)

    ws.freeze_panes(1, 0)

    print(f"  Sheet '{sheet_name}': {len(rows)} data rows, {len(dur_cols)} duration columns with data bars")


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Convert profiler comparison CSVs to colour-coded XLSX")
    parser.add_argument("sweep_dir", help="Directory containing comparison CSVs")
    parser.add_argument("--output", help="Output .xlsx path (default: <sweep_dir>/profiler_comparison.xlsx)")
    parser.add_argument(
        "--skip-if-unavailable", action="store_true", help="Exit 0 silently if xlsxwriter is unavailable"
    )
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.exists():
        print(f"Error: directory not found: {sweep_dir}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else sweep_dir / "profiler_comparison.xlsx"

    # CSVs to include, in sheet order
    csv_specs = [
        ("prefill_comparison.csv", "Prefill (aggregated)"),
        ("prefill_raw_comparison.csv", "Prefill (raw op-by-op)"),
        ("decode_comparison.csv", "Decode (aggregated)"),
        ("decode_raw_comparison.csv", "Decode (raw op-by-op)"),
    ]

    workbook = xlsxwriter.Workbook(str(output_path))

    found_any = False
    for csv_name, sheet_name in csv_specs:
        csv_path = sweep_dir / csv_name
        if not csv_path.exists():
            print(f"  Skipping {csv_name} (not found)")
            continue
        headers, rows = read_csv(csv_path)
        write_sheet(workbook, sheet_name, headers, rows)
        found_any = True

    workbook.close()

    if found_any:
        print(f"\nSaved: {output_path}")
    else:
        print("No CSV files found — nothing written.")
        output_path.unlink(missing_ok=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
