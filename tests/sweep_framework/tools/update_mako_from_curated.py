#!/usr/bin/env python3
import argparse
import csv
import os
from typing import Dict, Iterable, List, Tuple


MAKO_EXEC_N150_COL = "execution time (n150)"


def normalize_layout_to_mako(layout: str) -> str:
    if not layout:
        return layout
    layout = layout.strip()
    mapping = {
        "Layout.TILE": "TILE_LAYOUT",
        "Layout.ROW_MAJOR": "ROW_MAJOR_LAYOUT",
    }
    return mapping.get(layout, layout)


def normalize_dtype_to_mako(dtype: str) -> str:
    if not dtype:
        return dtype
    dtype = dtype.strip()
    # Convert DataType.X -> x (lowercase)
    if dtype.startswith("DataType."):
        return dtype.split(".", 1)[1].lower()
    return dtype.lower()


def curated_key_for_mako(row: Dict[str, str]) -> Tuple[str, str, str, str, str, str, str]:
    return (
        (row.get("batch_sizes") or "").strip(),
        (row.get("input_shape") or "").strip(),
        (row.get("height") or "").strip(),
        (row.get("width") or "").strip(),
        normalize_layout_to_mako(row.get("layout") or ""),
        normalize_dtype_to_mako(row.get("input_dtype") or ""),
        (row.get("op_name") or "").strip(),
    )


def mako_key(row: Dict[str, str]) -> Tuple[str, str, str, str, str, str, str]:
    return (
        (row.get("batch_sizes") or "").strip(),
        (row.get("input_shape") or "").strip(),
        (row.get("height") or "").strip(),
        (row.get("width") or "").strip(),
        (row.get("layout") or "").strip(),
        (row.get("input_dtype") or "").strip(),
        (row.get("op_name") or "").strip(),
    )


def load_curated_map(curated_csv_path: str) -> Dict[Tuple[str, str, str, str, str, str, str], str]:
    curated_map: Dict[Tuple[str, str, str, str, str, str, str], str] = {}
    with open(curated_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {
            "batch_sizes",
            "input_shape",
            "height",
            "width",
            "layout",
            "input_dtype",
            "op_name",
            "execution_time",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Curated CSV missing required columns: {sorted(missing)}")
        for row in reader:
            exec_time = (row.get("execution_time") or "").strip()
            if not exec_time:
                continue
            key = curated_key_for_mako(row)
            curated_map.setdefault(key, exec_time)
    return curated_map


def update_mako_rows(
    mako_rows: Iterable[Dict[str, str]],
    curated_map: Dict[Tuple[str, str, str, str, str, str, str], str],
    overwrite: bool,
) -> Tuple[List[Dict[str, str]], int, int]:
    updated_rows: List[Dict[str, str]] = []
    total = 0
    updated = 0
    for row in mako_rows:
        total += 1
        key = mako_key(row)
        exec_time_new = curated_map.get(key)
        if exec_time_new:
            current = (row.get(MAKO_EXEC_N150_COL) or "").strip()
            if overwrite or not current:
                row[MAKO_EXEC_N150_COL] = exec_time_new
                updated += 1
        updated_rows.append(row)
    return updated_rows, total, updated


def default_output_path(mako_csv_path: str) -> str:
    d = os.path.dirname(os.path.abspath(mako_csv_path))
    stem, ext = os.path.splitext(os.path.basename(mako_csv_path))
    return os.path.join(d, f"{stem}_updated{ext or '.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fill the MAKO CSV 'execution time (n150)' column using execution_time values "
            "from the curated CSV for matching rows. Matching uses batch_sizes, input_shape, "
            "height, width, layout, input_dtype, and op_name. Layout and dtype from curated are "
            "normalized to MAKO naming."
        )
    )
    parser.add_argument("--mako-csv", required=True, help="Path to MAKO CSV to update")
    parser.add_argument("--curated-csv", required=True, help="Path to curated CSV with execution_time column")
    parser.add_argument(
        "--output-csv",
        help="Optional output CSV path (defaults to in-place). If omitted and --no-in-place is set, a *mako*_updated.csv will be created.",
    )
    parser.add_argument(
        "--no-in-place",
        action="store_true",
        help="Do not overwrite the MAKO CSV; write to *_updated.csv unless --output-csv is provided.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Also overwrite non-empty 'execution time (n150)' cells.",
    )

    args = parser.parse_args()

    # Load curated execution times
    curated_map = load_curated_map(args.curated_csv)

    # Read MAKO CSV
    with open(args.mako_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if MAKO_EXEC_N150_COL not in fieldnames:
            raise ValueError(f"MAKO CSV missing column: {MAKO_EXEC_N150_COL}")
        mako_rows = list(reader)

    # Update rows
    updated_rows, total, updated = update_mako_rows(mako_rows, curated_map, args.overwrite_existing)

    # Determine output path
    if args.output_csv:
        out_path = args.output_csv
    else:
        out_path = args.mako_csv if not args.no_in_place else default_output_path(args.mako_csv)

    # Write output
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

    print(
        f"Wrote: {out_path}\n"
        f"Total MAKO rows: {total}\n"
        f"Rows updated: {updated}\n"
        f"Curated keys available: {len(curated_map)}"
    )


if __name__ == "__main__":
    main()
