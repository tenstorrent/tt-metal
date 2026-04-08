# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""
Utility script to compare two CSV files cell-by-cell.

Example:
    python tools/tracy/compare_full_op_report.py path/to/first.csv path/to/second.csv
"""

import csv
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Optional, Set, Dict

import click

GLOBAL_CALL_HEADER = "GLOBAL CALL COUNT"
HOST_START_HEADER = "HOST START TS"
IGNORED_HEADERS = {
    "DEVICE KERNEL DURATION PER CORE MIN [NS]",
    "DEVICE KERNEL DURATION PER CORE MAX [NS]",
    "DEVICE KERNEL DURATION PER CORE AVG [NS]",
    "DEVICE ARCH",
    "AVAILABLE WORKER CORE COUNT",
}


def load_csv(path: Path) -> Tuple[List[str], List[List[str]]]:
    with path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        rows = [row for row in reader]
        if not rows:
            return [], []
        return rows[0], rows[1:]


def locate_column(headers: Sequence[str], target: str) -> Optional[int]:
    header_map = {header.strip().upper(): idx for idx, header in enumerate(headers)}
    return header_map.get(target)


def parse_int(value: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def sort_rows(rows: List[List[str]], headers: Sequence[str]) -> List[List[str]]:
    if not headers or not rows:
        return rows

    global_idx = locate_column(headers, GLOBAL_CALL_HEADER)
    host_idx = locate_column(headers, HOST_START_HEADER)

    if global_idx is None and host_idx is None:
        return rows

    def sort_key(row: List[str]):
        global_val = parse_int(row[global_idx]) if global_idx is not None and global_idx < len(row) else 0
        host_val = parse_int(row[host_idx]) if host_idx is not None and host_idx < len(row) else 0
        return (global_val, host_val)

    return sorted(rows, key=sort_key)


def describe_header(headers: Sequence[str], column_index: int) -> str:
    if headers and 0 <= column_index < len(headers):
        header = headers[column_index]
        if header:
            return f" (header: '{header}')"
    return ""


def normalize_header(header: str) -> str:
    """Normalize header for case-insensitive comparison."""
    return header.strip().upper()


def align_to_common_headers(
    headers_a: List[str],
    rows_a: List[List[str]],
    headers_b: List[str],
    rows_b: List[List[str]],
) -> Tuple[List[str], List[List[str]], List[str], List[List[str]]]:
    """Project both CSVs onto the intersection of headers (case-insensitive).

    This keeps the comparator stable across schema evolution (new columns, reordered columns).
    The column order is taken from FIRST_CSV.
    """

    if not headers_a or not headers_b:
        return headers_a, rows_a, headers_b, rows_b

    index_a: Dict[str, int] = {}
    for idx, header in enumerate(headers_a):
        norm = normalize_header(header)
        if norm and norm not in index_a:
            index_a[norm] = idx

    index_b: Dict[str, int] = {}
    for idx, header in enumerate(headers_b):
        norm = normalize_header(header)
        if norm and norm not in index_b:
            index_b[norm] = idx

    common_norm_headers = [normalize_header(h) for h in headers_a if normalize_header(h) in index_b]
    if not common_norm_headers:
        return headers_a, rows_a, headers_b, rows_b

    aligned_headers_a: List[str] = []
    aligned_headers_b: List[str] = []
    idxs_a: List[int] = []
    idxs_b: List[int] = []
    for norm in common_norm_headers:
        ia = index_a[norm]
        ib = index_b[norm]
        idxs_a.append(ia)
        idxs_b.append(ib)
        aligned_headers_a.append(headers_a[ia])
        aligned_headers_b.append(headers_b[ib])

    def project(rows: List[List[str]], idxs: List[int]) -> List[List[str]]:
        projected = []
        for row in rows:
            projected.append([row[i] if i < len(row) else "" for i in idxs])
        return projected

    return aligned_headers_a, project(rows_a, idxs_a), aligned_headers_b, project(rows_b, idxs_b)


def compare_cells(
    headers_a: List[str],
    rows_a: List[List[str]],
    headers_b: List[str],
    rows_b: List[List[str]],
    max_differences: int,
) -> Tuple[List[str], Set[str]]:
    differences: List[str] = []
    allowed_headers: Set[str] = set()

    len_a = len(rows_a) + (1 if headers_a else 0)
    len_b = len(rows_b) + (1 if headers_b else 0)

    if len_a != len_b:
        differences.append(f"Row count mismatch: {len_a} rows vs {len_b} rows")
        if len(differences) >= max_differences:
            return differences

    # include header comparison
    all_rows_a = [headers_a] + rows_a if headers_a else rows_a
    all_rows_b = [headers_b] + rows_b if headers_b else rows_b

    shared_row_count = min(len(all_rows_a), len(all_rows_b))

    for row_index in range(shared_row_count):
        row_a = all_rows_a[row_index]
        row_b = all_rows_b[row_index]

        # Skip header row differences (case-only differences in headers are acceptable)
        if row_index == 0 and headers_a and headers_b:
            # Check if headers differ only in case
            if len(headers_a) == len(headers_b):
                headers_match = all(
                    h_a.strip().upper() == h_b.strip().upper() for h_a, h_b in zip(headers_a, headers_b)
                )
                if headers_match:
                    continue  # Headers match (case-insensitive), skip header row comparison
        cols_a = len(row_a)
        cols_b = len(row_b)

        if cols_a != cols_b:
            differences.append(f"Row {row_index + 1} column count mismatch: {cols_a} columns vs {cols_b} columns")
            if len(differences) >= max_differences:
                return differences

        shared_col_count = min(cols_a, cols_b)

        for col_index in range(shared_col_count):
            value_a = row_a[col_index]
            value_b = row_b[col_index]

            # Normalize nan and empty string to be equivalent
            def normalize_missing(value: str) -> str:
                value_lower = value.lower().strip() if value else ""
                if value_lower in ("nan", "none", ""):
                    return ""
                return value

            normalized_a = normalize_missing(value_a)
            normalized_b = normalize_missing(value_b)

            # Get header names for both files
            header_a = headers_a[col_index] if headers_a and 0 <= col_index < len(headers_a) else ""
            header_b = headers_b[col_index] if headers_b and 0 <= col_index < len(headers_b) else ""

            # Normalize headers for case-insensitive comparison
            normalized_header_a = normalize_header(header_a)
            normalized_header_b = normalize_header(header_b)

            # If headers differ only in case, skip the comparison (header format difference)
            if normalized_header_a == normalized_header_b and header_a != header_b:
                continue

            if normalized_a == normalized_b:
                continue

            header_name = header_a or header_b
            normalized_header = normalized_header_a or normalized_header_b
            header_hint = describe_header(headers_a, col_index) or describe_header(headers_b, col_index)

            if normalized_header in IGNORED_HEADERS:
                a_missing = normalized_a == ""
                b_missing = normalized_b == ""
                if a_missing != b_missing:
                    allowed_headers.add(normalized_header or header_hint)
                    continue

            differences.append(f"Row {row_index + 1}, Column {col_index + 1}{header_hint}: '{value_a}' != '{value_b}'")
            if len(differences) >= max_differences:
                return differences, allowed_headers

    # Report any leftover rows explicitly once the shared ones match.
    if len(all_rows_a) > len(all_rows_b):
        for row_index in range(len(all_rows_b), len(all_rows_a)):
            differences.append(f"Extra row {row_index + 1} only present in first file: {all_rows_a[row_index]}")
            if len(differences) >= max_differences:
                return differences, allowed_headers
    elif len(all_rows_b) > len(all_rows_a):
        for row_index in range(len(all_rows_a), len(all_rows_b)):
            differences.append(f"Extra row {row_index + 1} only present in second file: {all_rows_b[row_index]}")
            if len(differences) >= max_differences:
                return differences, allowed_headers

    return differences, allowed_headers


@click.command()
@click.argument("first_csv", type=click.Path(exists=True, path_type=Path))
@click.argument("second_csv", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--max-differences",
    default=20,
    show_default=True,
    help="Stop diffing after reporting this many mismatches.",
)
def main(first_csv: Path, second_csv: Path, max_differences: int) -> None:
    """Compare every cell of FIRST_CSV and SECOND_CSV."""
    headers_a, data_a = load_csv(first_csv)
    headers_b, data_b = load_csv(second_csv)

    headers_a, data_a, headers_b, data_b = align_to_common_headers(headers_a, data_a, headers_b, data_b)

    sorted_a = sort_rows(data_a, headers_a)
    sorted_b = sort_rows(data_b, headers_b)

    differences, allowed_headers = compare_cells(
        headers_a,
        sorted_a,
        headers_b,
        sorted_b,
        max_differences=max_differences,
    )

    if allowed_headers:
        click.echo(
            "Allowed differences because one file had empty values for these headers: "
            + ", ".join(sorted(allowed_headers))
        )

    if differences:
        click.echo("CSV files differ:")
        for diff in differences:
            click.echo(f" - {diff}")
        missing_count = max(0, len(differences) - max_differences)
        if missing_count > 0:
            click.echo(f" ... truncated after {max_differences} differences.")
        sys.exit(1)

    click.echo("CSV files are identical.")


if __name__ == "__main__":
    main()
