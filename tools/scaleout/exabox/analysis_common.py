#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for log analysis scripts."""

import argparse
import csv
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Error domain constants
DOMAIN_DISPATCH = "dispatch"
DOMAIN_FABRIC = "fabric"
DOMAIN_VALIDATION = "validation"


class Colors:
    GREEN = "\033[0;32m"
    RED = "\033[0;31m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[0;34m"
    CYAN = "\033[0;36m"
    BOLD = "\033[1m"
    NC = "\033[0m"

    @classmethod
    def disable(cls):
        """Disable colors (for --no-color flag or non-terminal output)."""
        for attr in ["GREEN", "RED", "YELLOW", "BLUE", "CYAN", "BOLD", "NC"]:
            setattr(cls, attr, "")


def read_log_file(filepath: str) -> str | None:
    """Read a log file with proper encoding handling.

    Returns file content as string, or None on failure.
    """
    try:
        with open(filepath, encoding="utf-8", errors="replace") as f:
            return f.read()
    except (OSError, PermissionError) as e:
        print(f"Warning: Could not read {filepath}: {e}", file=sys.stderr)
        return None


def deduplicate_and_count_messages(messages: list[str]) -> dict[str, int]:
    """Deduplicate messages by normalizing timestamps and MPI rank prefixes."""
    message_counts: dict[str, int] = defaultdict(int)
    for message in messages:
        normalized = re.sub(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+", "<timestamp>", message)
        normalized = re.sub(r"\[1,\d+\]<std(out|err)>:", "", normalized)
        message_counts[normalized.strip()] += 1
    return dict(message_counts)


def clean_mpi_line(line: str) -> str:
    """Remove MPI stdout/stderr prefixes from log line."""
    for prefix in ["<stdout>:", "<stderr>:"]:
        if prefix in line:
            line = line.split(prefix)[-1]
    return line.strip()


def print_section_header(title: str, width: int = 50) -> None:
    """Print a section header with separator lines."""
    print("=" * width)
    print(title)
    print("=" * width)


def print_separator(width: int = 50, char: str = "-") -> None:
    """Print a separator line."""
    print(char * width)


def print_message_section(messages: list[str], header: str, color: str, max_items: int = 10) -> None:
    """Print a section of deduplicated messages with counts."""
    print(f"{color}{header}{Colors.NC}")

    message_counts = deduplicate_and_count_messages(messages)
    items_to_show = list(sorted(message_counts.items(), key=lambda x: -x[1]))[:max_items]

    for message, count in items_to_show:
        if count > 1:
            print(f"  [{count}x] {message}")
        else:
            print(f"  {message}")

    if len(message_counts) > max_items:
        print(f"  ... ({len(message_counts) - max_items} more unique types)")
    print()


def analysis_timestamp() -> str:
    """Return current timestamp in ISO 8601 format."""
    return datetime.now().isoformat()


def analysis_timestamp_display() -> str:
    """Return current timestamp for display."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def build_base_argparser(description: str) -> argparse.ArgumentParser:
    """Build an argument parser with common flags shared across all analyzers."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    parser.add_argument("--csv", type=str, metavar="PATH", help="Save analysis results as CSV file(s) at PATH")
    parser.add_argument(
        "--csv-prefix",
        type=str,
        metavar="PREFIX",
        help=(
            "Optional prefix for CSV filenames. When set, --csv is treated as the "
            "output directory and files are named {domain}_test_{PREFIX}_*.csv "
            '(e.g. --csv-prefix "run42_3" produces dispatch_test_run42_3_summary.csv).'
        ),
    )
    parser.add_argument(
        "--hosts",
        type=str,
        metavar="HOSTS",
        help="Comma-separated list of hosts the test ran on (written to CSV output).",
    )
    return parser


def apply_common_args(args: argparse.Namespace) -> None:
    """Apply common argument side effects (e.g. disabling colors)."""
    if args.no_color or args.json:
        Colors.disable()


def validate_file_exists(path: str) -> Path:
    """Validate that a file exists, exit with error if not."""
    p = Path(path)
    if not p.is_file():
        print(f"Error: Log file not found: {path}", file=sys.stderr)
        sys.exit(1)
    return p


def validate_dir_exists(path: str) -> Path:
    """Validate that a directory exists, exit with error if not."""
    p = Path(path)
    if not p.is_dir():
        print(f"Error: Directory not found: {path}", file=sys.stderr)
        sys.exit(1)
    return p


def csv_stem_and_suffix(csv_path: str, csv_prefix: str | None, domain: str) -> tuple[str, str]:
    """Compute CSV file stem and suffix from CLI arguments.

    When *csv_prefix* is provided, *csv_path* is treated as a directory and
    filenames follow the documented convention ``{domain}_test_{prefix}``.
    Otherwise *csv_path* is split into stem + suffix directly (original
    behaviour).
    """
    csv_p = Path(csv_path)
    if csv_prefix:
        directory = csv_p if csv_p.suffix == "" else csv_p.parent
        stem = str(Path(directory) / f"{domain}_test_{csv_prefix}")
        suffix = ".csv"
    else:
        suffix = csv_p.suffix or ".csv"
        stem = str(csv_p.with_suffix(""))
    return stem, suffix


def write_csv(rows: list[dict], filepath: str, fieldnames: list[str]) -> None:
    """Write a list of dicts as a CSV file.

    Creates parent directories if needed. Prints the output path to stderr.
    """
    if not rows:
        print(f"No data to write to {filepath}", file=sys.stderr)
        return

    out = Path(filepath)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV written: {out} ({len(rows)} rows)", file=sys.stderr)
