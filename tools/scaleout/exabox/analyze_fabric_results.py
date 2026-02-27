#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Analyze fabric test logs for pass/fail summary."""

import json
import os
import re
import sys
from dataclasses import dataclass, field

from tools.scaleout.exabox.analysis_common import (
    DOMAIN_FABRIC,
    Colors,
    analysis_timestamp,
    analysis_timestamp_display,
    apply_common_args,
    build_base_argparser,
    csv_stem_and_suffix,
    print_section_header,
    print_separator,
    read_log_file,
    validate_file_exists,
    write_csv,
)

# Constants for limiting output
MAX_WARNINGS = 10
MAX_ERRORS = 10

# Error categories for fabric tests.
# Maps category_key -> (compiled_regex, severity, exclude_regex_or_None).
# Add new entries here to detect new error types automatically.
FABRIC_ERROR_CATEGORIES = {
    "tt_fatal": (
        re.compile(r"critical|TT_FATAL|segmentation fault|Signal: (Aborted|Segmentation)", re.IGNORECASE),
        "critical",
        re.compile(r"End of error message|Unknown error|MPI_ERRORS_ARE_FATAL"),
    ),
    "timeout": (
        re.compile(r"warning.*(timed out|failed)", re.IGNORECASE),
        "warning",
        re.compile(r"\[  FAILED  \]"),
    ),
}

# CSV field definitions
SUMMARY_CSV_FIELDS = [
    "timestamp",
    "log_file",
    "domain",
    "test_passed",
    "num_hosts",
    "warnings_count",
    "critical_errors_count",
    "status",
]

DETAIL_CSV_FIELDS = [
    "timestamp",
    "log_file",
    "domain",
    "record_type",
    "error_category",
    "error_severity",
    "error_message",
    "error_count",
]


@dataclass
class LogAnalysis:
    """Analysis results for fabric test log file."""

    filepath: str
    content: str = ""
    all_passed: bool = False
    num_hosts: int = 0
    warnings: list[str] = field(default_factory=list)
    critical_errors: list[str] = field(default_factory=list)
    categorized_errors: list[tuple[str, str, str]] = field(default_factory=list)


def analyze_log_file(filepath: str) -> LogAnalysis:
    """Analyze a fabric test log file."""
    result = LogAnalysis(filepath=filepath)

    content = read_log_file(filepath)
    if content is None:
        return result
    result.content = content

    # Count unique MPI ranks to determine number of hosts
    mpi_ranks = set()
    for match in re.finditer(r"^\[1,(\d+)\]", content, re.MULTILINE):
        mpi_ranks.add(match.group(1))

    expected_hosts = len(mpi_ranks) if mpi_ranks else 1
    result.num_hosts = expected_hosts
    success_count = content.count("All tests completed successfully")
    result.all_passed = success_count == expected_hosts and expected_hosts > 0

    lines = content.split("\n")

    # Classify errors using the category registry
    for cat_key, (pattern, severity, exclude) in FABRIC_ERROR_CATEGORIES.items():
        for line in lines:
            if pattern.search(line) and (exclude is None or not exclude.search(line)):
                msg = line.strip()
                result.categorized_errors.append((cat_key, severity, msg))
                if severity == "critical":
                    result.critical_errors.append(msg)
                else:
                    result.warnings.append(msg)

    return result


def print_summary(analysis: LogAnalysis, input_path: str) -> None:
    """Print analysis summary."""
    print_section_header("Fabric Test Log Analyzer")
    print(f"Log file: {input_path}")
    print(f"Timestamp: {analysis_timestamp_display()}")
    print("=" * 50 + "\n")

    status_color = Colors.GREEN if analysis.all_passed else Colors.RED
    status_text = "PASSED" if analysis.all_passed else "FAILED"
    print(f"Test Status: {status_color}{status_text}{Colors.NC}\n")


def print_warnings_and_errors(analysis: LogAnalysis) -> None:
    """Print warnings and critical errors section."""
    if not analysis.warnings and not analysis.critical_errors:
        return

    print_section_header("WARNINGS & FAILURES")

    if analysis.critical_errors:
        print(f"\n{Colors.RED}-- Critical Errors --{Colors.NC}")
        for entry in analysis.critical_errors[:MAX_ERRORS]:
            print(f"  {entry}")

    if analysis.warnings:
        print(f"\n{Colors.YELLOW}-- Warnings/Runtime Errors --{Colors.NC}")
        for entry in analysis.warnings[:MAX_WARNINGS]:
            print(f"  {entry}")

    print("=" * 50 + "\n")


def print_recommendations(analysis: LogAnalysis) -> None:
    """Print actionable recommendations."""
    print_section_header("Recommendations")
    print()

    recs = []

    if not analysis.all_passed:
        recs.append("• Fabric test failed. Review logs for connectivity or hardware issues.")
        recs.append("• Check cable connections, port status, and fabric topology.")
        recs.append("• If issues persist, report to SYSENG and SCALEOUT teams.")

    if analysis.critical_errors:
        recs.append(
            "• Critical errors detected (TT_FATAL, segmentation faults). "
            "Check for driver issues or hardware failures."
        )
        recs.append("• Escalate to SYSENG team for hardware diagnostics.")

    if analysis.warnings:
        recs.append("• Runtime warnings detected. Review timeout or failed operation messages.")

    if analysis.all_passed and not analysis.critical_errors:
        recs.append(f"{Colors.GREEN}✓ All fabric tests passed successfully. Cluster fabric is healthy.{Colors.NC}")

    for r in recs:
        print(r)
    print()


def output_json(analysis: LogAnalysis, input_path: str):
    """Output JSON results."""
    result = {
        "timestamp": analysis_timestamp(),
        "log_file": input_path,
        "test_passed": analysis.all_passed,
        "warnings_count": len(analysis.warnings),
        "critical_errors_count": len(analysis.critical_errors),
    }
    print(json.dumps(result, indent=2))


def output_text(analysis: LogAnalysis, input_path: str):
    """Output text results."""
    print_summary(analysis, input_path)
    print_warnings_and_errors(analysis)
    print_recommendations(analysis)

    print_separator()
    print("FINAL TEST RESULT")
    print_separator()

    if analysis.all_passed:
        print(f"{Colors.GREEN}✓ All fabric tests PASSED{Colors.NC}")
    else:
        print(f"{Colors.RED}✗ Fabric tests FAILED{Colors.NC}")


def output_csv(analysis: LogAnalysis, csv_path: str, csv_prefix: str | None = None) -> None:
    """Write analysis results to CSV files."""
    ts = analysis_timestamp()
    log_basename = os.path.basename(analysis.filepath)

    summary_row = {
        "timestamp": ts,
        "log_file": log_basename,
        "domain": DOMAIN_FABRIC,
        "test_passed": analysis.all_passed,
        "num_hosts": analysis.num_hosts,
        "warnings_count": len(analysis.warnings),
        "critical_errors_count": len(analysis.critical_errors),
        "status": "PASSED" if analysis.all_passed else "FAILED",
    }

    stem, suffix = csv_stem_and_suffix(csv_path, csv_prefix, DOMAIN_FABRIC)

    write_csv([summary_row], f"{stem}_summary{suffix}", SUMMARY_CSV_FIELDS)

    # Detail CSV: one row per categorized error, deduplicated with counts
    error_counts: dict[tuple[str, str], list[str]] = {}
    for cat_key, severity, msg in analysis.categorized_errors:
        key = (cat_key, severity)
        if key not in error_counts:
            error_counts[key] = []
        error_counts[key].append(msg)

    detail_rows: list[dict] = []
    for (cat_key, severity), msgs in error_counts.items():
        detail_rows.append(
            {
                "timestamp": ts,
                "log_file": log_basename,
                "domain": DOMAIN_FABRIC,
                "record_type": "error",
                "error_category": cat_key,
                "error_severity": severity,
                "error_message": msgs[0][:500],
                "error_count": len(msgs),
            }
        )

    if detail_rows:
        write_csv(detail_rows, f"{stem}_details{suffix}", DETAIL_CSV_FIELDS)


def main():
    parser = build_base_argparser("Analyze fabric test log for pass/fail summary.")
    parser.add_argument("path", help="Fabric test log file to analyze")
    args = parser.parse_args()
    apply_common_args(args)

    log_file = validate_file_exists(args.path)
    analysis = analyze_log_file(str(log_file))

    if args.csv:
        output_csv(analysis, args.csv, csv_prefix=args.csv_prefix)

    if args.json:
        output_json(analysis, args.path)
    else:
        output_text(analysis, args.path)

    if analysis.all_passed:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
