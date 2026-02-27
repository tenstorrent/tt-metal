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

# Exit codes for specific failure types
EXIT_CODE_PASSED = 0
EXIT_CODE_MGD_ERROR = 1
EXIT_CODE_FW_INIT_FAILED = 2
EXIT_CODE_FABRIC_ROUTER_SYNC_TIMEOUT = 3
EXIT_CODE_TEST_HANGING = 4
EXIT_CODE_NOC_CONFLICT = 5
EXIT_CODE_ETHERNET_CORE_TIMEOUT = 6
EXIT_CODE_INCONCLUSIVE = 50
EXIT_CODE_INPUT_ERROR = 66

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
    error_category: str = "inconclusive"
    exit_code: int = EXIT_CODE_INCONCLUSIVE
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

    # Detect specific error categories for exit codes
    if result.all_passed:
        result.exit_code = EXIT_CODE_PASSED
        result.error_category = "passed"
    else:
        if re.search(
            r"(Mesh Graph Descriptor|MGD).*(could not fit|cannot fit).*physical topology", result.content, re.IGNORECASE
        ):
            result.error_category = "mgd_error"
            result.exit_code = EXIT_CODE_MGD_ERROR
        elif re.search(r"failed to initialize FW|Timeout.*waiting for physical cores to finish", result.content):
            result.error_category = "fw_init_failed"
            result.exit_code = EXIT_CODE_FW_INIT_FAILED
        elif re.search(r"Fabric Router Sync: Timeout", result.content):
            result.error_category = "fabric_router_sync_timeout"
            result.exit_code = EXIT_CODE_FABRIC_ROUTER_SYNC_TIMEOUT
        elif re.search(r"Expected NOC address.*but got", result.content):
            result.error_category = "noc_conflict"
            result.exit_code = EXIT_CODE_NOC_CONFLICT
        elif re.search(r"Timed out while waiting for active ethernet core", result.content):
            result.error_category = "ethernet_core_timeout"
            result.exit_code = EXIT_CODE_ETHERNET_CORE_TIMEOUT
        else:
            lines_stripped = [line.strip() for line in result.content.split("\n") if line.strip()]
            if lines_stripped and re.match(
                r"^\[1,\d+\]<std(?:out|err)>:\s+\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+\s+\|", lines_stripped[-1]
            ):
                result.error_category = "test_hanging"
                result.exit_code = EXIT_CODE_TEST_HANGING
            else:
                result.error_category = "inconclusive"
                result.exit_code = EXIT_CODE_INCONCLUSIVE

    lines = content.split("\n")

    # Classify errors using the category registry for CSV output
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


# Recommendation generator registry
RECOMMENDATION_GENERATORS = {}


def register_recommendation(category: str):
    """Decorator to register a recommendation generator for a category."""

    def decorator(func):
        RECOMMENDATION_GENERATORS[category] = func
        return func

    return decorator


@register_recommendation("passed")
def _recommend_passed(analysis: LogAnalysis) -> list[str]:
    return [f"{Colors.GREEN}✓ All fabric tests passed successfully. Cluster fabric is healthy.{Colors.NC}"]


@register_recommendation("mgd_error")
def _recommend_mgd_error(analysis: LogAnalysis) -> list[str]:
    return [
        f"{Colors.RED}• MGD topology mismatch detected.{Colors.NC}",
        "• Check host order in host variables against cabling descriptors.",
        "• Verify MGD file matches the physical topology.",
        "• If host ordering is correct, escalate missing connections issue.",
    ]


@register_recommendation("fw_init_failed")
def _recommend_fw_init_failed(analysis: LogAnalysis) -> list[str]:
    return [
        f"{Colors.RED}• Firmware initialization failed.{Colors.NC}",
        "• Reset boards with: tt-smi -r",
        "• Wait 30 seconds and retry fabric test.",
        "• If issue persists after reset, escalate to SYSENG.",
    ]


@register_recommendation("fabric_router_sync_timeout")
def _recommend_fabric_router_sync_timeout(analysis: LogAnalysis) -> list[str]:
    return [
        f"{Colors.RED}• Fabric Router Sync timeout detected.{Colors.NC}",
        "• Check fabric connectivity and router status.",
        "• Try reset with: tt-smi -r",
        "• Escalate to SYSENG if issue persists.",
    ]


@register_recommendation("test_hanging")
def _recommend_test_hanging(analysis: LogAnalysis) -> list[str]:
    return [
        f"{Colors.YELLOW}• Test appears to be hanging (incomplete log).{Colors.NC}",
        "• Log shows test was running but never completed.",
        "• Test may have been killed or timed out.",
        "• Check if test process is still running.",
        "• Consider increasing timeout or investigating hang cause.",
    ]


@register_recommendation("noc_conflict")
def _recommend_noc_conflict(analysis: LogAnalysis) -> list[str]:
    return [
        f"{Colors.RED}• NOC address conflict detected.{Colors.NC}",
        "• UMD detected mismatch between expected and actual NOC addresses.",
        "• This typically indicates memory mapping or firmware issue.",
        "• Try resetting boards with: tt-smi -r",
        "• If issue persists, escalate to SYSENG - may require firmware update.",
    ]


@register_recommendation("ethernet_core_timeout")
def _recommend_ethernet_core_timeout(analysis: LogAnalysis) -> list[str]:
    return [
        f"{Colors.RED}• Ethernet core activation timeout detected.{Colors.NC}",
        "• One or more ethernet cores failed to become active.",
        "• This indicates a hardware or firmware issue with ethernet cores.",
        "• Reset boards with: tt-smi -r",
        "• Wait 30 seconds and retry fabric test.",
        "• If issue persists, escalate to SYSENG - board may need replacement.",
    ]


@register_recommendation("inconclusive")
def _recommend_inconclusive(analysis: LogAnalysis) -> list[str]:
    return [
        f"{Colors.YELLOW}• Fabric test failed with unrecognized error pattern.{Colors.NC}",
        "• Review critical errors and warnings above.",
        "• Check cable connections, port status, and fabric topology.",
        "• Report to SYSENG and SCALEOUT teams with full logs.",
    ]


def print_recommendations(analysis: LogAnalysis) -> None:
    """Print actionable recommendations."""
    print_section_header("Recommendations")
    print()

    recs = []
    if analysis.error_category in RECOMMENDATION_GENERATORS:
        recs.extend(RECOMMENDATION_GENERATORS[analysis.error_category](analysis))

    if analysis.critical_errors and analysis.error_category != "passed":
        recs.append("")
        recs.append(f"{Colors.RED}• Critical errors detected - check logs above for details.{Colors.NC}")

    for r in recs:
        print(r)
    print()


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


def print_summary(analysis: LogAnalysis, input_path: str) -> None:
    """Print analysis summary."""
    print_section_header("Fabric Test Log Analyzer")
    print(f"Log file: {input_path}")
    print(f"Timestamp: {analysis_timestamp_display()}")
    print("=" * 50 + "\n")

    status_color = Colors.GREEN if analysis.all_passed else Colors.RED
    status_text = "PASSED" if analysis.all_passed else "FAILED"
    print(f"Test Status: {status_color}{status_text}{Colors.NC}")
    if not analysis.all_passed:
        print(f"Error Category: {analysis.error_category}")
        print(f"Exit Code: {analysis.exit_code}")
    print()


def output_json(analysis: LogAnalysis, input_path: str):
    """Output JSON results."""
    result = {
        "timestamp": analysis_timestamp(),
        "log_file": input_path,
        "test_passed": analysis.all_passed,
        "error_category": analysis.error_category,
        "exit_code": analysis.exit_code,
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

    sys.exit(analysis.exit_code)


if __name__ == "__main__":
    main()
