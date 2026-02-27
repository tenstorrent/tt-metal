#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Analyze dispatch test logs and generate summary with exit codes."""

import json
import os
import re
import sys
from dataclasses import dataclass, field

from tools.scaleout.exabox.analysis_common import (
    DOMAIN_DISPATCH,
    Colors,
    analysis_timestamp,
    analysis_timestamp_display,
    apply_common_args,
    build_base_argparser,
    csv_stem_and_suffix,
    print_message_section,
    print_section_header,
    print_separator,
    read_log_file,
    validate_file_exists,
    write_csv,
)

# Constants for limiting output verbosity
MAX_ERROR_LINES = 30
MAX_SKIP_REASONS = 5

# Error categories for dispatch tests.
# Maps category_key -> (compiled_regex, severity).
# Add new entries here to detect new error types automatically.
DISPATCH_ERROR_CATEGORIES = {
    "segfault": (re.compile(r"segmentation fault|core dumped", re.IGNORECASE), "critical"),
    "signal_abort": (re.compile(r"Signal: (Aborted|Segmentation)", re.IGNORECASE), "critical"),
    "arc_failure": (re.compile(r"ARC core.*failed", re.IGNORECASE), "warning"),
    "timeout": (re.compile(r"warning.*timed out", re.IGNORECASE), "warning"),
}

# CSV field definitions
SUMMARY_CSV_FIELDS = [
    "timestamp",
    "log_file",
    "domain",
    "total_processes",
    "tests_run",
    "tests_passed",
    "tests_failed",
    "tests_skipped",
    "warnings_count",
    "critical_errors_count",
    "status",
]

DETAIL_CSV_FIELDS = [
    "timestamp",
    "log_file",
    "domain",
    "record_type",
    "test_name",
    "test_status",
    "error_category",
    "error_severity",
    "error_message",
]


@dataclass
class LogAnalysis:
    """Analysis results for dispatch test log."""

    filepath: str
    content: str = ""
    total_processes: int = 0
    tests_run: set[str] = field(default_factory=set)
    tests_passed: set[str] = field(default_factory=set)
    tests_failed: set[str] = field(default_factory=set)
    tests_skipped: set[str] = field(default_factory=set)
    failure_details: list[str] = field(default_factory=list)
    skip_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    critical_errors: list[str] = field(default_factory=list)
    categorized_errors: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def num_failed(self) -> int:
        return len(self.tests_failed)

    @property
    def num_passed(self) -> int:
        return len(self.tests_passed)

    @property
    def has_failures(self) -> bool:
        return len(self.tests_failed) > 0

    @property
    def has_critical(self) -> bool:
        return len(self.critical_errors) > 0

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    @property
    def status(self) -> str:
        if self.num_failed > 0 or self.num_passed == 0:
            return "FAILED"
        return "PASSED"


def analyze_log_file(filepath: str) -> LogAnalysis:
    """Analyze a single dispatch test log file."""
    result = LogAnalysis(filepath=filepath)

    content = read_log_file(filepath)
    if content is None:
        return result
    result.content = content

    lines = content.split("\n")

    result.total_processes = content.count("Running main() from gmock_main.cc")

    for match in re.finditer(r"\[ RUN      \] (.+)", content):
        result.tests_run.add(match.group(1).strip())

    for match in re.finditer(r"\[       OK \] ([^\(]+)", content):
        result.tests_passed.add(match.group(1).strip())

    for match in re.finditer(r"\[  FAILED  \] ([^\(]+)", content):
        test_name = match.group(1).strip()
        if not re.match(r"\d+ tests?", test_name):
            result.tests_failed.add(test_name)

    for match in re.finditer(r"\[  SKIPPED \] ([^\(]+)", content):
        test_name = match.group(1).strip()
        if not re.match(r"\d+ tests?", test_name):
            result.tests_skipped.add(test_name)

    # Extract failure details (lines around FAILED markers)
    in_failure = False
    failure_context: list[str] = []
    for line in lines:
        if "[  FAILED  ]" in line:
            in_failure = True
            failure_context = [line]
        elif in_failure:
            if any(kw in line for kw in ["FAILED", "exception", "description", "Failure"]):
                failure_context.append(line)
            if len(failure_context) >= 5:
                result.failure_details.extend(failure_context)
                in_failure = False
                failure_context = []
    if failure_context:
        result.failure_details.extend(failure_context)

    # Extract skip reasons
    for i, line in enumerate(lines):
        if "Skipped$" in line and i > 0:
            prev_line = lines[i - 1]
            if any(kw in prev_line for kw in ["This suite must be run with", "requires", "needs"]):
                reason_match = re.search(r"\| info.*\| (.+)", prev_line)
                if reason_match:
                    result.skip_reasons.append(reason_match.group(1).strip())
                else:
                    result.skip_reasons.append(prev_line.strip())

    # Classify errors using the category registry
    for cat_key, (pattern, severity) in DISPATCH_ERROR_CATEGORIES.items():
        for line in lines:
            if pattern.search(line):
                msg = line.strip()
                result.categorized_errors.append((cat_key, severity, msg))
                if severity == "critical":
                    result.critical_errors.append(msg)
                else:
                    result.warnings.append(msg)

    return result


def print_summary(analysis: LogAnalysis) -> None:
    """Print test results summary."""
    num_run = len(analysis.tests_run)
    num_passed = len(analysis.tests_passed)
    num_failed = len(analysis.tests_failed)
    num_skipped = len(analysis.tests_skipped)

    print_section_header("Dispatch Test Log Analyzer")
    print(f"Log file: {analysis.filepath}")
    print(f"Timestamp: {analysis_timestamp_display()}")
    print("=" * 50 + "\n")

    print_section_header("TEST RESULTS")
    print("Overview")
    print("=" * 50)
    print(f"MPI Processes:       {analysis.total_processes}")
    print(f"Total Tests:         {num_run}")
    print(f"Passed:              {num_passed}")
    print(f"Failed:              {num_failed}")
    print(f"Skipped:             {num_skipped}")
    print()


def print_test_details(analysis: LogAnalysis) -> None:
    """Print detailed test lists."""
    num_passed = len(analysis.tests_passed)
    num_failed = len(analysis.tests_failed)
    num_skipped = len(analysis.tests_skipped)

    if num_passed > 0:
        print_separator()
        print(f"PASSED TESTS ({num_passed})")
        print_separator()
        for i, test in enumerate(sorted(analysis.tests_passed), 1):
            print(f"{i:2d}. {test}")
        print()

    if num_failed > 0:
        print_separator()
        print(f"{Colors.RED}FAILED TESTS ({num_failed}){Colors.NC}")
        print_separator()
        for i, test in enumerate(sorted(analysis.tests_failed), 1):
            print(f"{i:2d}. {test}")
        print()

        if analysis.failure_details:
            print("Failure Details:")
            for line in analysis.failure_details[:MAX_ERROR_LINES]:
                print(f"  {line}")
            if len(analysis.failure_details) > MAX_ERROR_LINES:
                print(f"  ... ({len(analysis.failure_details) - MAX_ERROR_LINES} more lines)")
            print()

    if num_skipped > 0:
        print_separator()
        print(f"{Colors.YELLOW}SKIPPED TESTS ({num_skipped}){Colors.NC}")
        print_separator()
        for i, test in enumerate(sorted(analysis.tests_skipped), 1):
            print(f"{i:2d}. {test}")
        print()

        if analysis.skip_reasons:
            print("Skip Reasons:")
            for reason in analysis.skip_reasons[:MAX_SKIP_REASONS]:
                print(f"  - {reason}")
            if len(analysis.skip_reasons) > MAX_SKIP_REASONS:
                print(f"  ... ({len(analysis.skip_reasons) - MAX_SKIP_REASONS} more reasons)")
        else:
            print("  - Check log for details")
        print()


def print_warnings_and_errors(analysis: LogAnalysis) -> None:
    """Print warnings and critical errors section."""
    if not analysis.has_warnings and not analysis.has_critical:
        return

    print_separator()
    print("WARNINGS & CRITICAL ERRORS")
    print_separator()

    if analysis.has_critical:
        print_message_section(analysis.critical_errors, "⚠️  CRITICAL ERRORS DETECTED:", Colors.RED)

    if analysis.has_warnings:
        print_message_section(analysis.warnings, "⚠️  RUNTIME WARNINGS DETECTED:", Colors.YELLOW)

    print("Note: These are runtime issues that occurred during test execution.")
    print("      They may or may not affect test results.")
    print()


def print_recommendations(analysis: LogAnalysis) -> None:
    """Print actionable recommendations."""
    print_section_header("Recommendations")
    print()

    recs = []

    if analysis.num_failed > 0:
        recs.append(
            f"• {analysis.num_failed} test(s) failed. Review failure details above and "
            "check for device communication or timing issues."
        )
        recs.append("• If issues persist, report to SYSENG and SCALEOUT teams.")

    if analysis.has_critical:
        recs.append(
            "• Critical errors detected (segmentation faults, core dumps). " "Investigate hardware or driver issues."
        )
        recs.append("• Escalate to SYSENG team for hardware diagnostics.")

    if analysis.has_warnings:
        recs.append("• Runtime warnings detected. Review timeout messages and ARC core failures.")

    if analysis.num_failed == 0 and analysis.num_passed > 0:
        if analysis.has_critical or analysis.has_warnings:
            recs.append(f"{Colors.YELLOW}✓ All tests passed, but runtime warnings/errors were detected.{Colors.NC}")
            recs.append("  Review the WARNINGS & CRITICAL ERRORS section above.")
        else:
            recs.append(
                f"{Colors.GREEN}✓ All dispatch tests passed successfully. "
                f"System is functioning correctly.{Colors.NC}"
            )
    elif analysis.num_passed == 0:
        recs.append(
            f"{Colors.RED}• No tests passed. Check if tests executed properly and "
            f"review log for initialization errors.{Colors.NC}"
        )

    for r in recs:
        print(r)
    print()


def output_json(analysis: LogAnalysis):
    """Output JSON results."""
    result = {
        "timestamp": analysis_timestamp(),
        "log_file": analysis.filepath,
        "summary": {
            "total_processes": analysis.total_processes,
            "tests_run": len(analysis.tests_run),
            "tests_passed": len(analysis.tests_passed),
            "tests_failed": len(analysis.tests_failed),
            "tests_skipped": len(analysis.tests_skipped),
            "has_failures": analysis.has_failures,
            "warnings_count": len(analysis.warnings),
            "critical_errors_count": len(analysis.critical_errors),
        },
        "tests": {
            "passed": sorted(analysis.tests_passed),
            "failed": sorted(analysis.tests_failed),
            "skipped": sorted(analysis.tests_skipped),
        },
    }
    print(json.dumps(result, indent=2))


def output_text(analysis: LogAnalysis):
    print_summary(analysis)
    print_test_details(analysis)
    print_warnings_and_errors(analysis)
    print_recommendations(analysis)

    print_separator()
    print("FINAL TEST RESULT")
    print_separator()

    if analysis.num_failed > 0:
        print(f"STATUS: {Colors.RED}FAILED{Colors.NC}")
        print("Some tests failed. See details above.")
    elif analysis.num_passed == 0:
        print(f"STATUS: {Colors.RED}FAILED (NO TESTS PASSED){Colors.NC}")
        print("No tests were successfully executed.")
    else:
        print(f"STATUS: {Colors.GREEN}PASSED{Colors.NC}")
        print("All tests passed successfully.")


def output_csv(analysis: LogAnalysis, csv_path: str, csv_prefix: str | None = None) -> None:
    """Write analysis results to CSV files."""
    ts = analysis_timestamp()
    log_basename = os.path.basename(analysis.filepath)

    summary_row = {
        "timestamp": ts,
        "log_file": log_basename,
        "domain": DOMAIN_DISPATCH,
        "total_processes": analysis.total_processes,
        "tests_run": len(analysis.tests_run),
        "tests_passed": len(analysis.tests_passed),
        "tests_failed": len(analysis.tests_failed),
        "tests_skipped": len(analysis.tests_skipped),
        "warnings_count": len(analysis.warnings),
        "critical_errors_count": len(analysis.critical_errors),
        "status": analysis.status,
    }

    stem, suffix = csv_stem_and_suffix(csv_path, csv_prefix, DOMAIN_DISPATCH)

    write_csv([summary_row], f"{stem}_summary{suffix}", SUMMARY_CSV_FIELDS)

    # Detail CSV: one row per test result + one row per categorized error
    detail_rows: list[dict] = []

    for test in sorted(analysis.tests_passed):
        detail_rows.append(
            {
                "timestamp": ts,
                "log_file": log_basename,
                "domain": DOMAIN_DISPATCH,
                "record_type": "test_result",
                "test_name": test,
                "test_status": "PASSED",
                "error_category": "",
                "error_severity": "",
                "error_message": "",
            }
        )

    for test in sorted(analysis.tests_failed):
        detail_rows.append(
            {
                "timestamp": ts,
                "log_file": log_basename,
                "domain": DOMAIN_DISPATCH,
                "record_type": "test_result",
                "test_name": test,
                "test_status": "FAILED",
                "error_category": "test_failed",
                "error_severity": "error",
                "error_message": "",
            }
        )

    for test in sorted(analysis.tests_skipped):
        detail_rows.append(
            {
                "timestamp": ts,
                "log_file": log_basename,
                "domain": DOMAIN_DISPATCH,
                "record_type": "test_result",
                "test_name": test,
                "test_status": "SKIPPED",
                "error_category": "test_skipped",
                "error_severity": "info",
                "error_message": "",
            }
        )

    for cat_key, severity, msg in analysis.categorized_errors:
        detail_rows.append(
            {
                "timestamp": ts,
                "log_file": log_basename,
                "domain": DOMAIN_DISPATCH,
                "record_type": "error",
                "test_name": "",
                "test_status": "",
                "error_category": cat_key,
                "error_severity": severity,
                "error_message": msg[:500],
            }
        )

    if detail_rows:
        write_csv(detail_rows, f"{stem}_details{suffix}", DETAIL_CSV_FIELDS)


def main():
    parser = build_base_argparser("Analyze dispatch test logs and generate summary.")
    parser.add_argument("log_file", help="Dispatch test log file to analyze")
    args = parser.parse_args()
    apply_common_args(args)

    log_file = validate_file_exists(args.log_file)
    analysis = analyze_log_file(str(log_file))

    if args.csv:
        output_csv(analysis, args.csv, csv_prefix=args.csv_prefix)

    if args.json:
        output_json(analysis)
    else:
        output_text(analysis)

    if analysis.num_failed > 0 or analysis.num_passed == 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
