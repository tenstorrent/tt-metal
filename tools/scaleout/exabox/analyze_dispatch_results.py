#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Analyze dispatch test logs and generate summary with exit codes."""

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

MAX_ERROR_LINES = 30
MAX_SKIP_REASONS = 5


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
        """Disable colors (for --no-color flag)."""
        for attr in ["GREEN", "RED", "YELLOW", "BLUE", "CYAN", "BOLD", "NC"]:
            setattr(cls, attr, "")


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
    has_failures: bool = False


def analyze_log_file(filepath: str) -> LogAnalysis:
    """Analyze a single dispatch test log file."""
    result = LogAnalysis(filepath=filepath)

    try:
        with open(filepath, encoding="utf-8", errors="replace") as f:
            result.content = f.read()
    except (OSError, IOError, PermissionError) as e:
        print(f"Warning: Could not read {filepath}: {e}", file=sys.stderr)
        return result

    lines = result.content.split("\n")

    # Count MPI processes (GTest instances)
    result.total_processes = result.content.count("Running main() from gmock_main.cc")

    # Extract test names from different states
    # [ RUN      ] TestSuite.TestName
    for match in re.finditer(r"\[ RUN      \] (.+)", result.content):
        result.tests_run.add(match.group(1).strip())

    # [       OK ] TestSuite.TestName (123 ms)
    for match in re.finditer(r"\[       OK \] ([^\(]+)", result.content):
        test_name = match.group(1).strip()
        result.tests_passed.add(test_name)

    # [  FAILED  ] TestSuite.TestName (456 ms)
    for match in re.finditer(r"\[  FAILED  \] ([^\(]+)", result.content):
        test_name = match.group(1).strip()
        # Filter out summary lines like "[  FAILED  ] 2 tests, listed below:"
        if not re.match(r"\d+ tests?", test_name):
            result.tests_failed.add(test_name)

    # [  SKIPPED ] TestSuite.TestName
    for match in re.finditer(r"\[  SKIPPED \] ([^\(]+)", result.content):
        test_name = match.group(1).strip()
        # Filter out summary lines
        if not re.match(r"\d+ tests?", test_name):
            result.tests_skipped.add(test_name)

    result.has_failures = len(result.tests_failed) > 0

    # Extract failure details (lines around FAILED markers)
    in_failure = False
    failure_context = []
    for line in lines:
        if "[  FAILED  ]" in line:
            in_failure = True
            failure_context = [line]
        elif in_failure:
            if any(kw in line for kw in ["FAILED", "exception", "description", "Failure"]):
                failure_context.append(line)
            if len(failure_context) >= 5:  # Limit context lines
                result.failure_details.extend(failure_context)
                in_failure = False
                failure_context = []

    # Capture any remaining failure context if loop ended mid-collection
    if failure_context:
        result.failure_details.extend(failure_context)

    # Extract skip reasons
    for i, line in enumerate(lines):
        if "Skipped$" in line and i > 0:
            prev_line = lines[i - 1]
            if any(kw in prev_line for kw in ["This suite must be run with", "requires", "needs"]):
                # Extract the reason part
                reason_match = re.search(r"\| info.*\| (.+)", prev_line)
                if reason_match:
                    result.skip_reasons.append(reason_match.group(1).strip())
                else:
                    result.skip_reasons.append(prev_line.strip())

    # Extract warnings
    warning_pattern = re.compile(r"warning.*timed out|ARC core.*failed", re.IGNORECASE)
    for line in lines:
        if warning_pattern.search(line):
            result.warnings.append(line.strip())

    # Extract critical errors (actual crashes only, not test error cases)
    # TT_FATAL messages are often intentional error-handling tests, so only flag real crashes
    critical_pattern = re.compile(r"segmentation fault|core dumped|Signal: (Aborted|Segmentation)", re.IGNORECASE)
    for line in lines:
        if critical_pattern.search(line):
            result.critical_errors.append(line.strip())

    return result


def print_summary(analysis: LogAnalysis) -> None:
    """Print test results summary."""
    num_run = len(analysis.tests_run)
    num_passed = len(analysis.tests_passed)
    num_failed = len(analysis.tests_failed)
    num_skipped = len(analysis.tests_skipped)

    print("=" * 50)
    print("Dispatch Test Log Analyzer")
    print("=" * 50)
    print(f"Log file: {analysis.filepath}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50 + "\n")

    print("=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
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

    # Show passed tests
    if num_passed > 0:
        print("-" * 50)
        print(f"PASSED TESTS ({num_passed})")
        print("-" * 50)
        for i, test in enumerate(sorted(analysis.tests_passed), 1):
            print(f"{i:2d}. {test}")
        print()

    # Show failed tests
    if num_failed > 0:
        print("-" * 50)
        print(f"{Colors.RED}FAILED TESTS ({num_failed}){Colors.NC}")
        print("-" * 50)
        for i, test in enumerate(sorted(analysis.tests_failed), 1):
            print(f"{i:2d}. {test}")
        print()

        # Show failure details
        if analysis.failure_details:
            print("Failure Details:")
            for line in analysis.failure_details[:MAX_ERROR_LINES]:
                print(f"  {line}")
            if len(analysis.failure_details) > MAX_ERROR_LINES:
                print(f"  ... ({len(analysis.failure_details) - MAX_ERROR_LINES} more lines)")
            print()

    # Show skipped tests
    if num_skipped > 0:
        print("-" * 50)
        print(f"{Colors.YELLOW}SKIPPED TESTS ({num_skipped}){Colors.NC}")
        print("-" * 50)
        for i, test in enumerate(sorted(analysis.tests_skipped), 1):
            print(f"{i:2d}. {test}")
        print()

        # Show skip reasons
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
    has_warnings = len(analysis.warnings) > 0
    has_critical = len(analysis.critical_errors) > 0

    if not has_warnings and not has_critical:
        return

    print("-" * 50)
    print("WARNINGS & CRITICAL ERRORS")
    print("-" * 50)

    if has_critical:
        print(f"{Colors.RED}⚠️  CRITICAL ERRORS DETECTED:{Colors.NC}")
        # Deduplicate critical errors by removing timestamps
        error_counts = defaultdict(int)
        for error in analysis.critical_errors:
            # Remove timestamp patterns and MPI rank prefixes to group similar errors
            normalized = re.sub(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+", "<timestamp>", error)
            normalized = re.sub(r"\[1,\d+\]<std(out|err)>:", "", normalized)
            error_counts[normalized.strip()] += 1

        items_to_show = list(sorted(error_counts.items(), key=lambda x: -x[1]))[:10]

        for error, count in items_to_show:
            if count > 1:
                print(f"  [{count}x] {error}")
            else:
                print(f"  {error}")

        if len(error_counts) > 10:
            print(f"  ... ({len(error_counts) - 10} more unique error types)")
        print()

    if has_warnings:
        print(f"{Colors.YELLOW}⚠️  RUNTIME WARNINGS DETECTED:{Colors.NC}")
        # Deduplicate warnings by removing timestamps
        warning_counts = defaultdict(int)
        for warning in analysis.warnings:
            # Remove timestamp patterns and MPI rank prefixes to group similar warnings
            normalized = re.sub(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+", "<timestamp>", warning)
            normalized = re.sub(r"\[1,\d+\]<std(out|err)>:", "", normalized)
            warning_counts[normalized.strip()] += 1

        items_to_show = list(sorted(warning_counts.items(), key=lambda x: -x[1]))[:10]

        for warning, count in items_to_show:
            if count > 1:
                print(f"  [{count}x] {warning}")
            else:
                print(f"  {warning}")

        if len(warning_counts) > 10:
            print(f"  ... ({len(warning_counts) - 10} more unique warning types)")
        print()

    print("Note: These are runtime issues that occurred during test execution.")
    print("      They may or may not affect test results.")
    print()


def print_recommendations(analysis: LogAnalysis) -> None:
    """Print actionable recommendations."""
    num_failed = len(analysis.tests_failed)
    num_passed = len(analysis.tests_passed)
    has_critical = len(analysis.critical_errors) > 0
    has_warnings = len(analysis.warnings) > 0

    print("=" * 50)
    print("Recommendations")
    print("=" * 50 + "\n")

    recs = []

    if num_failed > 0:
        recs.append(
            f"• {num_failed} test(s) failed. Review failure details above and "
            "check for device communication or timing issues."
        )

    if has_critical:
        recs.append(
            "• Critical errors detected (segmentation faults, core dumps). " "Investigate hardware or driver issues."
        )

    if has_warnings:
        recs.append("• Runtime warnings detected. Review timeout messages and ARC core failures.")

    if num_failed == 0 and num_passed > 0:
        if has_critical or has_warnings:
            recs.append(f"{Colors.YELLOW}✓ All tests passed, but runtime warnings/errors were detected.{Colors.NC}")
            recs.append("  Review the WARNINGS & CRITICAL ERRORS section above.")
        else:
            recs.append(
                f"{Colors.GREEN}✓ All dispatch tests passed successfully. "
                f"System is functioning correctly.{Colors.NC}"
            )
    elif num_passed == 0:
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
        "timestamp": datetime.now().isoformat(),
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


def main():
    parser = argparse.ArgumentParser(
        description="Analyze dispatch test logs and generate summary.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("log_file", help="Dispatch test log file to analyze")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")

    args = parser.parse_args()

    if args.no_color or args.json:
        Colors.disable()

    # Verify log file exists
    log_file = Path(args.log_file)
    if not log_file.is_file():
        print(f"Error: Log file not found: {args.log_file}", file=sys.stderr)
        sys.exit(1)

    # Analyze log file
    analysis = analyze_log_file(str(log_file))

    # Output results
    if args.json:
        output_json(analysis)
    else:
        print_summary(analysis)
        print_test_details(analysis)
        print_warnings_and_errors(analysis)
        print_recommendations(analysis)

        # Final result
        print("-" * 50)
        print("FINAL TEST RESULT")
        print("-" * 50)

        num_failed = len(analysis.tests_failed)
        num_passed = len(analysis.tests_passed)

        if num_failed > 0:
            print(f"STATUS: {Colors.RED}FAILED{Colors.NC}")
            print("Some tests failed. See details above.")
            sys.exit(1)
        elif num_passed == 0:
            print(f"STATUS: {Colors.RED}FAILED (NO TESTS PASSED){Colors.NC}")
            print("No tests were successfully executed.")
            sys.exit(1)
        else:
            print(f"STATUS: {Colors.GREEN}PASSED{Colors.NC}")
            print("All tests passed successfully.")
            sys.exit(0)

    sys.exit(0)


if __name__ == "__main__":
    main()
