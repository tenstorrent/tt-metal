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

# Constants for limiting output verbosity
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

    @property
    def num_failed(self) -> int:
        """Number of failed tests."""
        return len(self.tests_failed)

    @property
    def num_passed(self) -> int:
        """Number of passed tests."""
        return len(self.tests_passed)

    @property
    def has_failures(self) -> bool:
        """Whether any tests failed."""
        return len(self.tests_failed) > 0

    @property
    def has_critical(self) -> bool:
        """Whether critical errors were detected."""
        return len(self.critical_errors) > 0

    @property
    def has_warnings(self) -> bool:
        """Whether warnings were detected."""
        return len(self.warnings) > 0


def analyze_log_file(filepath: str) -> LogAnalysis:
    """Analyze a single dispatch test log file."""
    result = LogAnalysis(filepath=filepath)

    try:
        with open(filepath, encoding="utf-8", errors="replace") as f:
            result.content = f.read()
    except (OSError, PermissionError) as e:
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


def deduplicate_and_count_messages(messages: list[str]) -> dict[str, int]:
    """Deduplicate messages by normalizing timestamps and MPI rank prefixes."""
    message_counts = defaultdict(int)
    for message in messages:
        # Remove timestamp patterns and MPI rank prefixes to group similar messages
        normalized = re.sub(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+", "<timestamp>", message)
        normalized = re.sub(r"\[1,\d+\]<std(out|err)>:", "", normalized)
        message_counts[normalized.strip()] += 1
    return message_counts


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


def print_warnings_and_errors(analysis: LogAnalysis) -> None:
    """Print warnings and critical errors section."""

    if not analysis.has_warnings and not analysis.has_critical:
        return

    print("-" * 50)
    print("WARNINGS & CRITICAL ERRORS")
    print("-" * 50)

    if analysis.has_critical:
        print_message_section(analysis.critical_errors, "⚠️  CRITICAL ERRORS DETECTED:", Colors.RED)

    if analysis.has_warnings:
        print_message_section(analysis.warnings, "⚠️  RUNTIME WARNINGS DETECTED:", Colors.YELLOW)

    print("Note: These are runtime issues that occurred during test execution.")
    print("      They may or may not affect test results.")
    print()


def print_recommendations(analysis: LogAnalysis) -> None:
    """Print actionable recommendations."""

    print("=" * 50)
    print("Recommendations")
    print("=" * 50 + "\n")

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


def output_text(analysis: LogAnalysis):
    print_summary(analysis)
    print_test_details(analysis)
    print_warnings_and_errors(analysis)
    print_recommendations(analysis)

    # Final result
    print("-" * 50)
    print("FINAL TEST RESULT")
    print("-" * 50)

    if analysis.num_failed > 0:
        print(f"STATUS: {Colors.RED}FAILED{Colors.NC}")
        print("Some tests failed. See details above.")
    elif analysis.num_passed == 0:
        print(f"STATUS: {Colors.RED}FAILED (NO TESTS PASSED){Colors.NC}")
        print("No tests were successfully executed.")
    else:
        print(f"STATUS: {Colors.GREEN}PASSED{Colors.NC}")
        print("All tests passed successfully.")


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
        output_text(analysis)

    if analysis.num_failed > 0:
        sys.exit(1)
    elif analysis.num_passed == 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
