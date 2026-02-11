#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Analyze fabric test logs for pass/fail summary."""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


# Constants for limiting output
MAX_WARNINGS = 10
MAX_ERRORS = 10


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
    """Analysis results for fabric test log file."""

    filepath: str
    content: str = ""
    all_passed: bool = False
    warnings: list[str] = field(default_factory=list)
    critical_errors: list[str] = field(default_factory=list)


def analyze_log_file(filepath: str) -> LogAnalysis:
    """Analyze a fabric test log file."""
    result = LogAnalysis(filepath=filepath)

    try:
        with open(filepath, encoding="utf-8", errors="replace") as f:
            result.content = f.read()
    except (OSError, PermissionError) as e:
        print(f"Warning: Could not read {filepath}: {e}", file=sys.stderr)
        return result

    # Check if all hosts passed
    # Count unique MPI ranks to determine number of hosts
    mpi_ranks = set()
    for match in re.finditer(r"^\[1,(\d+)\]", result.content, re.MULTILINE):
        mpi_ranks.add(match.group(1))

    expected_hosts = len(mpi_ranks) if mpi_ranks else 1
    success_count = result.content.count("All tests completed successfully")
    result.all_passed = success_count == expected_hosts and expected_hosts > 0

    # Extract warnings (excluding FAILED markers)
    warning_pattern = re.compile(r"warning.*(timed out|failed)", re.IGNORECASE)
    failed_marker = re.compile(r"\[  FAILED  \]")
    for line in result.content.split("\n"):
        if warning_pattern.search(line) and not failed_marker.search(line):
            result.warnings.append(line.strip())

    # Extract critical errors (excluding noise)
    critical_pattern = re.compile(r"critical|TT_FATAL|segmentation fault|Signal: (Aborted|Segmentation)", re.IGNORECASE)
    exclude_pattern = re.compile(r"End of error message|Unknown error|MPI_ERRORS_ARE_FATAL")
    for line in result.content.split("\n"):
        if critical_pattern.search(line) and not exclude_pattern.search(line):
            result.critical_errors.append(line.strip())

    return result


def print_summary(analysis: LogAnalysis, input_path: str) -> None:
    """Print analysis summary."""
    print("=" * 50)
    print("Fabric Test Log Analyzer")
    print("=" * 50)
    print(f"Log file: {input_path}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50 + "\n")

    status_color = Colors.GREEN if analysis.all_passed else Colors.RED
    status_text = "PASSED" if analysis.all_passed else "FAILED"
    print(f"Test Status: {status_color}{status_text}{Colors.NC}\n")


def print_warnings_and_errors(analysis: LogAnalysis) -> None:
    """Print warnings and critical errors section."""
    if not analysis.warnings and not analysis.critical_errors:
        return

    print("=" * 50)
    print("WARNINGS & FAILURES")
    print("=" * 50)

    if analysis.critical_errors:
        print(f"\n{Colors.RED}-- Critical Errors --{Colors.NC}")
        for entry in analysis.critical_errors[:MAX_ERRORS]:  # Limit output
            print(f"  {entry}")

    if analysis.warnings:
        print(f"\n{Colors.YELLOW}-- Warnings/Runtime Errors --{Colors.NC}")
        for entry in analysis.warnings[:MAX_WARNINGS]:  # Limit output
            print(f"  {entry}")

    print("=" * 50 + "\n")


def print_recommendations(analysis: LogAnalysis) -> None:
    """Print actionable recommendations."""
    print("=" * 50)
    print("Recommendations")
    print("=" * 50 + "\n")

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
        "timestamp": datetime.now().isoformat(),
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

    # Final result
    print("-" * 50)
    print("FINAL TEST RESULT")
    print("-" * 50)

    if analysis.all_passed:
        print(f"{Colors.GREEN}✓ All fabric tests PASSED{Colors.NC}")
    else:
        print(f"{Colors.RED}✗ Fabric tests FAILED{Colors.NC}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze fabric test log for pass/fail summary.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("path", help="Fabric test log file to analyze")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")

    args = parser.parse_args()

    if args.no_color or args.json:
        Colors.disable()

    # Verify log file exists
    log_file = Path(args.path)
    if not log_file.is_file():
        print(f"Error: Log file not found: {args.path}", file=sys.stderr)
        sys.exit(1)

    # Analyze log file
    analysis = analyze_log_file(str(log_file))

    # Output results
    if args.json:
        output_json(analysis, args.path)
    else:
        output_text(analysis, args.path)
    if analysis.all_passed:
        exit(0)
    else:
        exit(1)


if __name__ == "__main__":
    main()
