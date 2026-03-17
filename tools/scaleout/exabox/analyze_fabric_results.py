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
    error_category: str = "inconclusive"
    exit_code: int = EXIT_CODE_INCONCLUSIVE


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

    # Detect specific error categories
    if result.all_passed:
        result.exit_code = EXIT_CODE_PASSED
        result.error_category = "passed"
    else:
        # Check for specific error patterns (priority order)
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
            # Check if test is hanging - log ends with output format indicating test was still running
            # Pattern: [1,X]<stdout>: or [1,X]<stderr>: followed by timestamp and log level (| info | etc)
            lines = [line.strip() for line in result.content.split("\n") if line.strip()]
            if lines and re.match(
                r"^\[1,\d+\]<std(?:out|err)>:\s+\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+\s+\|", lines[-1]
            ):
                result.error_category = "test_hanging"
                result.exit_code = EXIT_CODE_TEST_HANGING
            else:
                result.error_category = "inconclusive"
                result.exit_code = EXIT_CODE_INCONCLUSIVE

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
    """Generate recommendations for passed tests."""
    return [f"{Colors.GREEN}✓ All fabric tests passed successfully. Cluster fabric is healthy.{Colors.NC}"]


@register_recommendation("mgd_error")
def _recommend_mgd_error(analysis: LogAnalysis) -> list[str]:
    """Generate recommendations for MGD topology mismatch."""
    return [
        f"{Colors.RED}• MGD topology mismatch detected.{Colors.NC}",
        "• Check host order in host variables against cabling descriptors.",
        "• Verify MGD file matches the physical topology.",
        "• If host ordering is correct, escalate missing connections issue.",
    ]


@register_recommendation("fw_init_failed")
def _recommend_fw_init_failed(analysis: LogAnalysis) -> list[str]:
    """Generate recommendations for firmware initialization failures."""
    return [
        f"{Colors.RED}• Firmware initialization failed.{Colors.NC}",
        "• Reset boards with: tt-smi -r",
        "• Wait 30 seconds and retry fabric test.",
        "• If issue persists after reset, escalate to SYSENG.",
    ]


@register_recommendation("fabric_router_sync_timeout")
def _recommend_fabric_router_sync_timeout(analysis: LogAnalysis) -> list[str]:
    """Generate recommendations for fabric router sync timeout."""
    return [
        f"{Colors.RED}• Fabric Router Sync timeout detected.{Colors.NC}",
        "• Check fabric connectivity and router status.",
        "• Try reset with: tt-smi -r",
        "• Escalate to SYSENG if issue persists.",
    ]


@register_recommendation("test_hanging")
def _recommend_test_hanging(analysis: LogAnalysis) -> list[str]:
    """Generate recommendations for hanging tests."""
    return [
        f"{Colors.YELLOW}• Test appears to be hanging (incomplete log).{Colors.NC}",
        "• Log shows test was running but never completed.",
        "• Test may have been killed or timed out.",
        "• Check if test process is still running.",
        "• Consider increasing timeout or investigating hang cause.",
    ]


@register_recommendation("noc_conflict")
def _recommend_noc_conflict(analysis: LogAnalysis) -> list[str]:
    """Generate recommendations for NOC address conflicts."""
    return [
        f"{Colors.RED}• NOC address conflict detected.{Colors.NC}",
        "• UMD detected mismatch between expected and actual NOC addresses.",
        "• This typically indicates memory mapping or firmware issue.",
        "• Try resetting boards with: tt-smi -r",
        "• If issue persists, escalate to SYSENG - may require firmware update.",
    ]


@register_recommendation("ethernet_core_timeout")
def _recommend_ethernet_core_timeout(analysis: LogAnalysis) -> list[str]:
    """Generate recommendations for ethernet core timeout."""
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
    """Generate recommendations for inconclusive results."""
    return [
        f"{Colors.YELLOW}• Fabric test failed with unrecognized error pattern.{Colors.NC}",
        "• Review critical errors and warnings above.",
        "• Check cable connections, port status, and fabric topology.",
        "• Report to SYSENG and SCALEOUT teams with full logs.",
    ]


def print_recommendations(analysis: LogAnalysis) -> None:
    """Print actionable recommendations."""
    print("=" * 50)
    print("Recommendations")
    print("=" * 50 + "\n")

    recs = []
    if analysis.error_category in RECOMMENDATION_GENERATORS:
        recs.extend(RECOMMENDATION_GENERATORS[analysis.error_category](analysis))

    # Add critical errors notice if present and test failed
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
    print(f"Test Status: {status_color}{status_text}{Colors.NC}")
    if not analysis.all_passed:
        print(f"Error Category: {analysis.error_category}")
        print(f"Exit Code: {analysis.exit_code}")
    print()


def output_json(analysis: LogAnalysis, input_path: str):
    """Output JSON results."""
    result = {
        "timestamp": datetime.now().isoformat(),
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
        sys.exit(EXIT_CODE_INPUT_ERROR)

    # Analyze log file
    analysis = analyze_log_file(str(log_file))

    # Output results
    if args.json:
        output_json(analysis, args.path)
    else:
        output_text(analysis, args.path)

    # Exit with specific code based on error category
    sys.exit(analysis.exit_code)


if __name__ == "__main__":
    main()
