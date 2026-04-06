#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Print a compact summary of homogenized sweep run results.

This script reads the final ``oprun_*.json`` files after ``run_collective_update.py``
has normalized shared metadata. The output is intended to give developers a quick
view of the same result set that will later appear in Superset.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter


def _find_oprun_files(results_dir: pathlib.Path) -> list[pathlib.Path]:
    return sorted(results_dir.glob("oprun_*.json"))


def _load_json(path: pathlib.Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _counter_to_lines(counter: Counter, *, limit: int | None = None) -> list[str]:
    items = counter.most_common(limit)
    return [f"- `{key}`: {value}" for key, value in items]


def _module_suite_key(test: dict) -> tuple[str, str]:
    module_name = str(test.get("filepath") or "unknown")
    suite_name = str(test.get("test_case_name") or "unknown")
    return module_name, suite_name


def _is_failure_status(status: str) -> bool:
    return status not in {"pass", "skipped", "xfail"}


def _format_failed_test(test: dict, run_card_type: str) -> str:
    status = str(test.get("status") or "unknown")
    full_test_name = str(test.get("full_test_name") or test.get("filepath") or "unknown")
    suite_name = str(test.get("test_case_name") or "unknown")
    card_type = str(test.get("card_type") or run_card_type or "unknown")
    message = test.get("error_message") or test.get("exception") or test.get("message")

    details = f"`{status}` `{full_test_name}`"
    details += f" suite=`{suite_name}` card=`{card_type}`"
    if message:
        details += f" error=`{str(message).strip()[:200]}`"
    return details


def build_stats(results_dir: pathlib.Path, run_type: str) -> dict:
    oprun_files = _find_oprun_files(results_dir)
    stats = {
        "results_dir": results_dir,
        "run_type": run_type,
        "oprun_files": oprun_files,
        "failed_tests": [],
        "total_tests": 0,
    }

    for path in oprun_files:
        data = _load_json(path)
        run_card_type = str(data.get("card_type") or "unknown")

        for test in data.get("tests", []):
            stats["total_tests"] += 1
            status = str(test.get("status") or "unknown")
            if _is_failure_status(status):
                stats["failed_tests"].append(_format_failed_test(test, run_card_type))

    return stats


def render_text(stats: dict, max_failures: int) -> str:
    failed_tests = stats["failed_tests"]
    lines = [
        "=" * 40,
        "Sweep Failed Tests",
        "=" * 40,
        f"Run type:     {stats['run_type']}",
        f"Results dir:  {stats['results_dir']}",
        f"Run files:    {len(stats['oprun_files'])}",
        f"Total tests:  {stats['total_tests']}",
        f"Failed tests: {len(failed_tests)}",
    ]
    if not failed_tests:
        lines.extend(["", "No failing tests found."])
        return "\n".join(lines) + "\n"

    lines.extend(["", "Failures:"])
    lines.extend([f"  - {entry}" for entry in failed_tests[:max_failures]])
    if len(failed_tests) > max_failures:
        lines.append("")
        lines.append(f"  ... and {len(failed_tests) - max_failures} more failures")
    return "\n".join(lines) + "\n"


def render_markdown(stats: dict, max_failures: int) -> str:
    failed_tests = stats["failed_tests"]
    lines = [
        "## Sweep Failed Tests",
        "",
        f"- `run_type`: `{stats['run_type']}`",
        f"- `results_dir`: `{stats['results_dir']}`",
        f"- `oprun_files`: `{len(stats['oprun_files'])}`",
        f"- `total_tests`: `{stats['total_tests']}`",
        f"- `failed_tests`: `{len(failed_tests)}`",
        "",
    ]
    if not failed_tests:
        lines.append("No failing tests found.")
        lines.append("")
        return "\n".join(lines)

    lines.append("### Failures")
    lines.extend([f"- {entry}" for entry in failed_tests[:max_failures]])
    if len(failed_tests) > max_failures:
        lines.append(f"- ... and {len(failed_tests) - max_failures} more failures")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Print a compact summary of homogenized sweep run results")
    parser.add_argument(
        "--results-dir",
        type=pathlib.Path,
        default=pathlib.Path(__file__).parent / "results_export",
        help="Directory containing oprun_*.json files",
    )
    parser.add_argument(
        "--run-type",
        type=str,
        default="nightly",
        help="Human-readable run type to print in the summary",
    )
    parser.add_argument(
        "--format",
        choices=("text", "markdown"),
        default="text",
        help="Output format",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=200,
        help="Maximum number of failed tests to print",
    )
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"No results directory found at {args.results_dir}", file=sys.stderr)
        return 0

    stats = build_stats(args.results_dir, args.run_type)
    if not stats["oprun_files"]:
        print(f"No oprun_*.json files found in {args.results_dir}", file=sys.stderr)
        return 0

    if args.format == "markdown":
        print(render_markdown(stats, args.max_failures))
    else:
        print(render_text(stats, args.max_failures), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
