#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# filter_t3k_log.py — T3K CI log filter for focused analysis
#
# Usage:
#   python3 filter_t3k_log.py [options] <logfile>
#   python3 filter_t3k_log.py [options] -              # read from stdin
#
# Options:
#   --allgather, -a     Show only AllGather-related lines (CCL ops, hang
#                       indicators, GAP-2x/GAP-38 test output)
#   --errors, -e        Show only lines that indicate failures or errors
#   --summary, -s       Print a pass/fail summary per test batch
#   --context N, -c N   Show N lines of context around each match (default 0)
#
# With no filter options, prints LOG_METAL lines plus test pass/fail markers.
#
# Examples:
#   # AllGather-only view of a CI log:
#   python3 tests/scripts/t3000/filter_t3k_log.py --allgather ci_log.txt
#
#   # AllGather with 5 lines of context around each match:
#   python3 tests/scripts/t3000/filter_t3k_log.py -a -c 5 ci_log.txt
#
#   # Pipe from a download:
#   cat /workspace/group/ci_log_latest.txt | python3 tests/scripts/t3000/filter_t3k_log.py -a -
#
#   # Summary view of what passed/failed:
#   python3 tests/scripts/t3000/filter_t3k_log.py --summary ci_log.txt

import argparse
import re
import sys
from collections import deque
from typing import IO, Iterator

# ---------------------------------------------------------------------------
# Pattern sets
# ---------------------------------------------------------------------------

# Lines that are always shown in the default (no-filter) view
_DEFAULT_PATTERNS = [
    re.compile(r"LOG_METAL:", re.IGNORECASE),
    re.compile(r"\[  FAILED  \]"),
    re.compile(r"\[  PASSED  \]"),
    re.compile(r"FAILED.*test session starts", re.IGNORECASE),
    re.compile(r"= \d+ passed"),
    re.compile(r"= \d+ failed"),
    re.compile(r"exit=\d+"),
    re.compile(r"ERROR —"),
    re.compile(r"Timeout detected"),
]

# Lines matched by --errors
_ERROR_PATTERNS = [
    re.compile(r"\[  FAILED  \]"),
    re.compile(r"FAILED\b", re.IGNORECASE),
    re.compile(r"ERROR\b", re.IGNORECASE),
    re.compile(r"TT_FATAL", re.IGNORECASE),
    re.compile(r"SIGABRT|SIGKILL|SIGSEGV", re.IGNORECASE),
    re.compile(r"Timeout detected"),
    re.compile(r"exit=[1-9]\d*"),          # non-zero exit
    re.compile(r"core dumped"),
    re.compile(r"Assertion.*failed"),
]

# Lines matched by --allgather
_ALLGATHER_PATTERNS = [
    # Test names / binary names
    re.compile(r"all[_-]?gather", re.IGNORECASE),
    re.compile(r"AllGather", re.IGNORECASE),
    re.compile(r"AsyncExecutionWorksCQ0"),
    re.compile(r"test_ccl_multi_cq_multi_device"),
    re.compile(r"unit_tests_ttnn_ccl"),
    re.compile(r"repro_ccl_cq0_hang"),
    # GAP tests that are AllGather-specific (21, 22, 23, 25, 38)
    re.compile(r"test_gap2[1235]_"),
    re.compile(r"test_gap38_"),
    # Hardware hang signatures related to AllGather
    re.compile(r"0x880030060"),             # unsafe NOC address in allgather hang
    re.compile(r"enqueue_write_shards"),
    re.compile(r"dispatch_thread_pool"),
    re.compile(r"ttnn::all_gather"),
    re.compile(r"ccl.*allgather|allgather.*ccl", re.IGNORECASE),
    # Batch header
    re.compile(r"\[BATCH 1/3\]"),
    re.compile(r"AllGather tests"),
    # Pass/fail for CCL binaries
    re.compile(r"unit_tests_ttnn_ccl.*exit=", re.IGNORECASE),
    re.compile(r"test_ccl.*exit=", re.IGNORECASE),
    re.compile(r"repro_ccl.*exit=", re.IGNORECASE),
    re.compile(r"gap2[1235].*PASSED|FAILED.*gap2[1235]", re.IGNORECASE),
    re.compile(r"gap38.*PASSED|FAILED.*gap38", re.IGNORECASE),
]

# Lines matched by --summary (batch delimiters + final result lines)
_SUMMARY_PATTERNS = [
    re.compile(r"LOG_METAL:.*\[BATCH"),
    re.compile(r"LOG_METAL:.*Running run_t3000"),
    re.compile(r"LOG_METAL:.*seconds to complete"),
    re.compile(r"LOG_METAL:.*T3K topology"),
    re.compile(r"LOG_METAL:.*ERROR"),
    re.compile(r"= \d+ passed"),
    re.compile(r"= \d+ failed"),
    re.compile(r"exit=\d+"),
    re.compile(r"\[  FAILED  \].*\("),    # GTest FAILED with test name
    re.compile(r"FAILED in \d+"),
    re.compile(r"::test_.*PASSED|::test_.*FAILED", re.IGNORECASE),
]


def _matches(line: str, patterns: list) -> bool:
    return any(p.search(line) for p in patterns)


def _iter_lines(source: IO) -> Iterator[str]:
    for line in source:
        yield line.rstrip("\n")


def _print_with_context(lines_buf: list, match_idx: int, ctx: int, already_printed: set, out: IO):
    """Print lines[match_idx] plus ctx lines before/after, deduplicating."""
    start = max(0, match_idx - ctx)
    end = min(len(lines_buf) - 1, match_idx + ctx)
    for i in range(start, end + 1):
        if i not in already_printed:
            print(lines_buf[i], file=out)
            already_printed.add(i)


def run(args, out: IO = sys.stdout):
    # Determine active pattern set
    if args.allgather:
        active = _ALLGATHER_PATTERNS
        label = "AllGather"
    elif args.errors:
        active = _ERROR_PATTERNS
        label = "errors"
    elif args.summary:
        active = _SUMMARY_PATTERNS
        label = "summary"
    else:
        active = _DEFAULT_PATTERNS
        label = None

    ctx = args.context

    # Open source
    if args.logfile == "-":
        source = sys.stdin
    else:
        try:
            source = open(args.logfile, "r", encoding="utf-8", errors="replace")
        except FileNotFoundError:
            print(f"error: file not found: {args.logfile}", file=sys.stderr)
            sys.exit(1)

    if label:
        print(f"# filter: {label}  context: ±{ctx} lines", file=out)
        print("#" + "-" * 70, file=out)

    if ctx == 0:
        # Streaming mode — no need to buffer
        for line in _iter_lines(source):
            if _matches(line, active):
                print(line, file=out)
    else:
        # Buffer all lines for context window
        all_lines = list(_iter_lines(source))
        printed: set = set()
        for i, line in enumerate(all_lines):
            if _matches(line, active):
                _print_with_context(all_lines, i, ctx, printed, out)

    if args.logfile != "-":
        source.close()


def main():
    p = argparse.ArgumentParser(
        description="Filter T3K CI logs for focused analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # AllGather-only view:
  python3 tests/scripts/t3000/filter_t3k_log.py --allgather /workspace/group/ci_log_latest.txt

  # AllGather with 5 lines of context:
  python3 tests/scripts/t3000/filter_t3k_log.py -a -c 5 /workspace/group/ci_log_latest.txt

  # Read from stdin:
  cat ci_log.txt | python3 tests/scripts/t3000/filter_t3k_log.py -a -

  # Summary of what passed/failed:
  python3 tests/scripts/t3000/filter_t3k_log.py --summary /workspace/group/ci_log_latest.txt
""",
    )
    p.add_argument("logfile", help='Log file path, or "-" for stdin')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--allgather", "-a",
        action="store_true",
        help="Show only AllGather-related lines (CCL ops, hang indicators, GAP-21/22/23/25/38)",
    )
    mode.add_argument(
        "--errors", "-e",
        action="store_true",
        help="Show only error/failure lines (FAILED, TT_FATAL, non-zero exit, SIGABRT, etc.)",
    )
    mode.add_argument(
        "--summary", "-s",
        action="store_true",
        help="Show batch headers, timing, and pass/fail summary lines",
    )
    p.add_argument(
        "--context", "-c",
        type=int,
        default=0,
        metavar="N",
        help="Show N lines of context around each match (default: 0)",
    )
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
