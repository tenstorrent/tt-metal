#!/usr/bin/env python3
"""
Read JIT compile stats log files (one file per process; format: timestamp  kernel_name  recompiled=N  cache_hit=M)
and categorize kernels by how many processes compiled them.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from collections import defaultdict


LOG_LINE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s{2}(.+?)\s{2}recompiled=(\d+)\s+cache_hit=(\d+)\s*$")


def parse_line(line: str) -> tuple[str, str, int, int] | None:
    """Return (timestamp, kernel_name, recompiled, cache_hit) or None if not a stats line."""
    line = line.rstrip("\n")
    m = LOG_LINE_RE.match(line)
    if not m:
        return None
    return m.group(1), m.group(2).strip(), int(m.group(3)), int(m.group(4))


def kernels_per_process(paths: list[Path]) -> tuple[list[set[str]], dict[str, int]]:
    """
    For each log file (process), collect the set of kernel names that appear.
    Return (list of per-file kernel sets), and kernel -> number of processes that compiled it.
    """
    per_process_kernels: list[set[str]] = []
    kernel_process_count: dict[str, int] = defaultdict(int)

    for p in paths:
        try:
            text = p.read_text()
        except Exception as e:
            print(f"Warning: could not read {p}: {e}", file=sys.stderr)
            continue
        kernels_in_file: set[str] = set()
        for line in text.splitlines():
            r = parse_line(line)
            if r is not None:
                k = r[1]
                kernels_in_file.add(k)
        per_process_kernels.append(kernels_in_file)
        for k in kernels_in_file:
            kernel_process_count[k] += 1

    return per_process_kernels, dict(kernel_process_count)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Categorize kernels by how many processes compiled them (one log file = one process)."
    )
    parser.add_argument(
        "logs",
        nargs="+",
        type=Path,
        help="Log files (each = one process) or directories to scan for *jit_compile_stats*.log",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only print summary and category counts, not kernel names per category",
    )
    args = parser.parse_args()

    paths: list[Path] = []
    for p in args.logs:
        if p.is_file():
            paths.append(p)
        elif p.is_dir():
            paths.extend(sorted(p.rglob("*jit_compile_stats*.log")))
        else:
            print(f"Warning: not found or not file/dir: {p}", file=sys.stderr)
    paths = sorted(set(paths))

    if not paths:
        print("No log files found.", file=sys.stderr)
        sys.exit(1)

    per_process_kernels, kernel_process_count = kernels_per_process(paths)
    if not kernel_process_count:
        print("No valid stats lines in the given logs.", file=sys.stderr)
        sys.exit(1)

    n_processes = len(per_process_kernels)
    # Categorize: by number of processes that compiled the kernel
    by_count: dict[int, list[str]] = defaultdict(list)
    for kernel, count in kernel_process_count.items():
        by_count[count].append(kernel)

    print("=== Kernels by number of processes that compiled them ===\n")
    print(f"Log files (processes): {n_processes}")
    print(f"Unique kernels:        {len(kernel_process_count)}")
    print()

    for n in sorted(by_count.keys(), reverse=True):
        kernels = sorted(by_count[n])
        label = "process" if n == 1 else "processes"
        print(f"--- Compiled by {n} {label} ({len(kernels)} kernel(s)) ---")
        if args.quiet:
            print(f"  (use without -q to list names)")
        else:
            for k in kernels:
                print(f"  {k!r}")
        print()

    # Summary distribution
    print("--- Summary: kernel count by # of processes ---")
    for n in sorted(by_count.keys()):
        print(f"  {n} process(es): {len(by_count[n])} kernel(s)")


if __name__ == "__main__":
    main()
