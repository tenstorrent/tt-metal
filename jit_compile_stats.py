#!/usr/bin/env python3
"""
Read JIT compile stats log files (format: timestamp  kernel_name  recompiled=N  cache_hit=M)
and report cache hit rate and per-kernel recompilation stats (min, max, avg, distribution).
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


def load_logs(paths: list[Path]) -> list[tuple[str, str, int, int]]:
    rows: list[tuple[str, str, int, int]] = []
    for p in paths:
        try:
            text = p.read_text()
        except Exception as e:
            print(f"Warning: could not read {p}: {e}", file=sys.stderr)
            continue
        for line in text.splitlines():
            r = parse_line(line)
            if r is not None:
                rows.append(r)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute JIT compile stats from log files (timestamp  kernel  recompiled=N  cache_hit=M)."
    )
    parser.add_argument(
        "logs",
        nargs="+",
        type=Path,
        help="Log files or directories to scan (directories are scanned for *jit_compile_stats*.log by default)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only print summary, no per-kernel table",
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

    rows = load_logs(paths)
    if not rows:
        print("No valid stats lines in the given logs.", file=sys.stderr)
        sys.exit(1)

    total_recompiled = sum(r[2] for r in rows)
    total_cache_hit = sum(r[3] for r in rows)
    total_objs = total_recompiled + total_cache_hit
    hit_rate = (total_cache_hit / total_objs * 100) if total_objs else 0.0

    by_kernel: dict[str, list[int]] = defaultdict(list)
    for _ts, kernel, recomp, _ch in rows:
        by_kernel[kernel].append(recomp)
    # Sum of recompilations per kernel (each list is one build of that kernel)
    recomp_sum_by_kernel = {k: sum(v) for k, v in by_kernel.items()}
    build_count_by_kernel = {k: len(v) for k, v in by_kernel.items()}

    print("=== JIT compile stats ===\n")
    print(f"Log files: {len(paths)}")
    print(f"Lines:    {len(rows)}")
    print()
    print("--- Cache hit rate ---")
    print(f"  Total recompiled: {total_recompiled}")
    print(f"  Total cache hit:  {total_cache_hit}")
    print(f"  Cache hit rate:   {hit_rate:.1f}%")
    print()

    # Per-kernel: sum of recompilations (over all builds of that kernel)
    sums = list(recomp_sum_by_kernel.values())
    if not sums:
        sys.exit(0)

    print("--- Recompilations by kernel (sum over all builds of that kernel) ---")
    print(f"  Kernels: {len(recomp_sum_by_kernel)}")
    print(f"  Min:     {min(sums)}")
    print(f"  Max:     {max(sums)}")
    print(f"  Avg:     {sum(sums) / len(sums):.1f}")
    print()

    # Distribution: histogram of sum values
    dist: dict[int, int] = defaultdict(int)
    for s in sums:
        dist[s] += 1
    print("  Distribution (sum recompilations -> number of kernels):")
    for k in sorted(dist.keys()):
        print(f"    {k:>6}: {dist[k]} kernel(s)")
    print()

    if not args.quiet:
        print("--- Per-kernel table (kernel name, build count, sum recompiled) ---")
        for name in sorted(recomp_sum_by_kernel.keys(), key=lambda k: (-recomp_sum_by_kernel[k], k)):
            n_builds = build_count_by_kernel[name]
            s = recomp_sum_by_kernel[name]
            print(f"  {name!r}  builds={n_builds}  sum_recompiled={s}")
    print()


if __name__ == "__main__":
    main()
