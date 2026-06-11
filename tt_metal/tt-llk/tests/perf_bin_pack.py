#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Slice-level greedy bin-pack for the Blackhole LLK perf shards.

Mirrors how pytest-split distributes items across shards, but at slice
granularity: files larger than an adaptive target are cut into K slices
(no upper cap on K) so an oversized test overflows into multiple
slices/bins instead of pinning a single bin. Each emitted slice becomes
its own pytest invocation (with its own preceding `tt-smi -r 0`) in
run_llk_perf_blackhole.sh.

Usage: perf_bin_pack.py <group> <n_groups>

Prints this group's slices, one per line, as
file:group_idx:total_splits:items. Run from the directory containing the
perf_*.py files (collection is CPU only -- no chip access).
"""

import glob
import math
import re
import subprocess
import sys


def collect_counts(files):
    """Return {file: collected perf-test count} via `pytest --collect-only`."""
    counts = {}
    for f in files:
        out = subprocess.run(
            ["pytest", "--collect-only", "-q", "-m", "perf", f],
            capture_output=True,
            text=True,
        ).stdout
        n = 0
        for line in out.splitlines():
            m = re.match(r"^\s*(\d+) tests? collected", line)
            if m:
                n = int(m.group(1))
                break
        counts[f] = n
    return counts


def assign(counts, group, n_groups):
    """Bin-pack sliced files into n_groups bins; return this group's slices.

    Returns a sorted list of (file, group_idx, total_splits, items) tuples
    for the requested 1-based group.
    """
    # Adaptive target: half of an ideal bin, floored at 1500. Scales with
    # suite size so it auto-adapts as tests are added or grow.
    total = sum(counts.values())
    target = max(1500, total // (2 * n_groups))

    # Slice each file into K parts (no upper cap on K).
    slices = []  # (file, group_idx, total_splits, items)
    for f, c in counts.items():
        K = max(1, math.ceil(c / target)) if c > 0 else 1
        base = c // K if K > 0 else 0
        rem = c - base * K
        # Even split; the rem extra items spill into the earliest slices.
        for i in range(K):
            slice_items = base + (1 if i < rem else 0)
            slices.append((f, i + 1, K, slice_items))

    # Greedy bin-pack slices into n_groups bins (first-fit-decreasing).
    bins = [[0, []] for _ in range(n_groups)]
    for s in sorted(slices, key=lambda x: -x[3]):
        bins.sort(key=lambda b: (b[0], len(b[1])))  # lightest bin first
        bins[0][0] += s[3]
        bins[0][1].append(s)

    return sorted(bins[group - 1][1])


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: perf_bin_pack.py <group> <n_groups>")
    group = int(sys.argv[1])
    n_groups = int(sys.argv[2])

    files = sorted(glob.glob("perf_*.py"))
    counts = collect_counts(files)
    for f, idx, total_splits, items in assign(counts, group, n_groups):
        print(f"{f}:{idx}:{total_splits}:{items}")


if __name__ == "__main__":
    main()
