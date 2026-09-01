# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Per-case perf table from two craq-sim per-dispatch traces produced by test_qpool_perf_matrix
(run via run_qpool.sh perf-ab): baseline first (e.g. num_threads=1), config second (num_threads=4).

Per case: median clocks over the measured iterations on each side, delta% kernel-duration-native
(NEGATIVE = config is faster), and the speedup ratio. Summary line: geometric-mean speedup.

SIM CLOCKS — relative comparison on the same sim build only; halo+pool quiesce as one dispatch,
so per-case clocks are the combined halo+pool envelope. Real numbers come from the emulator.

Usage: python qpool_perf_report.py <baseline.tsv> <config.tsv> [baseline_name config_name]
"""

import csv
import math
import re
import statistics
import sys
from collections import defaultdict

LABEL_RE = re.compile(r"^case::(.+)::i\d+$")


def load_cases(path):
    """Returns {case: [clocks, ...]} from measured-iteration rows."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows or "nodeid" not in rows[0] or "clocks" not in rows[0]:
        raise SystemExit(f"{path}: need a per-dispatch trace with nodeid + clocks columns")
    cases = defaultdict(list)
    for r in rows:
        m = LABEL_RE.match(r["nodeid"])
        if m:
            cases[m.group(1)].append(int(r["clocks"]))
    if not cases:
        raise SystemExit(f"{path}: no case:: labeled rows found")
    return dict(cases)


def main(argv):
    if len(argv) < 3:
        raise SystemExit(__doc__)
    base_name = argv[3] if len(argv) > 3 else "baseline"
    cfg_name = argv[4] if len(argv) > 4 else "config"
    base = load_cases(argv[1])
    cfg = load_cases(argv[2])

    common = [c for c in base if c in cfg]
    missing = sorted(set(base) ^ set(cfg))
    if missing:
        print(f"WARNING: cases present on only one side (skipped): {missing}")

    print(f"\nPer-case clocks, {base_name} vs {cfg_name} (delta% NEGATIVE = {cfg_name} faster; SIM-relative only):")
    print(f"  {'case':<24} {base_name:>12} {cfg_name:>12} {'delta%':>9} {'speedup':>9}  iter spread")
    ratios = []
    for c in common:
        mb, mc = statistics.median(base[c]), statistics.median(cfg[c])
        delta = (mc - mb) / mb * 100.0
        ratios.append(mb / mc)
        spread = f"{sorted(base[c])} -> {sorted(cfg[c])}"
        print(f"  {c:<24} {mb:>12.0f} {mc:>12.0f} {delta:>+8.1f}% {mb / mc:>8.2f}x  {spread}")
    geomean = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
    print(f"\n  geomean speedup: {geomean:.2f}x over {len(common)} cases")
    slow = [c for c, r in zip(common, ratios) if r < 1.0]
    if slow:
        print(f"  cases where {cfg_name} is SLOWER: {slow}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
