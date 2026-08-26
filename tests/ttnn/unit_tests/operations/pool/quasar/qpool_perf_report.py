# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Summarize a craq-sim per-dispatch perf trace (ttsim_perf_trace.tsv) produced by
test_qpool_perf.py / run_qpool_perf.sh.

Rows are grouped by their nodeid label ("warmup", "iter0", "iter1", ...). Each measured
iteration contains the dispatches of one pool op call (halo + pool, plus any small helpers);
the POOL program is identified as the iteration's row with the most math-engine instructions
(the reduce dominates; halo's pack_untilize is comparatively tiny). Reports per-iteration and
average clocks for the pool program alone and for the whole op (sum of the iteration's rows).

Usage: python qpool_perf_report.py <path/to/ttsim_perf_trace.tsv>
"""

import csv
import sys
from collections import defaultdict


def main(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        print(f"QPOOL-PERF: no rows in {path}")
        return 1
    if "nodeid" not in rows[0] or "clocks" not in rows[0]:
        print(
            f"QPOOL-PERF: {path} lacks nodeid/clocks columns — need TTSIM_PERF_TRACE_NODEID_COLUMN=1 "
            f"and a craq-sim build with the clocks column (branch wransom/qsr-csr-timeout-count)"
        )
        return 1

    iters = defaultdict(list)
    for r in rows:
        label = r["nodeid"]
        if label.startswith("iter"):
            iters[label].append(r)

    if not iters:
        print("QPOOL-PERF: no measured iterations found (labels iter0..N missing from trace)")
        return 1

    pool_clocks, op_clocks = [], []
    print(
        f"{'iter':8} {'dispatches':>10} {'op_clocks':>10} {'pool_clocks':>11} " f"{'pool_math':>10} {'pool_stalls':>11}"
    )
    for label in sorted(iters, key=lambda s: int(s[4:])):
        rs = iters[label]
        total = sum(int(r["clocks"]) for r in rs)
        pool = max(rs, key=lambda r: int(r["math_instr"]))
        pool_clocks.append(int(pool["clocks"]))
        op_clocks.append(total)
        print(
            f"{label:8} {len(rs):>10} {total:>10} {pool['clocks']:>11} "
            f"{pool['math_instr']:>10} {pool['total_stalls']:>11}"
        )

    n = len(pool_clocks)
    print(f"\nQPOOL-PERF AVERAGE over {n} iterations (SIM CLOCKS — relative A/B only, not silicon):")
    print(f"  pool program : {sum(pool_clocks) / n:12.1f} clocks  (min {min(pool_clocks)}, max {max(pool_clocks)})")
    print(f"  whole op     : {sum(op_clocks) / n:12.1f} clocks  (min {min(op_clocks)}, max {max(op_clocks)})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
