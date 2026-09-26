# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Summarise test_dram_read_bw.py: GB/s per case (bytes read / device kernel time, mean over iterations).

    python models/demos/mimo_v2_d_p/tests/perf/analyze_dram_read_bw.py <ops_perf_results.csv> [cases.jsonl] [--sort]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analyze_tags import load, summarize  # noqa: E402

PEAK = 512.0


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    stats = {}
    for line in open(args[1] if len(args) > 1 else "generated/mimo_dram_read/cases.jsonl"):
        s = json.loads(line)
        stats[s["tag"]] = s
    rows = []
    for tag, iters in load(args[0]).items():
        total, _ = summarize(iters)
        s = stats.get(tag)
        if not s or not total:
            continue
        gbps = s["bytes"] / (total * 1e3)
        rows.append((s["layout"], s["cores"], s["risc"], s["chunk"], s["assign"], total, gbps))
    if "--sort" in sys.argv:
        rows.sort(key=lambda r: -r[-1])
    print(
        f"{'layout':<14s} {'cores':>5s} {'risc':<7s} {'chunk':>6s} {'assign':<8s} {'time us':>9s} {'GB/s':>7s} {'% peak':>7s}"
    )
    for lay, c, r, ch, a, t, g in rows:
        print(f"{lay:<14s} {c:5d} {r:<7s} {ch:6d} {a:<8s} {t:9.1f} {g:7.1f} {100 * g / PEAK:6.1f}%")


if __name__ == "__main__":
    main()
