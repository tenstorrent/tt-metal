# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Summarise test_dram_read_fwd.py: DRAM read GB/s and NoC-delivered GB/s per case.

    python models/demos/mimo_v2_d_p/tests/perf/analyze_dram_read_fwd.py <ops_perf_results.csv> [cases.jsonl]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analyze_tags import load, summarize  # noqa: E402


def main():
    stats = {}
    for line in open(sys.argv[2] if len(sys.argv) > 2 else "generated/mimo_dram_fwd/cases.jsonl"):
        s = json.loads(line)
        stats[s["tag"]] = s
    print(
        f"{'mode':<11s} {'chunk':>6s} {'half KB':>7s} {'recv':>5s} {'time us':>9s} {'read GB/s':>9s} {'% of 512':>8s} {'delivered GB/s':>14s}"
    )
    for tag, iters in load(sys.argv[1]).items():
        total, _ = summarize(iters)
        s = stats.get(tag)
        if not s or not total:
            continue
        rd = s["read_bytes"] / (total * 1e3)
        dl = s["delivered_bytes"] / (total * 1e3)
        recv = "" if s["receivers"] is None else str(s["receivers"])
        print(
            f"{s['mode']:<11s} {s['chunk']:6d} {s['half_kb']:7d} {recv:>5s} {total:9.1f} {rd:9.1f} {100 * rd / 512:7.1f}% {dl:14.1f}"
        )


if __name__ == "__main__":
    main()
