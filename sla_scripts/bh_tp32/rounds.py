#!/usr/bin/env python3
"""Per-round TTFT of closed-loop vllm bench results (requests sorted by start; round = max_concurrency requests).
usage: rounds.py result.json [...]"""
import json
import statistics as st
import sys

for p in sys.argv[1:]:
    d = json.load(open(p))
    c = d.get("max_concurrency") or 32
    rows = sorted(zip(d["start_times"], d["ttfts"]))
    rs = [[t for _, t in rows[i : i + c]] for i in range(0, len(rows), c)]
    print(
        f"{p.rsplit('/', 1)[-1]:32s} mean {st.mean(d['ttfts']):.3f}  "
        + "  ".join(f"R{i} {st.mean(r):.3f}" for i, r in enumerate(rs))
        + f"  p99 {d['p99_ttft_ms'] / 1e3:.2f}  tpot {d['mean_tpot_ms']:.1f} p99 {d['p99_tpot_ms']:.1f}"
    )
