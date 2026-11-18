#!/usr/bin/env python3
"""Summarize bench.sh result JSONs: TTFT, per-user decode speed, end-to-end. usage: summary.py <OUT dir>"""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1])  # the OUT dir of bench.sh
tags = [
    "single_4k_500",
    "cold_single_16k",
    "cold_single_32k",
    "rpm60_mix",
    "rpm60_prefix40",
    "rpm240_mix",
    "c32_4k_500",
]
print(
    f"{'test':18} {'done':>6} {'TTFT mean':>10} {'TTFT p99':>9} {'tok/s/u mean':>13} {'tok/s/u p99':>12} {'e2e mean':>9} {'e2e p99':>8} {'total tok/s':>11}"
)
for t in tags:
    f = d / f"{t}.json"
    if not f.exists():
        print(f"{t:18} (not run)")
        continue
    r = json.load(open(f))
    tpot_mean, tpot_p99 = r.get("mean_tpot_ms"), r.get("p99_tpot_ms")
    print(
        f"{t:18} {r.get('completed', 0):>3}/{r.get('num_prompts', r.get('completed', 0)):<3}"
        f" {r.get('mean_ttft_ms', 0) / 1000:>9.2f}s {r.get('p99_ttft_ms', 0) / 1000:>8.2f}s"
        f" {1000 / tpot_mean if tpot_mean else 0:>13.1f} {1000 / tpot_p99 if tpot_p99 else 0:>12.1f}"
        f" {r.get('mean_e2el_ms', 0) / 1000:>8.1f}s {r.get('p99_e2el_ms', 0) / 1000:>7.1f}s"
        f" {r.get('output_throughput', 0):>11.1f}"
    )
p = d / "prefix_hit_probe.log"
if p.exists():
    txt = p.read_text()
    i = txt.rfind("{")
    try:
        print("prefix_hit_probe:", json.loads(txt[txt.rfind("{", 0, i) if txt[i:].count("}") < 1 else i :]))
    except Exception:
        print("prefix_hit_probe (raw tail):", txt[-600:])
