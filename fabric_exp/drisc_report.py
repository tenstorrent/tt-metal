#!/usr/bin/env python3
"""DRISC rows (CSV erisc 2) of a phase-profile CSV, per test: over the DRISCs that handled at least half the packets
of the busiest one, the utilization split, active cycles per packet, blocked events per packet, and per-packet phase
cycles with the timer's own cost (timer_cal, inner part) taken out of each timed occurrence.

usage: drisc_report.py <phase.csv> [summary.csv]   (summary.csv adds measured cycles/packet at 1.35 GHz)"""
import csv, sys
from collections import defaultdict

PHASES = ["rx_visit", "rx_decode_admit", "rx_forward", "rx_local", "rx_completion"]


def cyc_per_pkt(path):
    out = {}
    if path:
        for r in csv.DictReader(open(path)):
            key = r["test_name"] + "_size_" + r["packet_size"] + "_ntype_" + r["ntype"].lower().replace("noc_", "")
            out.setdefault(key, 1.35e9 / float(r["avg_packets_per_s"]))
    return out


rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["erisc"] == "2"]
bw = cyc_per_pkt(sys.argv[2] if len(sys.argv) > 2 else None)
by_test = defaultdict(list)
for r in rows:
    by_test[r["test"]].append(r)
print(
    f"{'test':44s} {'cyc/p':>6s} {'n':>2s} {'busy%':>5s} {'blk%':>5s} {'idle%':>5s} {'act/p':>6s} {'blkadm':>6s} {'blktrd':>6s}"
    + "".join(f" {p[3:]:>14s}" for p in PHASES)
    + "  rest"
)
for test, rs in by_test.items():
    top = max(int(r["pkts"]) for r in rs)
    sel = [r for r in rs if int(r["pkts"]) >= top / 2]
    agg = defaultdict(int)
    for r in sel:
        for k in ["pkts", "busy", "blocked", "idle", "rx_blk_admit", "rx_blk_trid"] + [
            p + s for p in PHASES for s in ("_ticks", "_count")
        ]:
            agg[k] += int(r[k])
    inner = (int(sel[0]["timer_cal"]) & 0xFFFFFFFF) / 100.0
    pk, act = agg["pkts"], agg["busy"] + agg["blocked"] + agg["idle"]
    cp = next((v for k, v in bw.items() if test.startswith(k)), None)
    line = f"{test[:44]:44s} {cp if cp else 0:6.1f} {len(sel):2d}"
    line += f" {100*agg['busy']/act if act else 0:5.1f} {100*agg['blocked']/act if act else 0:5.1f} {100*agg['idle']/act if act else 0:5.1f} {act/pk:6.1f}"
    line += f" {agg['rx_blk_admit']/pk:6.2f} {agg['rx_blk_trid']/pk:6.2f}"
    net = {}
    for p in PHASES:
        c = agg[p + "_count"]
        net[p] = (agg[p + "_ticks"] - inner * c) / pk if agg[p + "_ticks"] else 0.0
        line += f" {net[p]:7.1f}x{c/pk:5.2f}" if c else f" {'':>14s}"
    if net["rx_visit"]:
        line += f"  {net['rx_visit'] - net['rx_decode_admit'] - net['rx_forward'] - net['rx_local']:5.1f}"
    print(line)
print(f"timer_cal inner (cycles one empty timed open/close shows inside a phase): {inner:.1f}")
