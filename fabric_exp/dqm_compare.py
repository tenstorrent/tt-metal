#!/usr/bin/env python3
"""§39/§40: DRISC phase costs from one-phase-at-a-time runs (<prefix>_<phase>_15226240_mix), two prefixes side by
side: cycles per occurrence, net of the timer's inner cost, over the DRISCs that handled >= half the busiest one's
packets. usage: dqm_compare.py [old=dqm] [new=dqmh]"""
import csv, os, sys
from collections import defaultdict

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OLD, NEW = (sys.argv[1:3] + ["dqm", "dqmh"][len(sys.argv[1:3]) :])[:2]
PH = {
    "visit": "rx_visit",
    "admit": "rx_decode_admit",
    "local": "rx_local",
    "forward": "rx_forward",
    "completion": "rx_completion",
}


def phase(prefix, p):
    rows = [r for r in csv.DictReader(open(f"{R}/{prefix}_{p}_15226240_mix_phase.csv")) if r["erisc"] == "2"]
    inner = (int(rows[0]["timer_cal"]) & 0xFFFFFFFF) / 100
    by = defaultdict(list)
    for r in rows:
        by[r["test"]].append(r)
    out = {}
    for t, rs in by.items():
        top = max(int(r["pkts"]) for r in rs)
        sel = [r for r in rs if int(r["pkts"]) >= top / 2]
        pk = sum(int(r["pkts"]) for r in sel)
        c = sum(int(r[PH[p] + "_count"]) for r in sel)
        tk = sum(int(r[PH[p] + "_ticks"]) for r in sel)
        blk = sum(int(r["rx_blk_trid"]) for r in sel)
        out[t] = ((tk - inner * c) / c if c else None, c / pk, blk / pk)
    return out


old = {p: phase(OLD, p) for p in PH}
new = {p: phase(NEW, p) for p in PH}
print(
    f"cycles per occurrence, {OLD} -> {NEW} (occurrences per packet in the new run; "
    "forward shown only when >= 0.1 per packet)"
)
print(f"{'row':20s}" + "".join(f" {p:>17s}" for p in PH) + f" {'blk_trid/pkt':>14s}")
for t in new["visit"]:
    line = f"{t[:20]:20s}"
    for p in PH:
        a, b = old[p].get(t, (None, 0, 0)), new[p].get(t, (None, 0, 0))
        if b[0] is None or (p == "forward" and b[1] < 0.1):
            line += f" {'':>17s}"
        else:
            line += f" {a[0] if a[0] is not None else 0:6.1f}->{b[0]:6.1f}x{b[1]:3.1f}"
    line += f" {old['completion'].get(t,(0,0,0))[2]:5.2f}->{new['completion'].get(t,(0,0,0))[2]:5.2f}"
    print(line)
