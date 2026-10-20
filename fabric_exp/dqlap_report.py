#!/usr/bin/env python3
"""§43: DRISC main-loop lap breakdown (dqlap_15226240_<suite>): cycles per packet per section, with each lap mark's own
cost (calibrated at DRISC start-up, timer_cal upper half) taken out. Over the DRISCs that handled >= half the
busiest one's packets. usage: dqlap_report.py [suite ...]"""
import csv, os, sys
from collections import defaultdict

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SECT = [
    ("tx_send", "poll+loop"),
    ("tx_txq_spin", "decode/admit"),
    ("tx_completion", "trid+hdr"),
    ("tx_conn", "forward"),
    ("rx_ack", "local"),
    ("tx_txq_prespin", "bookkeep"),
    ("tx_visit", "batch/idle"),
]
P = next((a[4:] for a in sys.argv[1:] if a.startswith("--p=")), "dqlap")
for s in [a for a in sys.argv[1:] if not a.startswith("--")] or ["mix", "4352"]:
    rows = [r for r in csv.DictReader(open(f"{R}/{P}_15226240_{s}_phase.csv")) if r["erisc"] == "2"]
    rate, unprof = {}, {}
    for lab, d in ((f"{P}_15226240_{s}", rate), (f"dqi_15226240_{s}", unprof)):
        for r in csv.DictReader(open(f"{R}/{lab}.csv")):
            key = (
                r["test_name"]
                if ("Mix" in r["test_name"] or "Pure" in r["test_name"])
                else f"{r['test_name']}_size_{r['packet_size']}_ntype_{r['ntype'].lower().replace('noc_', '')}"
            )
            d.setdefault(key, 1.35e9 / float(r["avg_packets_per_s"]))
    lap_cal = (int(rows[0]["timer_cal"]) >> 32) / 100
    by = defaultdict(list)
    for r in rows:
        by[r["test"]].append(r)
    print(f"\n{s}: lap mark cost {lap_cal:.1f} cycles, subtracted per mark; cycles per packet (marks per packet)")
    print(
        f"{'row':34s} {'v4':>6s} {'probed':>6s}"
        + "".join(f" {n:>13s}" for _, n in SECT)
        + f" {'sum':>6s} {'marks':>6s}"
    )
    for t, rs in by.items():
        top = max(int(r["pkts"]) for r in rs)
        sel = [r for r in rs if int(r["pkts"]) >= top / 2]
        pk = sum(int(r["pkts"]) for r in sel)
        key = next((k for k in rate if t.startswith(k)), t)
        line = f"{t[:34]:34s} {unprof.get(key, 0):6.1f} {rate.get(key, 0):6.1f}"
        total, marks = 0.0, 0.0
        for c, _ in SECT:
            n = sum(int(r[c + "_count"]) for r in sel)
            tk = sum(int(r[c + "_ticks"]) for r in sel)
            v = (tk - lap_cal * n) / pk
            total += v
            marks += n / pk
            line += f" {v:7.1f}x{n / pk:4.2f}"
        print(line + f" {total:6.1f} {marks:6.2f}")
