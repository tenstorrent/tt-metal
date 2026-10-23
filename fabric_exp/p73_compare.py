#!/usr/bin/env python3
"""§73: E0's loop split into segments (variant e0seg, p73_runs.sh). One timed segment per run; its profiler copy counts
only the timed phase, so E0 is the TX ERISC with the most packets sent (pkts). Net cycles per packet =
(ticks - cal x occurrences) / packets, cal = the in-situ timer self-cost (~4.0). Also prints the unprofiled interval
of e0seg next to §72's (job 184, same code at level 0) and the profiled interval of each run."""
import csv, os

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SEG = [
    (16, "rx_decode_admit", "loop top (grouped read issue)"),
    (32, "rx_forward", "worker: can-send decision"),
    (1, "tx_send", "worker: send"),
    (64, "rx_local", "worker: completion step (all)"),
    (4, "tx_completion", "  of which credit returns"),
    (128, "rx_completion", "worker: connection check"),
    (8, "tx_conn", "  of which check_worker_connections"),
    (2048, "rx_visit", "worker visit (whole)"),
    (256, "rx_ack", "forwarding channels 1-3"),
    (512, "tx_txq_prespin", "whole iteration body"),
    (1024, "tx_visit", "all channel visits"),
]


def key(test):
    if "_ntype_" in test:
        base, nt = test.split("_size")[0], test.split("_ntype_")[1].split("_num_links")[0]
    else:
        base, nt = test, ""
    return base + (" scatter" if "scatter" in nt else "") if base.startswith("Mesh") else base


def cyc(label):
    f = os.path.join(R, label + ".csv")
    if not os.path.exists(f):
        return {}
    out = {}
    for r in csv.DictReader(open(f)):
        k = r["test_name"] + (" scatter" if "SCATTER" in r["ntype"] and r["test_name"].startswith("Mesh") else "")
        out[k] = 1.35e9 / float(r["avg_packets_per_s"])
    return out


def seg(label, ph):
    f = os.path.join(R, label + "_phase.csv")
    if not os.path.exists(f):
        return {}
    rows = [r for r in csv.DictReader(open(f)) if r["erisc"] == "0"]
    out = {}
    for t in {r["test"] for r in rows}:
        r = max((r for r in rows if r["test"] == t), key=lambda r: int(r["pkts"] or 0))
        pk = int(r["pkts"] or 0)
        if pk < 1000:
            continue
        cal = (int(r["timer_cal"]) & 0xFFFFFFFF) / 100 if r.get("timer_cal") else 4.0
        ticks, cnt = int(r[f"{ph}_ticks"] or 0), int(r[f"{ph}_count"] or 0)
        out[key(t)] = ((ticks - cal * cnt) / pk, cnt / pk, int(r["iters"] or 0) / pk, cal)
    return out


for n in ("p72", "4352"):
    base, old = cyc(f"p73_{n}_noprof"), cyc(f"p72_{n}_noprof")
    if not base:
        continue
    d = {m: seg(f"p73_{n}_m{m}", ph) for m, ph, _ in SEG}
    prof = {m: cyc(f"p73_{n}_m{m}") for m, _, _ in SEG}
    tests = [t for t in base if any(t in d[m] for m, _, _ in SEG)]
    print(f"== {n}: unprofiled cycles/packet, e0seg (§72 job 184)")
    for t in base:
        print(f"   {t[:44]:44s}{base[t]:8.1f} ({old.get(t, float('nan')):.1f})")
    for t in tests:
        print(f"\n   {t}: E0 net cycles per packet (occurrences per packet) | profiled interval of that run")
        for m, ph, name in SEG:
            if t in d[m]:
                v, c, it, cal = d[m][t]
                print(
                    f"      {name:36s}{v:8.1f} ({c:4.2f})   | {prof[m].get(t, float('nan')):6.1f}  iters/pkt {it:4.2f} cal {cal:.2f}"
                )
    print()
