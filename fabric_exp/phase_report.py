#!/usr/bin/env python3
"""Summarize a phase-profile CSV (TT_FABRIC_PHASE_PROFILE_CSV) per test.

For each test and ERISC role (erisc 0 = TX, erisc 1 = RX in two-ERISC mode; D = the offload DRISC, CSV erisc 2,
in DRISC core cycles) it reports, over the cores that moved
at least half the packets of the busiest core: utilization split (busy / blocked / idle of the active window),
active ticks per packet, busy ticks per packet, blocked events per packet, and per-packet sub-phase ticks.

usage: phase_report.py <phase.csv> [summary.csv]   (summary.csv adds measured GB/s and 1.35 GHz cycles/packet)
"""
import csv
import sys
from collections import defaultdict

PHASES = [
    "tx_send",
    "tx_txq_spin",
    "tx_txq_prespin",
    "tx_completion",
    "tx_conn",
    "rx_decode_admit",
    "rx_forward",
    "rx_local",
    "rx_completion",
    "rx_ack",
]
EVENTS = ["tx_blk_credit", "tx_blk_txq", "rx_blk_admit", "rx_blk_trid"]


def short(test):
    return (
        test.replace("_num_links_1", "")
        .replace("_ntype_unicast_", "_")
        .replace("Multicast", "Mc")
        .replace("Unicast", "Uc")
        .replace("NeighborExchange", "NbrX")
        .replace("_size", "")
    )


def load_bw(path):
    bw = {}
    if not path:
        return bw
    for r in csv.DictReader(open(path)):
        key = (r["test_name"], r["ntype"].replace("NOC_UNICAST_", "").lower(), r["packet_size"])
        bw[key] = (float(r["avg_bandwidth_gigabytes_per_s"]), 1.35e9 / float(r["avg_packets_per_s"]))
    return bw


def bw_for(test, bw):
    for (name, ntype, size), v in bw.items():
        if test.startswith(name + "_size_" + size + "_ntype_unicast_" + ntype):
            return v
    return None


def main():
    rows = list(csv.DictReader(open(sys.argv[1])))
    bw = load_bw(sys.argv[2] if len(sys.argv) > 2 else None)
    by_test = defaultdict(list)
    for r in rows:
        by_test[r["test"]].append(r)
    phases_present = [p for p in PHASES if any(int(r[p + "_count"]) for r in rows)]
    hdr = f"{'test':34s} {'GB/s':>6s} {'cyc/p':>6s} | {'E':1s} {'n':>2s} {'busy%':>5s} {'blk%':>5s} {'idle%':>5s} {'act/p':>6s} {'busy/p':>6s}"
    hdr += "".join(f" {e[:12]:>12s}" for e in EVENTS) + "".join(f" {p[:14]:>14s}" for p in phases_present)
    print(hdr)
    for test, rs in by_test.items():
        v = bw_for(test, bw)
        for erisc in ("0", "1", "2"):
            sel = [r for r in rs if r["erisc"] == erisc]
            if not sel:
                continue
            top = max(int(r["pkts"]) for r in sel)
            sel = [r for r in sel if int(r["pkts"]) >= top / 2]
            agg = defaultdict(int)
            for r in sel:
                for k, val in r.items():
                    if k not in ("test", "device", "eth_channel", "direction", "erisc", "wall", "cycles"):
                        agg[k] += int(val)
            pk = agg["pkts"]
            active = agg["busy"] + agg["blocked"] + agg["idle"]
            if not pk or not active:  # TX queue counters only (TT_FABRIC_TXQ_COUNTERS): see txq_report.py
                continue
            line = f"{short(test)[:34]:34s} "
            line += f"{v[0]:6.2f} {v[1]:6.1f}" if v else f"{'':6s} {'':6s}"
            line += (
                f" | {'D' if erisc == '2' else erisc} {len(sel):2d} {100*agg['busy']/active:5.1f} {100*agg['blocked']/active:5.1f}"
                f" {100*agg['idle']/active:5.1f} {active/pk:6.1f} {agg['busy']/pk:6.1f}"
            )
            line += "".join(f" {agg[e]/pk:12.2f}" for e in EVENTS)
            for p in phases_present:
                c = agg[p + "_count"]
                line += f" {agg[p + '_ticks']/pk:8.1f}x{c/pk:4.2f}" if c else f" {'':>14s}"
            print(line)


if __name__ == "__main__":
    main()
