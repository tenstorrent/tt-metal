#!/usr/bin/env python3
"""Ethernet TX queue counters per test from a phase-profile CSV (TT_FABRIC_TXQ_COUNTERS=1).

Each ERISC samples the TX queues it issues commands on (in two-ERISC mode: ERISC 0 the data queue, ERISC 1 the
queue that carries receiver acks). Counters are the change over one test; configuration registers are as read.

usage: txq_report.py <phase.csv> [summary.csv]   (summary.csv adds the measured cycles per packet)
"""
import csv
import sys

from phase_report import bw_for, load_bw, short

COUNTERS = ["transfers", "pkts_started", "pkts_sent", "words", "resends"]
CONFIG = [
    "max_pkt_size",
    "burst_len",
    "accept_ahead",
    "ctrl",
    "min_pkt_words",
    "remote_seq_timeout",
    "local_seq_update_timeout",
]


def main():
    rows = list(csv.DictReader(open(sys.argv[1])))
    bw = load_bw(sys.argv[2] if len(sys.argv) > 2 else None)
    print(
        f"{'test':34s} {'cyc/p':>6s} {'device':>8s} {'ch':>3s} E q {'transfers':>9s} {'eth pkts':>9s} {'/xfer':>5s} "
        f"{'B/xfer':>7s} {'resends':>7s}  config"
    )
    for r in rows:
        for q in range(3):
            pre = f"txq{q}_"
            if pre + "transfers" not in r or not int(r[pre + "transfers"] or 0):
                continue
            c = {k: int(r[pre + k]) for k in COUNTERS + CONFIG}
            v = bw_for(r["test"], bw)
            cfg = " ".join(f"{k}={c[k]:#x}" if k == "ctrl" else f"{k}={c[k]}" for k in CONFIG)
            print(
                f"{short(r['test'])[:34]:34s} {v[1] if v else float('nan'):6.1f} {r['device'][-7:-1]:>8s} "
                f"{r['eth_channel']:>3s} {r['erisc']} {q} {c['transfers']:9d} {c['pkts_started']:9d} "
                f"{c['pkts_started'] / c['transfers']:5.2f} {16 * c['words'] / c['transfers']:7.0f} "
                f"{c['resends']:7d}  {cfg}"
            )


if __name__ == "__main__":
    main()
