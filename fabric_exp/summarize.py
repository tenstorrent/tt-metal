#!/usr/bin/env python3
"""Summarize test_tt_fabric bandwidth summary CSVs: GB/s and effective 1.35 GHz cycles/packet.

usage: summarize.py label=path.csv [label=path.csv ...]   (first label is the reference for deltas)
"""
import csv
import sys

AICLK = 1.35e9


def load(path):
    out = {}
    for r in csv.DictReader(open(path)):
        key = (r["test_name"], r["ntype"].replace("NOC_UNICAST_", "").lower(), int(r["packet_size"]))
        out[key] = (float(r["avg_bandwidth_gigabytes_per_s"]), AICLK / float(r["avg_packets_per_s"]))
    return out


def main():
    runs = [(a.split("=", 1)[0], load(a.split("=", 1)[1])) for a in sys.argv[1:]]
    keys = sorted({k for _, d in runs for k in d}, key=lambda k: (k[0], k[1], k[2]))
    hdr = f"{'test':24s} {'ntype':14s} {'size':>5s}"
    for label, _ in runs:
        hdr += f" | {label[:14]:>14s} GB/s  cyc/pkt"
        if label != runs[0][0]:
            hdr += "   d%"
    print(hdr)
    for k in keys:
        line = f"{k[0][:24]:24s} {k[1][:14]:14s} {k[2]:5d}"
        ref = runs[0][1].get(k)
        for label, d in runs:
            if k in d:
                gb, cyc = d[k]
                line += f" | {'':>14s} {gb:5.2f} {cyc:8.1f}"
                if label != runs[0][0]:
                    line += f" {100 * (gb / ref[0] - 1):+5.1f}" if ref else "      "
            else:
                line += f" | {'':>14s} {'-':>5s} {'-':>8s}" + ("      " if label != runs[0][0] else "")
        print(line)


if __name__ == "__main__":
    main()
