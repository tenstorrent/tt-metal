#!/usr/bin/env python3
"""Compare two runs of the same yaml row by row: cycles/packet (1.35 GHz / packets per s) and GB/s.
usage: compare_runs.py OLD_PREFIX NEW_PREFIX"""
import csv, sys


def load(p):
    rows = {}
    for r in csv.DictReader(open(p + ".csv")):
        key = (r["test_name"], r["ntype"], r["num_links"], r["packet_size"])
        rows[key] = (1.35e9 / float(r["avg_packets_per_s"]), float(r["avg_bandwidth_gigabytes_per_s"]))
    return rows


old, new = load(sys.argv[1]), load(sys.argv[2])
print(f"{'test':30s} {'ntype':24s} links  old cyc/pkt  new cyc/pkt   change   old GB/s  new GB/s")
for k in old:
    if k not in new:
        continue
    (oc, og), (nc, ng) = old[k], new[k]
    print(
        f"{k[0][:30]:30s} {k[1][:24]:24s} {k[2]:>5}  {oc:11.1f}  {nc:11.1f}  {100*(nc-oc)/oc:+6.1f}%  {og:8.2f}  {ng:8.2f}"
    )
