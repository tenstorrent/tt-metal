#!/usr/bin/env python3
"""lat_compare.py NAME=label ...: median net round trip (ns) per latency case (lat_runs.sh, RESULTS §19/§98).

The first build is the base; the rest are shown with their change against it. Per-sample CSVs are
results/<label>_13129088_lat_latency.csv (the harness's latency mode; net = round trip minus the responder's time).
"""
import csv, os, re, statistics, sys

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def lat(label):
    f = os.path.join(R, label + "_13129088_lat_latency.csv")
    if not os.path.exists(f):
        return None
    g = {}
    for r in csv.DictReader(open(f)):
        g.setdefault(re.sub(r"_sample\d+$", "", r["test_name"]), []).append(float(r["net_avg_ns"]))
    return {k: statistics.median(v) for k, v in g.items()}


B = [a.split("=", 1) for a in sys.argv[1:]]
L = {n: lat(l) for n, l in B}
base = B[0][0]
print("median net round trip, ns" + " " * 13 + "".join(f"{n:>14s}" for n, _ in B))
for k in L[base]:
    short = k.replace("_ntype_unicast_write_size", "").replace("_ntype_fused_atomic_inc_size", "_fused")
    cells = [f"{L[base][k]:14.0f}"] + [
        (f"{L[n][k]:8.0f} {L[n][k] - L[base][k]:+5.0f}" if L[n] and k in L[n] else f"{'-':>14s}") for n, _ in B[1:]]
    print(f"   {short:34s}" + "".join(cells))
