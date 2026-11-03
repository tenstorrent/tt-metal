#!/usr/bin/env python3
"""lat_ab_report.py PASS... [--base ROUTER] [--routers R1,R2,...]: median net round trip (ns) per latency case,
per router, mean over the given passes (lat_ab_runs.sh labels lat<PASS>_<router>), each pass in brackets, and each
router's difference against the base (default main_fw).

Net round trip = round trip minus the responder's time (the harness's latency mode, RESULTS §19).
"""
import argparse
import csv
import os
import re
import statistics

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def medians(label):
    f = os.path.join(R, f"{label}_latency.csv")
    if not os.path.exists(f):
        return None
    g = {}
    for row in csv.DictReader(open(f)):
        g.setdefault(re.sub(r"_sample\d+$", "", row["test_name"]), []).append(float(row["net_avg_ns"]))
    return {k: statistics.median(v) for k, v in g.items()}


ap = argparse.ArgumentParser()
ap.add_argument("passes", nargs="+")
ap.add_argument("--base", default="main_fw")
ap.add_argument("--routers", default="main_fw,main,v4rxg3")
a = ap.parse_args()
routers = a.routers.split(",")
data = {r: [m for m in (medians(f"lat{p}_{r}") for p in a.passes) if m] for r in routers}
cases = sorted({k for r in routers for m in data[r] for k in m})
print(f"median net round trip (ns), mean of passes {' '.join(a.passes)}; difference against {a.base}")
print(f"{'case':40s}" + "".join(f"{r:>30s}" for r in routers))
for k in cases:
    base = [m[k] for m in data[a.base] if k in m]
    bmean = sum(base) / len(base) if base else None
    cells = []
    for r in routers:
        vals = [m[k] for m in data[r] if k in m]
        if not vals:
            cells.append(f"{'-':>30s}")
            continue
        mean = sum(vals) / len(vals)
        each = "/".join(f"{v:.0f}" for v in vals)
        diff = "" if r == a.base or bmean is None else f" {mean - bmean:+6.1f}"
        cells.append(f"{mean:9.1f} ({each}){diff}".rjust(30))
    short = k.replace("_ntype_unicast_write_size", "").replace("_ntype_fused_atomic_inc_size", "_fused")
    print(f"{short:40s}" + "".join(cells))
