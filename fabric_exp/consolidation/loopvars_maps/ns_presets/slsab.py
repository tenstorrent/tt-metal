#!/usr/bin/env python3
"""slsab.py PASSES ROUTERS: the sls_sl_runs.sh base-arm rows (results/sls<pass>sl_<router>_base_<2d|1d>_tests.csv),
cycles/packet = 1.35e9 / packets_per_second per test and size; per router the mean over the passes and the pass spread,
then Δ of each router against the first."""
import csv, os, sys, collections
R = "/home/aagarwal/tt-metal/fabric_exp/results"
passes = sys.argv[1].split(",")
routers = sys.argv[2].split(",")
vals = collections.defaultdict(dict)  # (kind, test, size) -> {(router, pass): cpp}
for p in passes:
    for r in routers:
        for kind in ("2d", "1d"):
            f = f"{R}/sls{p}sl_{r}_base_{kind}_tests.csv"
            if not os.path.exists(f):
                continue
            for row in csv.DictReader(open(f)):
                key = (kind, row["test_name"].split("_size_")[0], int(row["packet_size"]))
                pps = float(row["packets_per_second"])
                vals[key].setdefault((r, p), []).append(1.35e9 / pps)
hdr = "| row | size | " + " | ".join(f"{r} ({'/'.join(passes)})" for r in routers) + " | " + " | ".join(f"Δ {r}" for r in routers[1:]) + " |"
print(hdr)
print("| --- | ---: |" + " --- |" * len(routers) + " ---: |" * (len(routers) - 1))
for key in sorted(vals):
    kind, test, size = key
    means, cells = [], []
    for r in routers:
        per = [sum(vals[key][(r, p)]) / len(vals[key][(r, p)]) for p in passes if (r, p) in vals[key]]
        m = sum(per) / len(per) if per else float("nan")
        means.append(m)
        cells.append(" / ".join(f"{x:.1f}" for x in per))
    print(f"| {kind} {test} | {size} | " + " | ".join(cells) + " | " + " | ".join(f"{m - means[0]:+.1f}" for m in means[1:]) + " |")
