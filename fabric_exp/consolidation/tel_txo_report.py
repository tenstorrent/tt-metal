#!/usr/bin/env python3
"""tel_txo_report.py ROUTER...: the perf recorder's per-core totals (packets_sent, eth_words_sent) of tel_txo_runs.sh,
per test and core, main's router (v1) against each ROUTER, two ERISCs and one; and the gtests' results."""
import re, sys, collections
R = "/home/aagarwal/tt-metal/fabric_exp/results"
LINE = re.compile(r"Telemetry from (.+?) core (\S+): .*eth_words_sent=(\d+), packets_sent=(\d+)")
TEST = re.compile(r"Running Test[^:]*:\s*(\S+)|Test (\S+) ")
routers = ["v1"] + sys.argv[1:]
for m in ("two", "one"):
    for s in ("4352", "1d"):
        per = {}
        for r in routers:
            t = re.sub(r"\x1b\[[0-9;]*m", "", open(f"{R}/teltxob_{r}_{m}_{s}.log", errors="replace").read())
            per[r] = [(c, e, int(w), int(p)) for c, e, w, p in LINE.findall(t)]
        base = per["v1"]
        for r in routers[1:]:
            same = len(per[r]) == len(base) and all(a[3] == b[3] and a[2] == b[2] for a, b in zip(base, per[r]))
            tot = sum(x[3] for x in per[r]), sum(x[3] for x in base)
            print(f"{m} {s} {r}: {len(per[r])} core readings (main {len(base)}); packets {tot[0]} (main {tot[1]}); "
                  f"every core's packets and words equal main's: {same}")
    for r in routers:
        t = re.sub(r"\x1b\[[0-9;]*m", "", open(f"{R}/teltxo_{r}_{m}_gtest.log", errors="replace").read())
        res = re.findall(r"\[  (PASSED|FAILED)  \] (\d+) test", t)
        print(f"{m} gtest {r}: {res}")
