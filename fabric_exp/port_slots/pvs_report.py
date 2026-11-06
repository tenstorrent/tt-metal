#!/usr/bin/env python3
"""pvs_report.py [--results DIR ...] [--arms study port [portdec]]: port against study (pvs_runs.sh): per suite, test
row, packet size and noc send type, cycles/packet at 1.35 GHz from each run's per-test CSV (the busiest link's packets
over its sender's cycles; several CSV lines of a test: their mean), per pass and arm, and the port against the study
(mean of the two passes). With the portdec arm (the route-decode A/B, SLOT_ALLOCATOR.md section 12.13), also PORTDEC
against PORT per pass (cycles/packet and %), PORTDEC against STUDY (mean of the passes), and the share of the port's gap
to the study that the decode closes ((port - portdec) / (port - study), only where the port is slower by more than 1%).
Also the slot-row check from the job log when given (--log)."""
import argparse
import collections
import csv
import os
import re
import statistics as st

CLK = 1.35e9
SUITES = ["ns", "t42ns", "c2d", "c1lin", "c1ring", "c1ne"]


def rows_of(path):
    acc = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        pps = float(r["packets_per_second"])
        if pps > 0:
            m = re.match(r"(.*)_size_(\d+)_ntype_(\w+)$", r["test_name"])
            key = (m.group(1), int(m.group(2)), m.group(3)) if m else (r["test_name"], int(r["packet_size"]), "")
            acc[key].append(CLK / pps)
    return {k: st.mean(v) for k, v in acc.items()}


def find(dirs, name):
    for d in dirs:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", nargs="+", default=["/home/aagarwal/tt-metal/fabric_exp/results"])
    ap.add_argument("--log", default=None, help="the lb2 job log, for the slot-row check lines")
    ap.add_argument("--arms", nargs="+", default=["study", "port"])
    a = ap.parse_args()
    if a.log:
        print("Slot-row check (both hosts' logged rows per suite):")
        for line in open(a.log, errors="replace"):
            if "rows match" in line or "rows differ" in line:
                print("   " + line.strip())
        print()
    for s in SUITES:
        d = {}
        for p in "ab":
            for arm in a.arms:
                f = find(a.results, f"pvs{p}_{arm}_{s}_tests.csv")
                if f:
                    d[(p, arm)] = rows_of(f)
        if not d:
            continue
        keys = sorted(set.intersection(*(set(v) for v in d.values())), key=lambda k: (k[0], k[2], k[1]))
        dec = "portdec" in a.arms
        print(f"## {s} (runs: {', '.join(f'{p}/{arm}' for p, arm in sorted(d))})\n")
        head = "| row | size | ntype | " + " | ".join(f"{arm} {p}" for arm in a.arms for p in "ab") + " | port vs study |"
        if dec:
            head += " dec - port a | dec - port b | dec vs port | dec vs study | gap closed |"
        print(head)
        print("| --- | ---: | --- | " + " | ".join("---:" for _ in head.split("|")[4:-1]) + " |")

        def mean(arm, k):
            v = [d[(p, arm)][k] for p in "ab" if (p, arm) in d]
            return st.mean(v) if v else None

        def rel(x, y):
            return f"{100 * (x / y - 1):+.1f}%" if x is not None and y is not None else "-"

        for k in keys:
            cells = [f"{d[(p, arm)][k]:.1f}" if (p, arm) in d else "-" for arm in a.arms for p in "ab"]
            sv, pv = mean("study", k), mean("port", k)
            row = f"| {k[0]} | {k[1]} | {k[2]} | " + " | ".join(cells) + f" | {rel(pv, sv)} |"
            if dec:
                dv = mean("portdec", k)
                for p in "ab":
                    if (p, "port") in d and (p, "portdec") in d:
                        x, y = d[(p, "portdec")][k], d[(p, "port")][k]
                        row += f" {x - y:+.1f} ({100 * (x / y - 1):+.1f}%) |"
                    else:
                        row += " - |"
                gap = f"{100 * (pv - dv) / (pv - sv):.0f}%" if None not in (sv, pv, dv) and pv > 1.01 * sv else "-"
                row += f" {rel(dv, pv)} | {rel(dv, sv)} | {gap} |"
            print(row)
        print()


if __name__ == "__main__":
    main()
