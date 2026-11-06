#!/usr/bin/env python3
"""port_slot_report.py [--results DIR ...] [--prefix ps] [--tol 1.0]: the port slot measurements (SLOT_ALLOCATOR.md
section 12.12; port_slot_runs.sh, arms in port_arms.sh).

Per arm, test row and packet size: cycles/packet at 1.35 GHz from the run's per-test CSV (as knee_report.py: the
busiest link's packets over its sender's cycles; a row with several CSV lines takes their mean), mean of the two
passes, and the largest pass spread. Sweeps: F2 / F1 forwarding (per suite), S receiver and forwarding per slot size.
The knee = the smallest swept depth within TOL % of the sweep's best, as the interval the sweep bounds it to ("13-16");
">D" when the largest depth still gains past TOL over the one before it. Per packet size: the largest knee over the
rows and the row that sets it. Then S side by side: the same depth at 3072 and 4352 B slots.
"""
import argparse
import collections
import csv
import os
import re
import statistics as st
import subprocess

CLK = 1.35e9
HERE = os.path.dirname(os.path.abspath(__file__))


def arms():
    out = subprocess.run(["bash", "-c", f"source {HERE}/port_arms.sh; port_arms"], capture_output=True, text=True).stdout
    return [line.split()[:5] for line in out.splitlines() if line.strip()]


def rows_of(path):
    acc = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        pps = float(r["packets_per_second"])
        if pps > 0:
            acc[(re.sub(r"_size_\d+.*$", "", r["test_name"]), int(r["packet_size"]))].append(CLK / pps)
    return {k: st.mean(v) for k, v in acc.items()}


def find(dirs, name):
    for d in dirs:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def knee(pts, vals, tol):
    best = min(vals)
    i = next(i for i, v in enumerate(vals) if v <= best * (1 + tol / 100))
    lo = pts[i - 1] + 1 if i > 0 else None
    if i == len(pts) - 1 and i > 0 and vals[-1] < vals[-2] * (1 - tol / 100):
        return pts[i], True, f">{pts[i - 1]}"
    return pts[i], False, f"{pts[i]}" if lo is None or lo == pts[i] else f"{lo}-{pts[i]}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="+", default=["/home/aagarwal/tt-metal/fabric_exp/results"])
    ap.add_argument("--prefix", default="ps")
    ap.add_argument("--tol", type=float, default=1.0)
    a = ap.parse_args()
    data, spread, npass = {}, 0.0, {}
    for s, arm, kind, suite, row in arms():
        passes = [rows_of(p) for p in (find(a.results, f"{a.prefix}{x}_{s}_{arm}_tests.csv") for x in "ab") if p]
        if not passes:
            continue
        keys = set.intersection(*(set(d) for d in passes))
        data[(s, arm)] = {k: st.mean(d[k] for d in passes) for k in keys}
        npass[(s, arm)] = len(passes)
        if len(passes) == 2:
            spread = max([spread] + [abs(passes[0][k] - passes[1][k]) for k in keys])
    print(f"Largest pass spread: {spread:.1f} cycles/packet; passes per arm: {sorted(set(npass.values()))}\n")
    sweeps = collections.defaultdict(list)  # (title) -> [(depth, set, arm)]
    for s, arm, kind, suite, row in arms():
        if (s, arm) not in data or row == "-":
            continue
        w, f, r = map(int, row.split(","))
        if s in ("F2", "F1"):
            sweeps[(f"{s} {suite}: forwarding (W {w}, R {r})")].append((f, s, arm))
        elif s == "S":
            slot = arm.split("_")[-1]
            var, d = ("receiver", r) if arm.startswith("r") else ("forwarding", f)
            sweeps[(f"S {suite}: {var} at {slot} B slots")].append((d, s, arm))
    knees = {}
    for title, pts in sorted(sweeps.items()):
        pts = sorted(pts)
        depths = [d for d, _, _ in pts]
        keys = set.intersection(*(set(data[(s, arm)]) for _, s, arm in pts))
        print(f"## {title}\n")
        print("| row | size | " + " | ".join(map(str, depths)) + " | knee |")
        print("| --- | ---: | " + " | ".join("---:" for _ in depths) + " | ---: |")
        by_size = collections.defaultdict(list)
        for k in sorted(keys, key=lambda k: (k[1], k[0])):
            vals = [data[(s, arm)][k] for _, s, arm in pts]
            d, more, text = knee(depths, vals, a.tol)
            by_size[k[1]].append((d, more, k[0], text))
            print(f"| {k[0]} | {k[1]} | " + " | ".join(f"{v:.1f}" for v in vals) + f" | {text} |")
        print("\nKnee per packet size (largest over the rows):")
        for size in sorted(by_size):
            kn = max(by_size[size], key=lambda t: (t[0], t[1]))
            knees[(title, size)] = kn[3]
            print(f"- {size} B: {kn[3]} ({kn[2]}; rows: " + ", ".join(f"{t[2]} {t[3]}" for t in sorted(by_size[size], key=lambda t: t[2])) + ")")
        print()
    # S side by side
    print("## S: the same depths at 3072 and 4352 B slots (cycles/packet; 4352 against 3072)\n")
    print("| arm | row | size | 3072 B slots | 4352 B slots | diff |")
    print("| --- | --- | ---: | ---: | ---: | ---: |")
    for s, arm, kind, suite, row in arms():
        if s != "S" or not arm.endswith("_3072"):
            continue
        twin = arm.replace("_3072", "_4352")
        if (s, arm) not in data or (s, twin) not in data:
            continue
        for k in sorted(set(data[(s, arm)]) & set(data[(s, twin)]), key=lambda k: (k[0], k[1])):
            v3, v4 = data[(s, arm)][k], data[(s, twin)][k]
            print(f"| {arm[:-5]} | {k[0]} | {k[1]} | {v3:.1f} | {v4:.1f} | {100 * (v4 / v3 - 1):+.1f}% |")


if __name__ == "__main__":
    main()
