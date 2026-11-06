#!/usr/bin/env python3
"""cisp_shipab_report.py RESULTS_DIR: per arm of cisp_ship_ab.sh (base, shipA, pumpB), the slow-iteration clusters:
from the zone CSV, each chip's sender-kernel duration per launch (BRISC-KERNEL zones > 1M cycles), its excess over the
same row's minimum in that arm; a hit iteration has an excess > 300k cycles on at least one chip. Clusters: hits within
two iterations of each other. Reports the clusters' start times (log time), periods, lengths, the number of hit
iterations, the sender-CSV slow count (below 97% of the off arm's median for the test) and the mean excess."""
import collections, csv, os, re, statistics, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cisp_pcs2_report as P  # noqa: E402

D = sys.argv[1]


def zones(path):
    f = open(path)
    f.readline()
    rows = [{k.strip(): (v.strip() if v else v) for k, v in r.items()} for r in csv.DictReader(f)]
    byc = collections.defaultdict(list)
    for r in rows:
        if r["zone name"] == "BRISC-KERNEL":
            byc[(int(r["PCIe slot"]), int(r["core_x"]), int(r["core_y"]))].append(
                (int(r["time[cycles since reset]"]), r["type"]))
    chips = collections.defaultdict(list)
    for (ch, x, y), ev in byc.items():
        ev.sort()
        st = [t for t, ty in ev if ty == "ZONE_START"]
        en = [t for t, ty in ev if ty == "ZONE_END"]
        chips[ch] += [(s, e) for s, e in zip(st, en)]
    launches = {}
    for ch, z in chips.items():
        z.sort()
        groups = [[z[0]]]
        for a, b in zip(z, z[1:]):
            (groups.append([b]) if b[0] - a[0] > 1.35e6 else groups[-1].append(b))
        launches[ch] = [statistics.median([e - s for s, e in g if e - s > 1e6] or [0]) for g in groups]
    return launches


off = collections.defaultdict(list)
offp = os.path.join(D, "ab_off_bandwidth_results.csv")
if os.path.exists(offp):
    for r in csv.DictReader(open(offp)):
        off[r["test_name"]].append(float(r["bandwidth_GB_s"]))
for arm in (sys.argv[2].split(",") if len(sys.argv) > 2 else ("base", "shipA", "pumpB")):
    zp = os.path.join(D, f"ab_{arm}_zones.csv")
    if not os.path.exists(zp):
        print(f"== {arm}: no zone CSV")
        continue
    L = zones(zp)
    win = P.windows(os.path.join(D, f"ab_{arm}.log"))
    names = [re.sub(r"R\d\d_", "_", t) for t, *_ in win]
    n = min(len(win), min(len(v) for v in L.values()))
    rowmin = collections.defaultdict(lambda: collections.defaultdict(lambda: float("inf")))
    for i in range(n):
        for ch in L:
            if L[ch][i]:
                rowmin[names[i]][ch] = min(rowmin[names[i]][ch], L[ch][i])
    hits, ex = [], []
    for i in range(n):
        e = [L[ch][i] - rowmin[names[i]][ch] for ch in L if L[ch][i]]
        k = sum(1 for x in e if x > 300e3)
        if k:
            hits.append(i)
            ex += [x for x in e if x > 300e3]
    clusters = []
    for i in hits:
        (clusters[-1].append(i) if clusters and i - clusters[-1][-1] <= 2 else clusters.append([i]))
    t0 = win[0][2]
    starts = [win[c[0]][2] - t0 for c in clusters]
    per = [b - a for a, b in zip(starts, starts[1:])]
    sender = list(csv.DictReader(open(os.path.join(D, f"ab_{arm}_bandwidth_results.csv"))))
    slow = sum(1 for r in sender if r["test_name"] in off and float(r["bandwidth_GB_s"]) < 0.97 * statistics.median(off[r["test_name"]]))
    print(f"== {arm}: {n} iterations, run {win[n - 1][3] - t0:.1f} s; hit iterations {len(hits)} in {len(clusters)} clusters; "
          f"sender-CSV slow {slow}; excess per hit chip-iteration median {statistics.median(ex) / 1e3 if ex else 0:.0f}k cycles")
    print("   cluster starts (s): " + " ".join(f"{s:.2f}" for s in starts))
    print("   gaps (s): " + " ".join(f"{p:.2f}" for p in per))
    print("   cluster lengths (iterations): " + " ".join(str(c[-1] - c[0] + 1) for c in clusters))
    print("   iterations per period (cluster start index gaps): " +
          " ".join(str(b[0] - a[0]) for a, b in zip(clusters, clusters[1:])))
