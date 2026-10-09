# Summarize alt.py runs: per run, per case, the median device time of each form; the A/A (AA1 minus AA2) and the effect
# (OFF minus the median of AA1 and AA2 pooled, i.e. how much slower the case is without the define); then over the runs.
import csv, glob, os, statistics, sys
from collections import defaultdict
runs = sorted(glob.glob(sys.argv[1] + "/out_*"))
eff, aa, base = defaultdict(list), defaultdict(list), defaultdict(list)
for rd in runs:
    fs = glob.glob(f"{rd}/reports/*/ops_perf_results_*.csv")
    if not fs:
        print("no report", rd); continue
    per = defaultdict(lambda: defaultdict(float)); cur = None
    for row in csv.DictReader(open(fs[0])):
        if row.get("OP TYPE") == "signpost":
            p = row["OP CODE"].split("|"); cur = tuple(p) if len(p) == 3 else None; continue
        if cur is None:
            continue
        try:
            per[(cur[0], cur[1])][cur[2]] += float(row["DEVICE KERNEL DURATION [ns]"])
        except ValueError:
            pass
    names = sorted({k[0] for k in per})
    for n in names:
        a1 = list(per[(n, "AA1")].values()); a2 = list(per[(n, "AA2")].values()); b = list(per[(n, "OFF")].values())
        if not (a1 and a2 and b):
            continue
        ma = statistics.median(a1 + a2)
        aa[n].append(statistics.median(a1) - statistics.median(a2)); eff[n].append(statistics.median(b) - ma); base[n].append(ma)
        print(f"run {os.path.basename(rd)} {n}: with {ma:.0f} (AA1 {statistics.median(a1):.0f}, AA2 {statistics.median(a2):.0f}, {len(a1)}+{len(a2)} calls), without {statistics.median(b):.0f} ({len(b)} calls)")
print("| case | with define, ns (median over runs) | without minus with, ns (per run) | percent | A/A, ns (per run) |")
print("|---|---|---|---|---|")
for n in sorted(eff):
    m = statistics.median(base[n]); e = statistics.median(eff[n])
    f = lambda v: ", ".join(f"{x:+.0f}" for x in v)
    print(f"| {n} | {m:.0f} | {e:+.0f} ({f(eff[n])}) | {100 * e / m:+.2f} | {statistics.median(aa[n]):+.0f} ({f(aa[n])}) |")
