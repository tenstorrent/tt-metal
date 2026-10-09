# Summarize alt.py runs. Per run and case: the median device time of each form over its calls; the A/A of each side
# (AA1 minus AA2, OFF1 minus OFF2) and the effect (the median of OFF1 and OFF2 pooled minus the median of AA1 and AA2
# pooled: how much slower the case is without the define). Over the runs: the median effect, every run's effect, the
# A/A values, and the verdict: "separates" when every run's effect has the same sign and the median effect is larger
# than every A/A magnitude; "equal" otherwise.
import csv, glob, os, statistics, sys
from collections import defaultdict
runs = sorted(glob.glob(sys.argv[1] + "/out_*"))
eff, aa, base, pooled = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(lambda: defaultdict(list))
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
    for n in sorted({k[0] for k in per}):
        v = {f: list(per[(n, f)].values()) for f in ("AA1", "AA2", "OFF1", "OFF2")}
        if not all(v.values()):
            continue
        for f in v: pooled[n][f] += v[f]
        on, off = statistics.median(v["AA1"] + v["AA2"]), statistics.median(v["OFF1"] + v["OFF2"])
        a_on = statistics.median(v["AA1"]) - statistics.median(v["AA2"]); a_off = statistics.median(v["OFF1"]) - statistics.median(v["OFF2"])
        aa[n] += [a_on, a_off]; eff[n].append(off - on); base[n].append(on)
        print(f"run {os.path.basename(rd)} {n}: with {on:.0f} without {off:.0f} effect {off - on:+.0f} A/A {a_on:+.0f} {a_off:+.0f} calls {[len(v[f]) for f in v]}")
print("| case | with define, ns | without minus with, ns (median; per run) | percent | A/A, ns (on side; off side; per run) | verdict |")
print("|---|---|---|---|---|---|")
for n in sorted(eff):
    m = statistics.median(base[n]); e = statistics.median(eff[n]); big = max(abs(x) for x in aa[n])
    same = all(x > 0 for x in eff[n]) or all(x < 0 for x in eff[n])
    verdict = ("separates, define faster" if e > 0 else "separates, define slower") if same and abs(e) > big else "equal"
    f = lambda v: ", ".join(f"{x:+.0f}" for x in v)
    print(f"| {n} | {m:.0f} | {e:+.0f} ({f(eff[n])}) | {100 * e / m:+.2f} | on {f(aa[n][0::2])}; off {f(aa[n][1::2])} | {verdict} |")
print()
print("Pooled over runs (every call): case, calls per form, with (median), without (median), effect, A/A on, A/A off")
for n in sorted(pooled):
    v = pooled[n]; on = statistics.median(v["AA1"] + v["AA2"]); off = statistics.median(v["OFF1"] + v["OFF2"])
    print(f"{n}: {len(v['AA1'])} calls, with {on:.0f}, without {off:.0f}, effect {off - on:+.0f} ({100 * (off - on) / on:+.2f} percent), "
          f"A/A on {statistics.median(v['AA1']) - statistics.median(v['AA2']):+.0f}, off {statistics.median(v['OFF1']) - statistics.median(v['OFF2']):+.0f}")
