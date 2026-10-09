# Summarize alt3.py runs. The four forms of a case are on1, off1, on2, off2 in the order alt3.py lists them. Per run and
# case: the median device time of each form over its calls (pool cases: only the pool op, not the halo op); the A/A of
# each side (on1 - on2, off1 - off2) and the effect (median of off1 and off2 pooled minus median of on1 and on2 pooled:
# how much slower the case is without the edit). Verdict rule, written before the run: "separates" when every run's
# effect has the same sign and the median effect is larger than every A/A magnitude of those runs; "equal" otherwise.
import csv, glob, os, statistics, sys
from collections import defaultdict
ROLES = {"AA1": 0, "OFF1": 1, "AA2": 2, "OFF2": 3, "BA1": 0, "BOFF1": 1, "BA2": 2, "BOFF2": 3,
         "SA1": 0, "SOFF1": 1, "SA2": 2, "SOFF2": 3, "PA1": 0, "POFF1": 1, "PA2": 2, "POFF2": 3}
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
        code = (row.get("OP CODE") or "").lower()
        if cur[0].startswith("pool") and ("pool" not in code or "halo" in code):
            continue
        try:
            per[(cur[0], ROLES[cur[1]])][cur[2]] += float(row["DEVICE KERNEL DURATION [ns]"])
        except (ValueError, KeyError):
            pass
    for n in sorted({k[0] for k in per}):
        v = [list(per[(n, r)].values()) for r in range(4)]
        if not all(v):
            continue
        for r in range(4): pooled[n][r] += v[r]
        on, off = statistics.median(v[0] + v[2]), statistics.median(v[1] + v[3])
        a_on = statistics.median(v[0]) - statistics.median(v[2]); a_off = statistics.median(v[1]) - statistics.median(v[3])
        aa[n] += [a_on, a_off]; eff[n].append(off - on); base[n].append(on)
        print(f"run {os.path.basename(rd)} {n}: with {on:.0f} without {off:.0f} effect {off - on:+.0f} A/A {a_on:+.0f} {a_off:+.0f} calls {[len(x) for x in v]}")
print("| case | with the edit, ns | without minus with, ns (median; per run) | percent | A/A, ns (on side; off side; per run) | verdict |")
print("|---|---|---|---|---|---|")
for n in sorted(eff):
    m = statistics.median(base[n]); e = statistics.median(eff[n]); big = max(abs(x) for x in aa[n])
    same = all(x > 0 for x in eff[n]) or all(x < 0 for x in eff[n])
    verdict = ("separates, the edit faster" if e > 0 else "separates, the edit slower") if same and abs(e) > big else "equal"
    f = lambda v: ", ".join(f"{x:+.0f}" for x in v)
    print(f"| {n} | {m:.0f} | {e:+.0f} ({f(eff[n])}) | {100 * e / m:+.2f} | on {f(aa[n][0::2])}; off {f(aa[n][1::2])} | {verdict} |")
print()
print("Pooled over runs (every call): case, calls per form, with (median), without (median), effect, A/A on, A/A off")
for n in sorted(pooled):
    v = pooled[n]; on = statistics.median(v[0] + v[2]); off = statistics.median(v[1] + v[3])
    print(f"{n}: {len(v[0])} calls, with {on:.0f}, without {off:.0f}, effect {off - on:+.0f} ({100 * (off - on) / on:+.2f} percent), "
          f"A/A on {statistics.median(v[0]) - statistics.median(v[2]):+.0f}, off {statistics.median(v[1]) - statistics.median(v[3]):+.0f}")
