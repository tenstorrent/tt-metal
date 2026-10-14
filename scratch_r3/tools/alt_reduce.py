#!/usr/bin/env python3
"""#58817 legacy port: reduce the in-process alternation runs of test_leg_alt.py.
usage: alt_reduce.py <out dir> <run tag>   (reads out_<tag>_*_*/reports/*/ops_perf_results_*.csv)
Per group (case, variant, round): the median device kernel duration of its launches after the first. Round 0 (the
builds) is dropped. Each variant is paired with variant 0 by round and process; the table gives the median group value,
the median paired change in ns and percent, the paired t, and how many pairs read faster. Variant 4 against 0 is the A/A."""
import csv, glob, math, os, statistics, sys
from collections import defaultdict

d, tag = sys.argv[1], sys.argv[2]
COL = "DEVICE KERNEL DURATION [ns]"
NAMES = {0: "none", 1: "both (port)", 2: "row MOP only", 3: "Auto TTSync only", 4: "none, A/A copy"}
groups = defaultdict(dict)  # (case, v) -> {(run, round): value}
order = []
for rd in sorted(glob.glob(os.path.join(d, f"out_{tag}_*_*"))):
    fs = glob.glob(os.path.join(rd, "reports", "*", "ops_perf_results_*.csv"))
    if not fs:
        print(f"# no report in {rd}", file=sys.stderr)
        continue
    run = os.path.basename(rd)
    cur, vals = None, []

    def close():
        if cur and cur[2] > 0 and len(vals) > 1:
            groups[(cur[0], cur[1])][(run, cur[2])] = statistics.median(vals[1:])

    for row in csv.DictReader(open(fs[0])):
        if row.get("OP TYPE") == "signpost":
            close()
            h = row["OP CODE"].split("|")
            cur = (h[1], int(h[2]), int(h[3])) if len(h) == 4 and h[0] == "ALT" else None
            vals = []
            if cur and cur[0] not in order:
                order.append(cur[0])
            continue
        if cur:
            try:
                vals.append(float(row[COL]))
            except (ValueError, KeyError):
                pass
    close()
print("| case | variant | pairs | median ns | change ns | change % | paired t | faster pairs |")
print("|---|---|---|---|---|---|---|---|")
for case in order:
    base = groups.get((case, 0), {})
    for v in (1, 2, 3, 4):
        g = groups.get((case, v), {})
        keys = sorted(set(base) & set(g))
        if not keys:
            continue
        diffs = [g[k] - base[k] for k in keys]
        m0 = statistics.median(base[k] for k in keys)
        md = statistics.median(diffs)
        sd = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
        t = statistics.mean(diffs) / (sd / math.sqrt(len(diffs))) if sd > 0 else float("nan")
        faster = sum(x < 0 for x in diffs)
        print(f"| {case} | {v} {NAMES[v]} | {len(keys)} | {statistics.median(g[k] for k in keys):.0f} (none {m0:.0f}) | "
              f"{md:+.0f} | {100 * md / m0:+.2f} | {t:+.1f} | {faster}/{len(keys)} |")
