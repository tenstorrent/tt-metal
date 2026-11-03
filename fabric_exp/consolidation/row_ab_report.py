#!/usr/bin/env python3
"""row_ab_report.py TAG: cycles/packet (1.35 GHz) per row and router from row_ab_runs.sh, each repetition and the mean.
A row whose yaml runs several tests (or sizes, or a trimming capture + replay) prints one line per CSV row, labelled
row:test ntype size (and #1 for a repeated one: the trimming replay); a one-test row prints as before."""
import collections, csv, glob, os, re, sys
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
tag = sys.argv[1]
data = collections.defaultdict(lambda: collections.defaultdict(list))
routers = []
for f in sorted(glob.glob(f"{R}/rab{tag}_*.csv")):
    m = re.match(rf"rab{tag}_(v[^_]+)_(.+)_(\d+)\.csv$", os.path.basename(f))
    if not m:
        continue
    router, row, rep = m.groups()
    if router not in routers:
        routers.append(router)
    rs = list(csv.DictReader(open(f)))
    seen = collections.Counter()
    for r in rs:
        sub = row
        if len(rs) > 1:
            k = (r["test_name"], r["ntype"].replace("NOC_UNICAST_", "").lower(), r["packet_size"])
            sub = f"{row}:{k[0]} {k[1]} {k[2]}" + (f" #{seen[k]}" if seen[k] else "")
            seen[k] += 1
        data[sub][router].append(1.35e9 / float(r["avg_packets_per_s"]))
w = max([12] + [len(r) for r in data])
print(f"{'row':{w}}" + "".join(f"{r:>22}" for r in routers))
for row, per in data.items():
    print(f"{row:{w}}" + "".join(f"{sum(per[r]) / len(per[r]):9.1f} ({' / '.join(f'{v:.1f}' for v in per[r])})".rjust(22) if per[r] else f"{'-':>22}" for r in routers))
