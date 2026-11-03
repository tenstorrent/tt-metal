#!/usr/bin/env python3
"""txc_trim_report.py TAG: cycles/packet (1.35 GHz) per test, ntype and pass kind (capture: untrimmed; replay: channels
compiled out) and router, from txc_trim_ab_runs.sh's CSVs (each test writes its capture rows, then its replay rows),
with each pass's value and the mean."""
import collections, csv, glob, os, re, sys
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
tag = sys.argv[1]
data = collections.defaultdict(lambda: collections.defaultdict(list))
routers = []
for f in sorted(glob.glob(f"{R}/txt{tag}_*.csv"), key=lambda p: int(re.search(r"_(\d+)\.csv$", p).group(1))):
    router = re.match(rf"txt{tag}_(.+)_\d+\.csv$", os.path.basename(f)).group(1)
    if router not in routers:
        routers.append(router)
    seen = collections.Counter()
    for r in csv.DictReader(open(f)):
        key = (r["test_name"], r["ntype"])
        kind = "capture" if seen[key] == 0 else "replay"
        seen[key] += 1
        data[(r["test_name"], r["ntype"].replace("NOC_UNICAST_", "").lower(), kind)][router].append(
            1.35e9 / float(r["avg_packets_per_s"]))
print(f"{'row':44}" + "".join(f"{r:>26}" for r in routers) + "   delta")
for row, per in data.items():
    means = [sum(per[r]) / len(per[r]) if per[r] else None for r in routers]
    cells = "".join(f"{m:9.1f} ({' / '.join(f'{v:.1f}' for v in per[r])})".rjust(26) if m else f"{'-':>26}"
                    for m, r in zip(means, routers))
    d = f"{means[1] - means[0]:+7.1f}" if len(means) > 1 and None not in means[:2] else ""
    print(f"{' '.join(row):44}" + cells + "  " + d)
