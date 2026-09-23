"""Per-op kernel table from a raw Tracy ops_perf_results CSV (no tt-perf-report needed).

Sums DEVICE KERNEL DURATION [ns] between the 'start' and 'stop' signposts.
Usage: python3 raw_stack.py <ops_perf_results.csv> [wall_ms]
"""

import collections
import csv
import sys

path = sys.argv[1]
wall = float(sys.argv[2]) if len(sys.argv) > 2 else None
rows = list(csv.DictReader(open(path)))
inside = False
agg = collections.OrderedDict()
for r in rows:
    code = r.get("OP CODE", "")
    if r.get("OP TYPE", "") == "signpost":
        inside = code == "start" if code in ("start", "stop") else inside
        continue
    if not inside:
        continue
    ns = r.get("DEVICE KERNEL DURATION [ns]", "")
    if not ns:
        continue
    key = code
    e = agg.setdefault(key, [0, 0.0, set()])
    e[0] += 1
    e[1] += float(ns) / 1000.0
    e[2].add(r.get("CORE COUNT", "?"))
total = sum(v[1] for v in agg.values())
print("  %-42s %4s %10s %7s %9s  cores" % ("OP", "x", "us", "share", "us/call"))
for k, (n, us, cores) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
    print("  %-42s %4d %10.1f %6.1f%% %9.1f  %s" % (k[:42], n, us, 100 * us / total, us / n, ",".join(sorted(cores))))
print(
    "  kernel total %.1f us over %d ops%s"
    % (total, sum(v[0] for v in agg.values()), "" if wall is None else " (traced wall %.3f ms)" % wall)
)
