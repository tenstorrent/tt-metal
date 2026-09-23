"""Aggregate one Tracy CSV into a per-op-code kernel-time table.

Sums DEVICE KERNEL DURATION only, over the ops after the last signpost("start").
Never sums op-to-op gaps: the gap is dispatch, not kernel work, and the wall
time from execute_trace is the authoritative number.
"""

import collections
import csv
import sys

path = sys.argv[1]
rows = list(csv.DictReader(open(path)))
starts = [i for i, r in enumerate(rows) if (r.get("OP CODE") or "").strip() == "start"]
if not starts:
    print("  no signpost found in", path)
    sys.exit(1)
window = rows[starts[-1] + 1 :]

agg = collections.defaultdict(lambda: [0, 0.0])
for r in window:
    op = (r.get("OP CODE") or "").strip()
    dur = (r.get("DEVICE KERNEL DURATION [ns]") or "").strip()
    if not op or op in ("start", "stop") or not dur:
        continue
    try:
        ns = float(dur)
    except ValueError:
        continue
    agg[op][0] += 1
    agg[op][1] += ns

total = sum(v[1] for v in agg.values())
calls = sum(v[0] for v in agg.values())
print("WINDOWS=%d OPS=%d KERNEL_SUM_US=%.1f" % (len(starts), calls, total / 1000))
for op, (n, ns) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
    print("ROW|%s|%d|%.1f|%.1f" % (op, n, ns / 1000, 100 * ns / total))
