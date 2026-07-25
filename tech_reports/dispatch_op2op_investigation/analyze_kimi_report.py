#!/usr/bin/env python3
"""Analyze the Kimi K2.6 per-op report from #50932 (branch ppopovic/op2opgaps).

Central question: is the op-to-op gap explained by "host per-op cost exceeds device per-op time"?
If so, an op's gap should collapse whenever the PRECEDING op gives the host enough cover -- i.e.
gap should fall as the previous op's kernel time rises, and ops following a long op should show
~zero gap. That is a testable prediction, and it distinguishes host-starvation from CCL skew.
"""

import re
import statistics
import sys
from collections import defaultdict

PATH = sys.argv[1] if len(sys.argv) > 1 else "/data/dgomez/perf-48314/results/kimi_perlayer_report.txt"

# e.g. "   24 mla    RingJointSDPADeviceOperation  32   965.84  1276.29   1564.46   1739.62"
ROW = re.compile(
    r"^\s*(\d+)\s+(\w+)\s+(\w*DeviceOperation)\s+(\d+)\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
)
LAYER = re.compile(r"^LAYER\s+(\d+)")

rows = []
layer = None
with open(PATH) as fh:
    for line in fh:
        ml = LAYER.match(line)
        if ml:
            layer = int(ml.group(1))
            continue
        m = ROW.match(line)
        if m and layer is not None:
            rows.append(
                {
                    "layer": layer,
                    "idx": int(m.group(1)),
                    "bucket": m.group(2),
                    "op": m.group(3).replace("DeviceOperation", ""),
                    "kern_min": float(m.group(5)),
                    "kern_max": float(m.group(6)),
                    "o2o_min": float(m.group(7)),
                    "o2o_max": float(m.group(8)),
                    "excluded": "region-entry sync" in line,
                }
            )

print(f"parsed {len(rows)} op rows across {len({r['layer'] for r in rows})} layers\n")
counted = [r for r in rows if not r["excluded"]]

tot_kern = sum(r["kern_max"] for r in counted)
tot_o2o = sum(r["o2o_max"] for r in counted)
print(f"totals (sum of per-op max, critical path): device {tot_kern/1000:.1f} ms   op2op {tot_o2o/1000:.1f} ms")
print(f"op2op fraction: {100*tot_o2o/(tot_kern+tot_o2o):.1f}%\n")

# ---- Prediction test: does the PREVIOUS op's kernel time predict this op's gap? ----
print("=" * 78)
print("Does the previous op's kernel time hide this op's dispatch?")
print("=" * 78)
by_layer = defaultdict(list)
for r in rows:
    by_layer[r["layer"]].append(r)

pairs = []  # (prev_kern_max, this_o2o_max)
for layer, ops in by_layer.items():
    ops.sort(key=lambda r: r["idx"])
    for prev, cur in zip(ops, ops[1:]):
        if cur["excluded"]:
            continue
        pairs.append((prev["kern_max"], cur["o2o_max"]))

BUCKETS = [(0, 25), (25, 50), (50, 100), (100, 250), (250, 500), (500, 1000), (1000, 1e9)]
print(f"{'prev op kernel (us)':>22} {'n':>5} {'median gap (us)':>17} {'mean gap':>10}")
for lo, hi in BUCKETS:
    sel = [g for k, g in pairs if lo <= k < hi]
    if not sel:
        continue
    label = f"{lo}-{hi if hi < 1e9 else 'inf'}"
    print(f"{label:>22} {len(sel):>5} {statistics.median(sel):>17.1f} {statistics.mean(sel):>10.1f}")

zero = [(k, g) for k, g in pairs if g < 5.0]
print(f"\nops with a near-zero gap (<5 us): {len(zero)} of {len(pairs)}")
if zero:
    print(f"  their predecessors' kernel time: median {statistics.median([k for k,_ in zero]):.1f} us, "
          f"min {min(k for k,_ in zero):.1f}, max {max(k for k,_ in zero):.1f}")
    others = [k for k, g in pairs if g >= 5.0]
    print(f"  everyone else's predecessors:     median {statistics.median(others):.1f} us")

# ---- Where the gap time actually accumulates ----
print("\n" + "=" * 78)
print("Total op2op contribution by op type (this is what to fix first)")
print("=" * 78)
agg = defaultdict(lambda: [0.0, 0, 0.0])  # op -> [sum gap, count, sum kern]
for r in counted:
    a = agg[r["op"]]
    a[0] += r["o2o_max"]
    a[1] += 1
    a[2] += r["kern_max"]
print(f"{'op':<40} {'n':>4} {'sum gap ms':>11} {'%of gap':>8} {'med gap us':>11} {'sum kern ms':>12}")
for op, (g, n, k) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:18]:
    med = statistics.median([r["o2o_max"] for r in counted if r["op"] == op])
    print(f"{op:<40} {n:>4} {g/1000:>11.1f} {100*g/tot_o2o:>7.1f}% {med:>11.1f} {k/1000:>12.1f}")

# ---- CCL vs non-CCL, to address the debate on the ticket ----
print("\n" + "=" * 78)
print("CCL vs non-CCL (the #50932 debate)")
print("=" * 78)
CCL = ("AllGather", "ReduceScatter", "Dispatch", "Combine", "AllToAll", "Broadcast")
is_ccl = lambda op: any(c.lower() in op.lower() for c in CCL)
for label, sel in (("CCL", [r for r in counted if is_ccl(r["op"])]),
                   ("non-CCL", [r for r in counted if not is_ccl(r["op"])])):
    if not sel:
        continue
    g = sum(r["o2o_max"] for r in sel)
    k = sum(r["kern_max"] for r in sel)
    print(
        f"{label:>8}: n={len(sel):>4}  sum gap {g/1000:>7.1f} ms ({100*g/tot_o2o:>4.1f}% of all gap)  "
        f"median gap {statistics.median([r['o2o_max'] for r in sel]):>7.1f} us  "
        f"gap/kern ratio {g/k:>5.2f}"
    )
