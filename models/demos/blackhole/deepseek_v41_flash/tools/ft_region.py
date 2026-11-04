"""python3 -I ft_region.py <run dir> [layer mark name, default layer3] [--trace 0]

Fuse-tail region table of a replay profile (tests/test_fusedec_round_prof.py run with the profiler on, FD_ONLY=M FD_REPS=1): for the attention tail (wo_b .. mHC post) and the MoE tail (moe_tail .. mHC post) of one layer
it prints per op the DEVICE KERNEL DURATION (mean / slowest / fastest chip), the per-chip SUM of the region and the per-chip SPAN (min DEVICE KERNEL START .. max DEVICE KERNEL END of the region's ops on that chip;
mean / max over chips). The span is what the layer pays, the sum double counts waiting. Regions are found by caller: attention = the op before the first reduce_scatter (ccl_lean.py:77) / fuse-tail op .. the
mHC expand / fuse-tail op after it; MoE = moe_tail .. the next mHC expand / fuse-tail op.
"""
import csv
import json
import os
import statistics
import sys
from collections import defaultdict

run = sys.argv[1]
layer = next((a for a in sys.argv[2:] if a.startswith("layer")), "layer3")
tid = sys.argv[sys.argv.index("--trace") + 1] if "--trace" in sys.argv else "0"
O = json.load(open(os.path.join(run, "oprows.json")))
rows = [r for r in O["rows"] if str(r.get("tid")) == tid]
marks = [m for m in O["marks"]]
names = [m[0] for m in marks]
i = names.index(layer)
lo = marks[i][1]
hi = marks[i + 1][1] if i + 1 < len(marks) else 1 << 60
lay = [r for r in rows if lo <= r["id0"] < hi]


def is_ft(r):
    return "fuse_tail_ar.py" in r["caller"]


def is_exp(r):
    return "mhc_expand3.py" in r["caller"] or is_ft(r)


def region(kind):
    if kind == "attn":
        k = next(j for j, r in enumerate(lay) if "ccl_lean.py:77" in r["caller"] or is_ft(r))
        s = k - 1
        e = next(j for j in range(k, len(lay)) if is_exp(lay[j]))
    else:
        s = next(j for j, r in enumerate(lay) if "moe_tail.py" in r["caller"])
        e = next(j for j in range(s + 1, len(lay)) if is_exp(lay[j]))
    return lay[s : e + 1]


K = "DEVICE KERNEL DURATION [ns]"
FS, FE = "DEVICE KERNEL START CYCLE", "DEVICE KERNEL END CYCLE"
byid = {}
for r in csv.DictReader(open(os.path.join(run, "prof/.logs/cpp_device_perf_report.csv"))):
    g = int(r["GLOBAL CALL COUNT"])
    if not r[FS] or not r[FE]:
        continue
    byid[(g >> 10, g & 1023)] = (float(r[K]) if r[K] else 0.0, int(r[FS]), int(r[FE]))
chips = sorted({c for (_, c) in byid})
CLK = 1.35
for kind in ("attn", "moe"):
    reg = region(kind)
    print(f"== {layer} {kind} tail: {len(reg)} ops")
    sums = defaultdict(float)
    spans = {}
    for r in reg:
        d = [byid[(r["id0"], c)][0] / 1000 for c in chips if (r["id0"], c) in byid]
        if not d:
            continue
        for c in chips:
            if (r["id0"], c) in byid:
                sums[c] += byid[(r["id0"], c)][0] / 1000
        print(
            f"  {r['op'][:38]:38s} {r['caller'][:38]:38s} mean {statistics.mean(d):6.1f} slowest {max(d):6.1f} fastest {min(d):6.1f}"
        )
    for c in chips:
        fs = [byid[(r["id0"], c)][1] for r in reg if (r["id0"], c) in byid]
        fe = [byid[(r["id0"], c)][2] for r in reg if (r["id0"], c) in byid]
        if fs:
            spans[c] = (max(fe) - min(fs)) / CLK / 1000
    sv = list(sums.values())
    sp = list(spans.values())
    print(
        f"  region SUM per chip: mean {statistics.mean(sv):.1f} max {max(sv):.1f} min {min(sv):.1f} | SPAN per chip: mean {statistics.mean(sp):.1f} max {max(sp):.1f} min {min(sp):.1f} us"
    )
