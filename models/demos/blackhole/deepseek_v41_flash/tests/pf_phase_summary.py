"""python pf_phase_summary.py <profiler dir> <marks.json> [nl=2] [trace id]: per prefill-layer PHASE (marks written by prefill_layer._mark)
  traced replay (METAL TRACE ID rows of the last replay): kernel sum, wall span (first kernel start -> last kernel end), gap = span - kernel, op count
  eager pass (rows without trace id, the compile run): per phase and per OP NAME kernel sums.
Marks hold (tag, device-op-id at the END of the phase); the last nl 'start' marks are the traced capture, the first nl the eager compile pass."""
import bisect
import csv
import json
import sys
from collections import defaultdict

d, mf = sys.argv[1], sys.argv[2]
nl = int(sys.argv[3]) if len(sys.argv) > 3 else 2
tid = sys.argv[4] if len(sys.argv) > 4 else "1"
marks = json.load(open(mf))
starts = [i for i, m in enumerate(marks) if m[0] == "start"]
FREQ = 1.35  # cycles per ns
K = "DEVICE KERNEL DURATION [ns]"
rows = list(csv.DictReader(open(d + "/.logs/cpp_device_perf_report.csv")))


def phases(sel):
    """sel: marks slice -> (ids, tags): op id oid belongs to the first mark whose end id > oid."""
    ids = [m[1] for m in sel]
    return ids, [m[0] for m in sel]


def run(label, sel, rows_sel, names=False):
    ids, tags = phases(sel)
    per = defaultdict(
        lambda: defaultdict(lambda: [0.0, 10**30, 0, 0])
    )  # phase -> dev -> [kernel ns, min start, max end, count]
    byname = defaultdict(lambda: defaultdict(float))
    lo = ids[0] - 1
    for r in rows_sel:
        oid = int(float(r["GLOBAL CALL COUNT"])) >> 10
        if oid < ids[0] or oid >= ids[-1]:
            continue
        mi = bisect.bisect_right(ids, oid)
        ph = sel[mi][0]
        x = per[ph][(r["DEVICE ID"], mi)]
        x[0] += float(r[K] or 0)
        try:
            x[1] = min(x[1], int(r["DEVICE KERNEL START CYCLE"]))
            x[2] = max(x[2], int(r["DEVICE KERNEL END CYCLE"] or 0))
        except ValueError:
            pass
        x[3] += 1
        if names:
            byname[(ph, r["OP NAME"] or "?")][r["DEVICE ID"]] += float(r[K] or 0)
    nd = len({k[0] for v in per.values() for k in v})
    print(f"== {label}")
    tot_k = tot_s = 0
    for ph, v in sorted(per.items(), key=lambda kv: -sum(x[0] for x in kv[1].values())):
        k = sum(x[0] for x in v.values()) / nd / 1e6
        sp = sum((x[2] - x[1]) / FREQ for x in v.values() if x[2] > x[1]) / nd / 1e6
        c = sum(x[3] for x in v.values()) / nd
        tot_k += k
        tot_s += sp
        print(
            f"  {ph:22s} kernel {k:8.3f} ms  span {sp:8.3f} ms  gap {sp - k:7.3f} ms ({(sp - k) / max(sp, 1e-9):5.1%})  ops {c:6.0f}"
        )
    print(f"  TOTAL kernel {tot_k:.2f} ms  span {tot_s:.2f} ms")
    if names:
        t = sorted(((sum(v.values()) / nd / 1e6, k) for k, v in byname.items()), reverse=True)[:45]
        for ms, (ph, n) in t:
            print(f"     {ph:20s} {n:44s} {ms:8.3f} ms")


if len(starts) >= 2 * nl:
    run(
        "EAGER compile pass (first %d layers)" % nl,
        marks[starts[0] : starts[nl]],
        [r for r in rows if not r["METAL TRACE ID"]],
        names=True,
    )
    run("TRACED replay (last %d layers)" % nl, marks[starts[-nl] :], [r for r in rows if r["METAL TRACE ID"] == tid])
