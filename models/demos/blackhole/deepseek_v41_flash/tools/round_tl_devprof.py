"""Attribute the traced device time of one speculative round to phases.
python3 -I round_tl_devprof.py <cpp_device_perf_report.csv> <marks_trace<id>.json> <trace_id> [session_id]
Per op (GLOBAL CALL COUNT >> 10 = device-operation id of the capture, & 1023 = device): median over the devices of DEVICE KERNEL DURATION and of OP TO OP LATENCY, summed per phase of the
op-id ranges recorded at capture time (embedding, engram, layerN, head, sampler, commit, drafter; ops between two ranges = glue named after the neighbours; the accept rule is the glue between
the sampler and the commit)."""
import csv
import json
import statistics
import sys
from collections import defaultdict

csvp, marksp, tid = sys.argv[1], sys.argv[2], sys.argv[3]
M = json.load(open(marksp))
K, G = "DEVICE KERNEL DURATION [ns]", "OP TO OP LATENCY [ns]"
rows = defaultdict(lambda: defaultdict(list))  # session -> oid -> [(dev, kernel, gap, opcode)]
for r in csv.DictReader(open(csvp)):
    if r["METAL TRACE ID"] != tid:
        continue
    g = int(float(r["GLOBAL CALL COUNT"]))
    oid, dev = g >> 10, g & 1023
    f = lambda x: float(x) if x not in ("", None) else float("nan")
    rows[r["METAL TRACE REPLAY SESSION ID"]][oid].append((dev, f(r[K]), f(r[G]), r["OP CODE"]))
n_ops = M["end"] - M["begin"]
cnt = {s: sum(len(v) for v in d.values()) for s, d in rows.items()}
print("sessions (rows):", cnt, "ops in the captured round:", n_ops)
sess = sys.argv[4] if len(sys.argv) > 4 else max((s for s in cnt if s not in ("1", "2", "3")), key=lambda s: cnt[s], default=None)
if sess is None:
    sess = max(cnt, key=cnt.get)
print("using session", sess)
d = rows[sess]
ops = sorted(d)
marks = sorted(M["marks"], key=lambda m: m[1])


def phase_of(oid):
    prev, nxt = None, None
    for name, i0, i1 in marks:
        if i0 <= oid < i1:
            return "layers" if name.startswith("layer") else name
        if i1 <= oid:
            prev = name
        if i0 > oid and nxt is None:
            nxt = name
    pn = lambda x: "layers" if (x or "").startswith("layer") else (x or "start")
    if prev == "sampler" and nxt == "commit":
        return "accept(glue)"
    return f"glue {pn(prev)}->{pn(nxt)}"


agg = defaultdict(lambda: [0.0, 0.0, 0])
per_layer = defaultdict(lambda: [0.0, 0.0])
for oid in ops:
    ks = [x[1] for x in d[oid] if x[1] == x[1]]
    gs = [x[2] for x in d[oid] if x[2] == x[2]]
    k = statistics.median(ks) if ks else 0.0
    g = statistics.median(gs) if gs else 0.0
    ph = phase_of(oid)
    agg[ph][0] += k
    agg[ph][1] += g
    agg[ph][2] += 1
    for name, i0, i1 in marks:
        if i0 <= oid < i1 and name.startswith("layer"):
            per_layer[name][0] += k
            per_layer[name][1] += g
tot_k = sum(v[0] for v in agg.values()) / 1e6
tot_g = sum(v[1] for v in agg.values()) / 1e6
print(f"\ntotal: kernel {tot_k:.2f} ms + op-to-op gap {tot_g:.2f} ms = {tot_k + tot_g:.2f} ms over {len(ops)} ops (median over devices per op)\n")
print("| phase | ops | kernel ms | op-to-op ms | sum ms | % |\n|---|---|---|---|---|---|")
for ph, (k, g, c) in sorted(agg.items(), key=lambda kv: -(kv[1][0] + kv[1][1])):
    print(f"| {ph} | {c} | {k / 1e6:.3f} | {g / 1e6:.3f} | {(k + g) / 1e6:.3f} | {100 * (k + g) / 1e6 / (tot_k + tot_g):.1f} |")
if per_layer:
    ls = [(v[0] + v[1]) / 1e6 for v in per_layer.values()]
    print(f"\nlayers: {len(ls)} layers, sum {sum(ls):.2f} ms, mean {sum(ls) / len(ls):.3f} ms, min {min(ls):.3f}, max {max(ls):.3f}")
