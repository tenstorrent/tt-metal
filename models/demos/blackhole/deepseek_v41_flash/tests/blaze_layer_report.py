"""Offline: one decode layer under trace replay on all chips -> per-op DRAM busy/idle table.
python blaze_layer_report.py <run dir> <ceiling GB/s per chip> [<ceiling for a narrow op GB/s> ...]

Inputs in <run dir>: raw.json (+ trace_cpp.csv = profiler digest after the trace replays, + moe_meta.pt = moe_compute dispatch metadata).
Per op: median-chip kernel us, slowest-chip us, op-to-op gap, DRAM bytes read per chip (estimated from the input tensors in DRAM; moe_compute from the
number of distinct local experts the routing activates on that chip), achieved GB/s, fraction of the measured ceiling, DRAM-idle us (kernel us - bytes/ceiling).
Layer: sum over ops, plus the replay window per chip from the kernel start/end cycles of the first and last program.
"""
import csv
import json
import re
import statistics
import sys
from collections import defaultdict

d = sys.argv[1]
CEIL = float(sys.argv[2])  # GB/s per chip (bytes per ns)
raw = json.load(open(d + "/raw.json"))
K, G = "DEVICE KERNEL DURATION [ns]", "OP TO OP LATENCY [ns]"
SC, EC = "DEVICE KERNEL START CYCLE", "DEVICE KERNEL END CYCLE"
op0, op1 = raw["op0"], raw["op1"]
n = op1 - op0
T0 = op0 - n
BPE = {
    "BFLOAT16": 2.0,
    "FLOAT32": 4.0,
    "BFLOAT8_B": 1.0625,
    "BFLOAT4_B": 0.5625,
    "UINT16": 2.0,
    "UINT32": 4.0,
    "INT32": 4.0,
}
PAT = re.compile(r"\[([0-9x]+)\] (\w+) (T|RM) (\S+)")


def F(x):
    return float(x) if x not in ("", None) else float("nan")


# ---- digest: prog -> session -> dev -> (dur, gap, start_cyc, end_cyc)
rows = defaultdict(lambda: defaultdict(dict))
for r in csv.DictReader(open(sys.argv[3] if len(sys.argv) > 3 else d + "/trace_cpp.csv")):
    if not r["METAL TRACE ID"]:
        continue
    gc = int(float(r["GLOBAL CALL COUNT"]))
    oid = gc >> 10
    if T0 <= oid < T0 + n:
        rows[oid - T0][r["METAL TRACE REPLAY SESSION ID"]][int(r["DEVICE ID"])] = (F(r[K]), F(r[G]), F(r[SC]), F(r[EC]))
cnt = defaultdict(int)
for p in rows:
    for s, v in rows[p].items():
        cnt[s] += len(v)
good = [s for s, c in cnt.items() if c == n * 32 and s not in ("1", "2", "3")]
print("good trace sessions", good, "programs", n)
devs = sorted({dv for p in rows for s in good for dv in rows[p][s]})

# ---- section names of the recorded calls
sec = {}
st = 0
for name, _, nrows in raw["eager_marks"]:
    for i in range(st, nrows):
        sec[i] = name
    st = nrows

# ---- routed-expert activity per chip
moe_active = {}
per_expert_mb = None
try:
    import torch

    meta = torch.load(d + "/moe_meta.pt")
    idx, mp = meta["idx"].to(torch.int64), meta["map"].to(torch.int64)
    ids = set(int(v) for v in idx.flatten().tolist())
    row0 = mp[0] if mp.dim() == 2 else mp
    nchip = int(row0.max()) + 1
    for c in range(32):
        moe_active[c] = len([e for e in ids if e < row0.numel() and int(row0[e]) == c])
    print(
        "moe active experts per chip (idx distinct %d, map max %d):" % (len(ids), int(row0.max())),
        [moe_active[c] for c in range(32)],
    )
except Exception as e:  # noqa
    print("moe meta not usable:", type(e).__name__, e)


def tensor_bytes(desc):
    m = PAT.match(desc)
    if not m or "DRAM" not in m.group(4):
        return 0.0
    dims = [int(v) for v in m.group(1).split("x")]
    if m.group(3) == "T" and len(dims) >= 2:
        dims[-1] = -(-dims[-1] // 32) * 32
        dims[-2] = -(-dims[-2] // 32) * 32
    v = 1
    for x in dims:
        v *= x
    return v * BPE.get(m.group(2), 2.0)


progs = []
for c in raw["eager_rows"]:
    for pid in range(c["id0"], c["id1"]):
        p = pid - op0
        op = c["op"].replace("ttnn.", "")
        ins = c["ins"]
        if (
            op.endswith("generic_op") and c["outs"]
        ):  # the output buffers are the trailing io tensors of a generic_op: not DRAM reads
            ins = ins[: len(ins) - len(c["outs"])]
        if op.endswith("moe_compute"):
            wb = (
                3 * 5120 * 2304 * 1.0625
            )  # one routed expert: gate, up, down (hidden 5120, intermediate 2304, bfp8); the weight tensors hold 12 experts/chip x several layers
            b_by_chip = {dv: moe_active.get(dv, 5) * wb for dv in devs}
            b = statistics.median(b_by_chip.values())
            small = sum(tensor_bytes(t) for t in ins[:4])
            b_by_chip = {dv: v + small for dv, v in b_by_chip.items()}
        else:
            b = sum(tensor_bytes(t) for t in ins)
            b_by_chip = None
        durs = [rows[p][s][dv] for s in good for dv in devs if dv in rows[p][s]]
        per_chip = defaultdict(list)
        for s in good:
            for dv, v in rows[p][s].items():
                per_chip[dv].append(v)
        progs.append(
            dict(
                p=p,
                sec=sec.get(c["idx"], "?"),
                op=op,
                caller=c["caller"].split("/")[-1],
                cores=0,
                bytes=b,
                b_by_chip=b_by_chip,
                per_chip=per_chip,
            )
        )


# per chip per program medians
def med(chip, p, i):
    v = [x[i] for x in progs[p]["per_chip"].get(chip, []) if x[i] == x[i]]
    return statistics.median(v) if v else float("nan")


layer_us = raw["traced_ms"] * 1e3
lines = []
lines.append(
    f"# layer {raw['layer']} B={raw['B']}: traced {layer_us:.1f} us (50 replays, no profiler), ceiling {CEIL:.1f} GB/s per chip\n"
)
lines.append(
    "| # | section | op | caller | kernel us med chip | slowest chip us | gap us | MB read/chip | achieved GB/s | frac of ceiling | DRAM-idle us (kernel-bytes/ceil) |"
)
lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
tot = dict(k=0.0, kmax=0.0, gap=0.0, mb=0.0, busy=0.0, idle=0.0)
secs = defaultdict(lambda: dict(k=0.0, gap=0.0, mb=0.0, busy=0.0, idle=0.0))
cats = defaultdict(lambda: dict(k=0.0, gap=0.0, mb=0.0, busy=0.0, idle=0.0))


def catof(pr):
    op, cl, sc = pr["op"], pr["caller"], pr["sec"]
    if any(w in op for w in ("reduce_scatter", "all_gather", "all_to_all")):
        return "CCL (reduce_scatter / all_gather / dispatch)"
    if "mhc" in cl or "mhc" in sc:
        return "mHC (mixes, collapse, expand)"
    if op == "rms_norm":
        return "norms"
    if "router" in cl:
        return "router"
    if "scaled_dot_product" in op:
        return "attention sdpa"
    if sc == "attention" and op == "linear":
        return "attention linears"
    if sc.startswith("shared_expert"):
        return "shared expert"
    if op.endswith("moe_compute"):
        return "routed experts (moe_compute)"
    return "glue (slice, layout, rope, moe_tail, add, attn_pre)"


for i, pr in enumerate(progs):
    ku = statistics.median([med(c, i, 0) for c in devs if med(c, i, 0) == med(c, i, 0)]) / 1e3
    kmax = max(med(c, i, 0) for c in devs if med(c, i, 0) == med(c, i, 0)) / 1e3
    gap = (statistics.median([med(c, i, 1) for c in devs if med(c, i, 1) == med(c, i, 1)]) / 1e3) if i else 0.0
    byt = (
        max(pr["b_by_chip"].values()) if pr["b_by_chip"] else pr["bytes"]
    )  # moe_compute: the busiest chip (most active experts) sets the op time
    tk = kmax if pr["b_by_chip"] else ku
    mb = byt / 1e6
    busy = byt / CEIL / 1e3
    ach = byt / (tk * 1e3) if tk else 0
    idle = max(0.0, ku - busy)
    lines.append(
        f"| {i} | {pr['sec']} | {pr['op']} | {pr['caller']} | {ku:.1f} | {kmax:.1f} | {gap:.1f} | {mb:.2f} | {ach:.0f} | {ach/CEIL:.2f} | {idle:.1f} |"
    )
    for t, v in (("k", ku), ("kmax", kmax), ("gap", gap), ("mb", mb), ("busy", busy), ("idle", idle)):
        if t in tot:
            tot[t] += v
    s = secs[pr["sec"]]
    s["k"] += ku
    s["gap"] += gap
    s["mb"] += mb
    s["busy"] += busy
    s["idle"] += idle
    cat = catof(pr)
    s = cats[cat]
    s["k"] += ku
    s["gap"] += gap
    s["mb"] += mb
    s["busy"] += busy
    s["idle"] += idle
lines.append("")
lines.append(
    "| section | kernel us | gaps us | MB read | DRAM busy us @ceiling | idle us inside kernels | idle share of (kernel+gap) |"
)
lines.append("|---|---|---|---|---|---|---|")
for k_, s in secs.items():
    lines.append(
        f"| {k_} | {s['k']:.1f} | {s['gap']:.1f} | {s['mb']:.1f} | {s['busy']:.1f} | {s['idle']:.1f} | {100*(1-s['busy']/(s['k']+s['gap'])) if s['k']+s['gap'] else 0:.0f}% |"
    )
lines.append("")
lines.append(
    "| category | kernel us (slowest chip for moe) | gaps us | MB read | DRAM busy us @ceiling | idle us inside kernels | idle share |"
)
lines.append("|---|---|---|---|---|---|---|")
for k_, s in sorted(cats.items(), key=lambda kv: -kv[1]["k"]):
    lines.append(
        f"| {k_} | {s['k']:.1f} | {s['gap']:.1f} | {s['mb']:.1f} | {s['busy']:.1f} | {s['idle']:.1f} | {100*(1-s['busy']/(s['k']+s['gap'])) if s['k']+s['gap'] else 0:.0f}% |"
    )
lines.append("")
lines.append(
    f"Sum: kernel {tot['k']:.1f} us (median chip) / {tot['kmax']:.1f} (sum of per-op slowest chip), gaps {tot['gap']:.1f} us, traced layer {layer_us:.1f} us."
)
lines.append(
    f"DRAM bytes {tot['mb']:.1f} MB/chip -> busy {tot['busy']:.1f} us at {CEIL:.0f} GB/s => idle {layer_us - tot['busy']:.1f} us of {layer_us:.1f} ({100*(1-tot['busy']/layer_us):.0f}%)."
)

# per chip: window and busy (moe bytes differ per chip)
lines.append("")
lines.append(
    "| chip | replay window us (first start to last end) | sum kernel us | moe_compute us | moe active experts | busy us @ceiling | idle us |"
)
lines.append("|---|---|---|---|---|---|---|")
moe_i = [i for i, pr in enumerate(progs) if pr["op"].endswith("moe_compute")]
win_all = []
for c in devs:
    cyc = []
    for i in range(len(progs)):
        v = [x for x in progs[i]["per_chip"].get(c, []) if x[2] == x[2]]
        if v:
            cyc.append(
                (
                    statistics.median([x[2] for x in v]),
                    statistics.median([x[3] for x in v]),
                    statistics.median([x[0] for x in v]),
                )
            )
    if not cyc:
        continue
    ghz = (cyc[0][1] - cyc[0][0]) / cyc[0][2] if cyc[0][2] else 1.35
    win = (cyc[-1][1] - cyc[0][0]) / ghz / 1e3
    sk = sum(x[2] for x in cyc) / 1e3
    byt = sum((pr["b_by_chip"][c] if pr["b_by_chip"] else pr["bytes"]) for pr in progs)
    busy = byt / CEIL / 1e3
    mu = med(c, moe_i[0], 0) / 1e3 if moe_i else 0
    win_all.append(win)
    lines.append(f"| {c} | {win:.1f} | {sk:.1f} | {mu:.1f} | {moe_active.get(c, '')} | {busy:.1f} | {win - busy:.1f} |")
if win_all:
    lines.append(
        f"\nwindow us: median {statistics.median(win_all):.1f}, max {max(win_all):.1f}, min {min(win_all):.1f}"
    )
out = "\n".join(lines)
open(d + "/layer_report.md", "w").write(out)
print(out)
