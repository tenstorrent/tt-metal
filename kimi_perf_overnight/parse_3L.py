#!/usr/bin/env python3
# Parse a 3-layer ops_perf_results CSV -> per-layer (0 dense, 1 MoE, 2 MoE) ops on the SLOWEST device,
# WARM pass (last forward_chunk). Delimits by ROW ORDER (signpost rows have empty GLOBAL CALL COUNT).
# Usage: parse_3L.py <ops_csv> <label> <logical_n> <out_log>
import csv, sys
from collections import defaultdict

ops_csv, label, logical_n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
LAYER_START = {0: "forward_chunk_layer_0_start", 1: "forward_chunk_layer_1_start", 2: "forward_chunk_layer_2_start"}
LAYER_END2 = "forward_chunk_layer_2_end"
SIG_ALL = {f"forward_chunk_layer_{i}_{e}" for i in (0, 1, 2) for e in ("start", "end")} | {"MLA_START", "MLA_END", "MoE_END"}

def fnum(s):
    try: return float(s)
    except: return 0.0

rows = []  # (rowidx, oc, dev_or_None, fw, kd, lat, cores)
sig = defaultdict(list)  # signpost name -> [rowidx]
with open(ops_csv) as f:
    for i, r in enumerate(csv.DictReader(f)):
        oc = (r.get("OP CODE") or "").strip()
        if oc in SIG_ALL:
            sig[oc].append(i); continue
        try: dev = int(r["DEVICE ID"])
        except: continue
        if not oc: continue
        rows.append((i, oc, dev, fnum(r.get("DEVICE FW DURATION [ns]")), fnum(r.get("DEVICE KERNEL DURATION [ns]")),
                     fnum(r.get("OP TO OP LATENCY [ns]")), (r.get("CORE COUNT") or "").strip()))

# WARM pass = LAST forward_chunk: from last layer_0_start to last layer_2_end (by row index)
warm_lo = max(sig[LAYER_START[0]]); warm_hi = max(sig[LAYER_END2])
bnd = {L: max(sig[LAYER_START[L]]) for L in (0, 1, 2)}  # last occurrence of each layer start
def layer_of(ix):
    return 2 if ix >= bnd[2] else (1 if ix >= bnd[1] else 0)
warm = [x for x in rows if warm_lo <= x[0] <= warm_hi]

ksum = defaultdict(float)
for x in warm: ksum[x[2]] += x[4]
slow = max(ksum, key=ksum.get)
sd = [x for x in warm if x[2] == slow]
layers = defaultdict(list)
for x in sd: layers[layer_of(x[0])].append(x)

LNAME = {0: "layer0 DENSE", 1: "layer1 MoE", 2: "layer2 MoE"}
full = [f"# 3-LAYER per-op profile, SLOWEST device — {label} (logical_n={logical_n}), WARM pass",
        f"# slowest DEVICE ID={slow}; warm total kernel={ksum[slow]/1e6:.2f}ms over {len(sd)} ops"]
summary = [f"\n===== {label} (logical_n={logical_n}) — slowest device {slow} ====="]
for L in (0, 1, 2):
    ops = layers.get(L, [])
    if not ops: continue
    ks = sum(x[4] for x in ops) / 1e6; fs = sum(x[3] for x in ops) / 1e6; o2 = sum(x[5] for x in ops) / 1e6
    summary.append(f"  {LNAME[L]:12s}: {len(ops):3d} ops  kernel={ks:7.2f}ms  op2op(warm gaps)={o2:7.2f}ms")
    # aggregate by op code for the summary
    agg = defaultdict(lambda: [0, 0.0, 0.0])
    for x in ops:
        agg[x[1]][0] += 1; agg[x[1]][1] += x[4]; agg[x[1]][2] += x[5]
    top = sorted(agg.items(), key=lambda kv: -kv[1][1])[:6]
    for nm, (n, k, o) in top:
        summary.append(f"        {nm[:38]:38s} x{n:<3d} kernel={k/1e3:8.1f}us  op2op={o/1e3:7.1f}us")
    # full per-op listing for the log
    full.append(f"\n## {LNAME[L]}: {len(ops)} ops  kernel_sum={ks:.3f}ms fw_sum={fs:.3f}ms op2op_sum={o2:.3f}ms")
    full.append(f"  {'idx':>3} {'OP CODE':42s} {'KERNEL_us':>10} {'FW_us':>10} {'OP2OP_us':>9} {'cores':>6}")
    for j, x in enumerate(ops):
        full.append(f"  {j:3d} {x[1][:42]:42s} {x[4]/1e3:10.2f} {x[3]/1e3:10.2f} {x[5]/1e3:9.2f} {x[6]:>6}")
open(out, "w").write("\n".join(full) + "\n")
print("\n".join(summary))
