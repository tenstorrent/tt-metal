#!/usr/bin/env python3
# Per-OP detail (like ops_slowest_device_3L_first.log) for N layers, SLOWEST device, WARM (last) pass.
# Lists every op: idx, OP CODE, KERNEL_us, FW_us, OP2OP_us, cores. Layers delimited by
# forward_chunk_layer_{i}_start/_end signposts. Usage: parse_NL_detail.py <ops_csv> <label> <out_log>
import csv, sys, re
from collections import defaultdict

ops_csv, label, out = sys.argv[1], sys.argv[2], sys.argv[3]

def fnum(s):
    try: return float(s)
    except: return 0.0

rows = []
start_sig = defaultdict(list); end_sig = defaultdict(list)
re_s = re.compile(r"^forward_chunk_layer_(\d+)_start$"); re_e = re.compile(r"^forward_chunk_layer_(\d+)_end$")
with open(ops_csv) as f:
    for i, r in enumerate(csv.DictReader(f)):
        oc = (r.get("OP CODE") or "").strip()
        ms = re_s.match(oc); me = re_e.match(oc)
        if ms: start_sig[int(ms.group(1))].append(i); continue
        if me: end_sig[int(me.group(1))].append(i); continue
        try: dev = int(r["DEVICE ID"])
        except: continue
        if not oc: continue
        rows.append((i, oc, dev, fnum(r.get("DEVICE FW DURATION [ns]")), fnum(r.get("DEVICE KERNEL DURATION [ns]")),
                     fnum(r.get("OP TO OP LATENCY [ns]")), (r.get("CORE COUNT") or "").strip()))

layers = sorted(start_sig.keys())
warm_start = {L: max(start_sig[L]) for L in layers}
warm_lo = warm_start[layers[0]]
warm_hi = max(end_sig[layers[-1]]) if layers[-1] in end_sig else max(x[0] for x in rows)
warm = [x for x in rows if warm_lo <= x[0] <= warm_hi]
ksum = defaultdict(float)
for x in warm: ksum[x[2]] += x[4]
slow = max(ksum, key=ksum.get)
sd = [x for x in warm if x[2] == slow]
bounds = sorted((warm_start[L], L) for L in layers)
def layer_of(ix):
    cur = layers[0]
    for rix, L in bounds:
        if ix >= rix: cur = L
        else: break
    return cur
by = defaultdict(list)
for x in sorted(sd, key=lambda r: r[0]): by[layer_of(x[0])].append(x)

full = [f"# N-LAYER per-op profile, SLOWEST device — {label} (WARM pass)",
        f"# slowest DEVICE ID={slow}; {len(layers)} layers; warm total kernel={ksum[slow]/1e6:.2f}ms over {len(sd)} ops"]
for L in layers:
    ops = by.get(L, [])
    if not ops: continue
    ks = sum(x[4] for x in ops)/1e6; fs = sum(x[3] for x in ops)/1e6; o2 = sum(x[5] for x in ops)/1e6
    typ = "DENSE" if L == 0 else "MoE"
    full.append(f"\n## layer{L} {typ}: {len(ops)} ops  kernel_sum={ks:.3f}ms fw_sum={fs:.3f}ms op2op_sum={o2:.3f}ms")
    full.append(f"  {'idx':>3} {'OP CODE':42s} {'KERNEL_us':>10} {'FW_us':>10} {'OP2OP_us':>9} {'cores':>6}")
    for j, x in enumerate(ops):
        full.append(f"  {j:3d} {x[1][:42]:42s} {x[4]/1e3:10.2f} {x[3]/1e3:10.2f} {x[5]/1e3:9.2f} {x[6]:>6}")
open(out, "w").write("\n".join(full) + "\n")
print(f"wrote {out}: slowest dev {slow}, {len(layers)} layers, {len(sd)} ops")
