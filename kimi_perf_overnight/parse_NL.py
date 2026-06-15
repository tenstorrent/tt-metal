#!/usr/bin/env python3
# Parse an N-layer ops_perf_results CSV -> per-layer op2op + kernel aggregation on the SLOWEST device,
# WARM pass (LAST forward_chunk). Layers delimited by forward_chunk_layer_{i}_start/_end signpost rows
# (signpost rows have empty DEVICE ID). Generalizes parse_3L.py to any number of layers.
# Usage: parse_NL.py <ops_csv> <label> <out_log>
import csv, sys, re
from collections import defaultdict

ops_csv, label, out = sys.argv[1], sys.argv[2], sys.argv[3]

def fnum(s):
    try: return float(s)
    except: return 0.0

rows = []                     # (rowidx, oc, dev, fw, kd, lat, cores)
start_sig = defaultdict(list) # layer i -> [rowidx of last layer_i_start]
end_sig = defaultdict(list)   # layer i -> [rowidx of layer_i_end]
re_start = re.compile(r"^forward_chunk_layer_(\d+)_start$")
re_end = re.compile(r"^forward_chunk_layer_(\d+)_end$")

with open(ops_csv) as f:
    for i, r in enumerate(csv.DictReader(f)):
        oc = (r.get("OP CODE") or "").strip()
        ms = re_start.match(oc); me = re_end.match(oc)
        if ms:
            start_sig[int(ms.group(1))].append(i); continue
        if me:
            end_sig[int(me.group(1))].append(i); continue
        try: dev = int(r["DEVICE ID"])
        except: continue
        if not oc: continue
        rows.append((i, oc, dev,
                     fnum(r.get("DEVICE FW DURATION [ns]")),
                     fnum(r.get("DEVICE KERNEL DURATION [ns]")),
                     fnum(r.get("OP TO OP LATENCY [ns]")),
                     (r.get("CORE COUNT") or "").strip()))

layers = sorted(start_sig.keys())
if not layers:
    sys.stderr.write("no forward_chunk_layer_* signposts found\n"); sys.exit(1)

# WARM pass = the LAST occurrence of each layer's signposts (last forward_chunk).
warm_start = {L: max(start_sig[L]) for L in layers}
warm_end = {L: max(end_sig[L]) for L in layers if L in end_sig}
warm_lo = warm_start[layers[0]]
warm_hi = warm_end[layers[-1]] if layers[-1] in warm_end else max(x[0] for x in rows)
warm = [x for x in rows if warm_lo <= x[0] <= warm_hi]

# slowest device = max kernel sum across warm ops
ksum = defaultdict(float)
for x in warm: ksum[x[2]] += x[4]
slow = max(ksum, key=ksum.get)
sd = [x for x in warm if x[2] == slow]

# assign each op row to a layer by row-index boundaries (warm_start[L] .. next layer start)
bounds = sorted((warm_start[L], L) for L in layers)
def layer_of(ix):
    cur = layers[0]
    for rowix, L in bounds:
        if ix >= rowix: cur = L
        else: break
    return cur

# group ops by layer, preserving row order so we can identify each layer's FIRST op (the one whose
# op2op is perturbed when PREFILL_PROFILE_READ_EVERY inserts a profiler-read at the layer boundary).
ops_by_layer = defaultdict(list)
for x in sorted(sd, key=lambda r: r[0]):
    ops_by_layer[layer_of(x[0])].append(x)

per = {}  # L -> dict(kd, fw, op2op, op2op_excl1, nops, first_op2op, first_oc)
for L in layers:
    ol = ops_by_layer.get(L, [])
    kd = sum(x[4] for x in ol); fw = sum(x[3] for x in ol); op2 = sum(x[5] for x in ol)
    first_o = ol[0][5] if ol else 0.0
    first_oc = ol[0][1] if ol else "-"
    per[L] = dict(kd=kd, fw=fw, op2=op2, op2x=op2 - first_o, n=len(ol), f=first_o, foc=first_oc)

tot_k = sum(v["kd"] for v in per.values())
tot_o = sum(v["op2"] for v in per.values())
tot_ox = sum(v["op2x"] for v in per.values())
with open(out, "w") as o:
    o.write(f"# N-LAYER per-layer profile, SLOWEST device — {label} (WARM pass)\n")
    o.write(f"# slowest DEVICE ID={slow}; {len(layers)} layers; warm kernel_sum={tot_k/1e6:.2f}ms "
            f"op2op_sum={tot_o/1e6:.3f}ms (excl per-layer 1st op={tot_ox/1e6:.3f}ms) over {len(sd)} ops\n")
    o.write("# op2op_excl1 = op2op summed over all ops EXCEPT each layer's first op (which is perturbed by\n"
            "# the per-layer profiler-read drain). Use op2op_excl1 as the clean within-layer dispatch gap.\n#\n")
    o.write(f"{'layer':>5} {'type':>5} {'nops':>5} {'kernel_ms':>10} {'op2op_ms':>9} {'op2op_excl1':>12} "
            f"{'excl1%wall':>10}  first_op(op2op_us)\n")
    for L in layers:
        v = per[L]; typ = "dense" if L == 0 else "MoE"
        wall = v["kd"] + v["op2x"]
        pct = (100.0 * v["op2x"] / wall) if wall > 0 else 0.0
        o.write(f"{L:>5} {typ:>5} {v['n']:>5} {v['kd']/1e6:>10.3f} {v['op2']/1e6:>9.3f} {v['op2x']/1e6:>12.3f} "
                f"{pct:>9.1f}%  {v['foc'][:28]}({v['f']/1e3:.1f})\n")
    o.write("\n## grouped in 3s (kernel / op2op_excl1 ms):\n")
    for g0 in range(layers[0], layers[-1] + 1, 3):
        grp = [L for L in layers if g0 <= L < g0 + 3]
        if not grp: continue
        gk = sum(per[L]["kd"] for L in grp) / 1e6
        gox = sum(per[L]["op2x"] for L in grp) / 1e6
        o.write(f"  L{grp[0]:>2}-{grp[-1]:>2}: kernel={gk:8.3f}ms  op2op_excl1={gox:7.3f}ms\n")
print(f"wrote {out}: slowest dev {slow}, {len(layers)} layers, kernel {tot_k/1e6:.1f}ms "
      f"op2op {tot_o/1e6:.2f}ms (excl1 {tot_ox/1e6:.2f}ms)")
