#!/usr/bin/env python3
# Totals for the WARM pass on the SLOWEST device: total op-kernel time and total op-to-op gap.
# The op2op total is reported two ways: raw (signed) and POSITIVE-ONLY (negatives clamped to 0).
# Negative op2op = cross-op device overlap (e.g. the MoE DispatchDeviceOperation runs on a disjoint
# core set and the dispatcher front-runs it, so its start precedes the prior op's end -> negative gap).
# Those negatives are overlap, not idle, so the positive-only sum is the honest "dispatch idle" total.
# Warm pass = the LAST occurrence of each layer's forward_chunk_layer_{i}_start signpost (compile pass
# excluded). Usage: op_totals.py <ops_csv> [label]
import csv, sys, re
from collections import defaultdict

ops_csv = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else ops_csv


def fnum(s):
    try:
        return float(s)
    except Exception:
        return 0.0


rows = []
start_sig = defaultdict(list)
end_sig = defaultdict(list)
re_s = re.compile(r"^forward_chunk_layer_(\d+)_start$")
re_e = re.compile(r"^forward_chunk_layer_(\d+)_end$")
with open(ops_csv) as f:
    for i, r in enumerate(csv.DictReader(f)):
        oc = (r.get("OP CODE") or "").strip()
        ms = re_s.match(oc)
        me = re_e.match(oc)
        if ms:
            start_sig[int(ms.group(1))].append(i)
            continue
        if me:
            end_sig[int(me.group(1))].append(i)
            continue
        try:
            dev = int(r["DEVICE ID"])
        except Exception:
            continue
        if not oc:
            continue
        rows.append((i, oc, dev, fnum(r.get("DEVICE KERNEL DURATION [ns]")), fnum(r.get("OP TO OP LATENCY [ns]"))))

layers = sorted(start_sig)
if not layers:
    sys.stderr.write("no forward_chunk_layer_* signposts found\n")
    sys.exit(1)
warm_start = {L: max(start_sig[L]) for L in layers}
lo = warm_start[layers[0]]
hi = max(end_sig[layers[-1]]) if layers[-1] in end_sig else max(x[0] for x in rows)
warm = [x for x in rows if lo <= x[0] <= hi]

ks = defaultdict(float)
for i, oc, dev, kd, o2 in warm:
    ks[dev] += kd
slow = max(ks, key=ks.get)
sd = [x for x in warm if x[2] == slow]

n = len(sd)
kernel = sum(x[3] for x in sd)
o2_raw = sum(x[4] for x in sd)
o2_pos = sum(x[4] for x in sd if x[4] > 0)
n_neg = sum(1 for x in sd if x[4] < 0)
neg_mass = sum(x[4] for x in sd if x[4] < 0)

print(f"# TOTALS — {label}")
print(f"#   slowest DEVICE ID = {slow}; layers = {len(layers)} ({layers[0]}..{layers[-1]}); warm ops = {n}")
print(f"  total_op_kernel_time   = {kernel/1e6:10.3f} ms")
print(f"  total_op2op_gap_raw    = {o2_raw/1e6:10.3f} ms   (signed; includes negative overlap)")
print(f"  total_op2op_gap_posonly= {o2_pos/1e6:10.3f} ms   (negatives clamped to 0 -> honest dispatch idle)")
print(f"  (#ops with negative op2op = {n_neg}, summing {neg_mass/1e6:.3f} ms of overlap discounted)")
print(f"  op2op_posonly % of kernel = {100.0*o2_pos/kernel if kernel else 0:.1f}%")
