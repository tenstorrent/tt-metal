"""Per-layer device-time budget from ops_perf_results CSVs.

Two corrections that matter, both easy to get wrong:

1. ONE ROW PER DEVICE. A stage spans 8 chips and the CSV has a row per (op instance, device). Those
   run CONCURRENTLY, so an op's cost is the MAX across devices, not the sum. Summing inflates
   everything 8x and silently turns a 12 ms layer into 96 ms.
2. SOCKET OPS ARE WAIT, NOT WORK. `InboundSocketServiceSyncOperation` is the receiver BLOCKING until
   upstream data arrives; it routinely shows ~99% of device time in a PP stage and is pure idle.
   Reported separately, never folded into the compute budget.

Op instances are keyed by GLOBAL CALL COUNT. The number of forward passes is taken from the count of
RingJointSDPA instances (exactly one per layer per chunk).

Usage: analyze_layer_budget.py <ops_perf_results.csv> [label]
"""
import collections
import csv
import sys

SOCKET = {"InboundSocketServiceSyncOperation", "OutboundSocketServiceSyncOperation"}
MLA = {
    "RingJointSDPADeviceOperation",
    "NlpCreateHeadsDeviceOperation",
    "NLPConcatHeadsDeviceOperation",
    "RotaryEmbeddingIndexedDeviceOperation",
    "UpdatePaddedKvCacheDeviceOperation",
    "ZeroPaddedKvCacheDeviceOperation",
    "SoftmaxDeviceOperation",
}
MOE = {
    "UnifiedRoutedExpertFfnDeviceOperation",
    "CombineDeviceOperation",
    "DispatchDeviceOperation",
    "PostCombineReduceDeviceOperation",
    "MaskedBincountDeviceOperation",
    "TopKDeviceOperation",
    "OffsetCumsumDeviceOperation",
}
NORM = {"LayerNormDeviceOperation"}

path = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else path.split("/")[-1]
rows = [r for r in csv.DictReader(open(path)) if (r.get("DEVICE ID") or "").strip()]

# max across devices per op instance
inst = collections.defaultdict(float)
opof = {}
for r in rows:
    op = (r.get("OP CODE") or "").strip()
    key = (op, r.get("GLOBAL CALL COUNT"))
    d = float(r.get("DEVICE KERNEL DURATION [ns]") or 0) / 1000.0
    if d > inst[key]:
        inst[key] = d
    opof[key] = op

by_op = collections.defaultdict(lambda: [0.0, 0])
for key, d in inst.items():
    by_op[opof[key]][0] += d
    by_op[opof[key]][1] += 1

npass = by_op.get("RingJointSDPADeviceOperation", [0, 0])[1] or 1
work = {k: v for k, v in by_op.items() if k not in SOCKET}
sock = {k: v for k, v in by_op.items() if k in SOCKET}
total = sum(v[0] for v in work.values())

print(f"\n=== {label} ===")
print(f"forward passes (layer x chunk): {npass}")
print(f"COMPUTE per layer per chunk (socket waits excluded): {total/npass/1000:.2f} ms")
for name, s in (("MLA/attn", MLA), ("MoE", MOE), ("norm", NORM)):
    t = sum(v[0] for k, v in work.items() if k in s)
    print(f"   {name:9s} {t/npass/1000:7.3f} ms  ({t/total*100:5.1f}%)")
other = sum(v[0] for k, v in work.items() if k not in MLA | MOE | NORM)
print(f"   {'other':9s} {other/npass/1000:7.3f} ms  ({other/total*100:5.1f}%)")
print("  top ops:")
for k, (t, n) in sorted(work.items(), key=lambda kv: -kv[1][0])[:8]:
    print(f"    {k:52s} {t/npass/1000:7.3f} ms  x{n/npass:.1f}/pass")
for k, (t, n) in sock.items():
    print(f"  [wait] {k:50s} {t/npass/1000:9.2f} ms  x{n/npass:.1f}/pass")
