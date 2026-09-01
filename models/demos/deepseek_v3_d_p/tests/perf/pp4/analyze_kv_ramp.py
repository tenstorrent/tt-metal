"""How does a layer's cost grow with KV depth? Per-chunk op durations from one stage's capture.

Chunk c attends to KV[0:c*5120], so MLA/SDPA should grow with c while MoE (per-token work, blind to
KV depth) should stay flat. That is the falsifiable claim; this prints the evidence.

Instances are ordered by GLOBAL CALL COUNT within one device, which is program order = chunk order.
"""
import collections
import csv
import sys

path = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else ""
rows = [r for r in csv.DictReader(open(path)) if (r.get("DEVICE ID") or "").strip()]
dev = sorted({r["DEVICE ID"] for r in rows})[0]  # one device is representative
rows = [r for r in rows if r["DEVICE ID"] == dev]
rows.sort(key=lambda r: int(r["GLOBAL CALL COUNT"]))

WATCH = [
    "RingJointSDPADeviceOperation",
    "UnifiedRoutedExpertFfnDeviceOperation",
    "CombineDeviceOperation",
    "DispatchDeviceOperation",
    "MatmulDeviceOperation",
]
seq = collections.defaultdict(list)
for r in rows:
    op = r["OP CODE"].strip()
    if op in WATCH:
        seq[op].append(float(r.get("DEVICE KERNEL DURATION [ns]") or 0) / 1000.0)

n = len(seq.get("RingJointSDPADeviceOperation", []))
print(f"\n=== KV-depth ramp {label} (device {dev}, {n} passes; pass 0 is warm-up) ===")
for op in WATCH:
    v = seq.get(op, [])
    if not v:
        continue
    if op == "MatmulDeviceOperation" and n:  # ~10 matmuls per pass -> fold to per-pass total
        per = len(v) // n
        v = [sum(v[i * per : (i + 1) * per]) for i in range(n)]
    first, last = v[1] if len(v) > 1 else v[0], v[-1]
    growth = last / first if first else 0
    print(f"  {op:42s} {' '.join(f'{x:6.0f}' for x in v)}  us   growth {growth:4.2f}x")
