# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
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

WATCH = [
    "RingJointSDPADeviceOperation",
    "UnifiedRoutedExpertFfnDeviceOperation",
    "CombineDeviceOperation",
    "DispatchDeviceOperation",
    "MatmulDeviceOperation",
]

# One streaming pass, keeping only the watched ops' durations per device: the raw ops CSVs carry 457
# columns, so materialising every row as a dict costs hundreds of MB for a read used once. Unlike the
# layer budget, this reports ONE device rather than the max across devices -- see the module docstring.
per_dev = collections.defaultdict(lambda: collections.defaultdict(list))
with open(path) as fh:
    for r in csv.DictReader(fh):
        device = (r.get("DEVICE ID") or "").strip()
        op = (r.get("OP CODE") or "").strip()
        if not device or op not in WATCH:
            continue
        call = int(r["GLOBAL CALL COUNT"])
        dur = float(r.get("DEVICE KERNEL DURATION [ns]") or 0) / 1000.0
        per_dev[device][op].append((call, dur))

dev = sorted(per_dev)[0]  # one device is representative
# Program order is call order, which is chunk order.
seq = {op: [d for _, d in sorted(v)] for op, v in per_dev[dev].items()}

n = len(seq.get("RingJointSDPADeviceOperation", []))
print(f"\n=== KV-depth ramp {label} (device {dev}, {n} passes; pass 0 is warm-up) ===")
for op in WATCH:
    v = seq.get(op, [])
    if not v:
        continue
    if op == "MatmulDeviceOperation" and n:  # ~10 matmuls per pass -> fold to per-pass total
        per = len(v) // n
        v = [sum(v[i * per : (i + 1) * per]) for i in range(n)]
    # Skip pass 0: it is warm-up, so its duration is not on the ramp.
    first = v[1] if len(v) > 1 else v[0]
    growth = v[-1] / first if first else 0
    print(f"  {op:42s} {' '.join(f'{x:6.0f}' for x in v)}  us   growth {growth:4.2f}x")
