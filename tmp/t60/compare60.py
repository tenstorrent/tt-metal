# #60 decode A/B check: every arm's yuv output must equal the first arm's; prints min decode times and deltas.
# Usage (on blx03): python compare60.py [dir=/var/tmp/fasth3/t60] [extra yuv .pt files to compare, e.g. #44's]
import glob
import os
import re
import sys

import torch

V = sys.argv[1] if len(sys.argv) > 1 else "/var/tmp/fasth3/t60"
outs = sorted(glob.glob(f"{V}/yuv_t*w*.pt")) + sys.argv[2:]
times = {}
for f in sorted(glob.glob(f"{V}/run60_*.log")):
    for line in open(f):
        m = re.search(r"AB arm=(t\dw\d) decode_s=.* min=([\d.]+)", line)
        if m:
            times[m.group(1)] = float(m.group(2))
ok = len(outs) >= 2
ref = torch.load(outs[0]) if outs else None
for f in outs[1:]:
    t = torch.load(f)
    same = t.shape == ref.shape and torch.equal(t, ref)
    diff = (t.int() - ref.int()).abs().max().item() if t.shape == ref.shape else "shape"
    print(f"T60 {os.path.basename(f)} vs {os.path.basename(outs[0])}: identical={same} max_abs_diff={diff}")
    ok &= same
for arm, s in sorted(times.items()):
    print(f"T60 arm={arm} min_decode_s={s:.4f}")
for a, b in (("t0w0", "t1w0"), ("t1w0", "t1w1"), ("t0w0", "t1w1"), ("t0w0", "t0w1")):
    if a in times and b in times:
        print(f"T60 saving {a}->{b}: {1000 * (times[a] - times[b]):.1f} ms")
print(f"T60 PASS={ok}")
sys.exit(0 if ok else 1)
