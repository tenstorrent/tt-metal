#!/usr/bin/env python3
"""Compare the bits sweep outputs of two variants: same inputs, every output datum bit for bit.
usage: bits_cmp.py <dir> <tag a> <tag b>"""
import glob, os, sys
import torch
d, a, b = sys.argv[1:4]
fa = sorted(glob.glob(os.path.join(d, a, "*.pt")))
n = same = diff_files = 0
for f in fa:
    g = os.path.join(d, b, os.path.basename(f))
    if not os.path.exists(g):
        print("missing in", b, os.path.basename(f)); diff_files += 1; continue
    x, y = torch.load(f), torch.load(g)
    assert torch.equal(x["in"], y["in"]), f
    eq = (x["out"] == y["out"])
    n += eq.numel(); same += int(eq.sum())
    if not bool(eq.all()):
        diff_files += 1
        if diff_files <= 5:
            idx = torch.nonzero(~eq)[:3].tolist()
            print("DIFF", os.path.basename(f), int((~eq).sum()), "datums, first", idx)
print(f"== bits {a} against {b}: {len(fa)} runs, {n} output datums, identical {same}, files differing {diff_files}")
