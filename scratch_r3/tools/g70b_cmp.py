#!/usr/bin/env python3
"""Compare the g70b outputs of every variant: against the head's bit for bit, and against a float64 reference of the
dequantized inputs at the elements where the variant and the head differ. usage: g70b_cmp.py <out dir>"""
import glob, os, sys, torch
d = sys.argv[1]
head = {os.path.basename(f): f for f in glob.glob(os.path.join(d, "bits_head", "g70b_*.pt"))}
for n in sorted(head):
    inp = os.path.join(d, "bits_head", n.replace("g70b_", "in_").replace(".pt", ".ptin"))
    a, b = torch.load(inp)
    ref = (a.double() @ b.double())
    h = torch.load(head[n]).view(torch.bfloat16).double()
    for vd in sorted(glob.glob(os.path.join(d, "bits_*"))):
        v = os.path.basename(vd)[5:]
        f = os.path.join(vd, n)
        if v == "head" or not os.path.isfile(f):
            continue
        x = torch.load(f).view(torch.bfloat16).double()
        m = x != h
        k = int(m.sum())
        if k == 0:
            print(f"   {n} {v}: bit-identical to head")
            continue
        eh, ex = (h - ref).abs()[m], (x - ref).abs()[m]
        idx = m.nonzero()
        tiles = {(int(r[-2]) // 32, int(r[-1]) // 32) for r in idx}
        print(f"   {n} {v}: {k} elements differ in {len(tiles)} tiles; at them mean |err| head {eh.mean():.3e}, {v} {ex.mean():.3e};"
              f" {v} closer at {int((ex < eh).sum())}, head closer at {int((eh < ex).sum())}; first tiles {sorted(tiles)[:6]}")
