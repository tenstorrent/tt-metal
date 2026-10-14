#!/usr/bin/env python3
"""Where a variant's g70b output differs from the head's, are all differing elements inside the tiles the float64
reference flags as wrong (above 10x the median tile error) on both sides? usage: gold_diff.py <out dir> <variant>"""
import glob, os, sys, torch
d, v = sys.argv[1], sys.argv[2]
for hf in sorted(glob.glob(os.path.join(d, "bits_head", "g70b_*.pt"))):
    n = os.path.basename(hf)
    vf = os.path.join(d, f"bits_{v}", n)
    if not os.path.isfile(vf):
        continue
    h, x = torch.load(hf), torch.load(vf)
    m = h != x
    if not bool(m.any()):
        continue
    tag = n[:-3]
    bh = set(torch.load(os.path.join(d, "bits_head", f"badtiles_{tag}.pt")).tolist())
    bv = set(torch.load(os.path.join(d, f"bits_{v}", f"badtiles_{tag}.pt")).tolist())
    rows, cols = h.shape[-2] // 32, h.shape[-1] // 32
    idx = m.nonzero()
    tiles = set()
    for r in idx.tolist():
        lead = 0
        for k, s in zip(r[:-2], h.shape[:-2]):
            lead = lead * s + k
        tiles.add(lead * rows * cols + (r[-2] // 32) * cols + r[-1] // 32)
    print(f"   {n}: {int(m.sum())} elements differ in {len(tiles)} tiles; outside the head's wrong tiles: {len(tiles - bh)}, "
          f"outside {v}'s: {len(tiles - bv)}; wrong tiles head {len(bh)}, {v} {len(bv)}, same set {bh == bv}")
