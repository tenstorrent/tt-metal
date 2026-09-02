#!/usr/bin/env python3
"""Diff two SFPU_EQUIV dumps bit-for-bit. Exit non-zero if any output bit differs."""
import glob
import os
import sys

import numpy as np

base_dir, a_tag, b_tag = sys.argv[1], sys.argv[2], sys.argv[3]
names = sorted(
    {
        os.path.basename(p).split("__", 1)[1][:-4]
        for p in glob.glob(f"{base_dir}/{a_tag}__*.npz")
    }
)
if not names:
    print(f"no dumps for tag {a_tag} in {base_dir}")
    sys.exit(2)

bad = 0
print(f"{'case':44s} {'points':>8s}  verdict")
print("-" * 72)
for n in names:
    pa, pb = f"{base_dir}/{a_tag}__{n}.npz", f"{base_dir}/{b_tag}__{n}.npz"
    if not os.path.exists(pb):
        print(f"{n:44s} {'-':>8s}  MISSING {b_tag}")
        bad += 1
        continue
    A, B = np.load(pa), np.load(pb)
    if not np.array_equal(A["x"], B["x"]):
        print(f"{n:44s} {'-':>8s}  STIMULUS MISMATCH")
        bad += 1
        continue
    ya, yb = A["y"], B["y"]
    if ya.shape != yb.shape:
        print(f"{n:44s} {'-':>8s}  SHAPE {ya.shape} vs {yb.shape}")
        bad += 1
        continue
    d = ya != yb
    ndiff = int(d.sum())
    if ndiff:
        idx = np.flatnonzero(d)[:5]
        ex = ", ".join(f"x={A['x'][i]:#x} {ya[i]:#x}->{yb[i]:#x}" for i in idx)
        print(f"{n:44s} {ya.size:8d}  DIFF {ndiff} ({ex})")
        bad += 1
    else:
        print(f"{n:44s} {ya.size:8d}  identical")
print("-" * 72)
print(f"{len(names)-bad}/{len(names)} cases bit-identical")
sys.exit(1 if bad else 0)
