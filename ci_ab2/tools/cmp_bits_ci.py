#!/usr/bin/env python3
# Raw-word comparison of each variant's dumps (<dir>/<variant>/<case>.npz) against main's. usage: cmp_bits_ci.py <dir>
import glob, os, sys
import numpy as np

d = sys.argv[1]


def load(v):
    return {os.path.basename(p)[:-4]: p for p in glob.glob(os.path.join(d, v, "*.npz"))}


def ndiff(x, y):
    if x.dtype != y.dtype or x.shape != y.shape:
        return x.size or 1
    s = x.dtype.itemsize
    a = np.ascontiguousarray(x).view(np.uint8).reshape(-1, s)
    b = np.ascontiguousarray(y).view(np.uint8).reshape(-1, s)
    return int((a != b).any(axis=1).sum())


m = load("main")
for v in ("main2", "q", "tc"):
    o = load(v)
    same = diff = words = 0
    bad = []
    for n in sorted(set(m) | set(o)):
        if n not in m or n not in o:
            bad.append(n + "(missing)")
            continue
        a, b = np.load(m[n]), np.load(o[n])
        nd = 0
        for k in a.files:
            if k not in b.files:
                nd += 1
                continue
            words += a[k].size
            nd += ndiff(a[k], b[k])
        if nd:
            diff += 1
            bad.append(f"{n}({nd})")
        else:
            same += 1
    print(f"== bits {v} against main: {same} cases identical, {diff} differ, {words} words; {' '.join(bad[:20])}")
