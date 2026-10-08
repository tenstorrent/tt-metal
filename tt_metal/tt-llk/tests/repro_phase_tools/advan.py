import csv
import glob
import os
import sys
from collections import Counter

D = sys.argv[1]


def L(n):
    out = {}
    for f in glob.glob(f"{D}/{n}/*.csv"):
        if f.endswith(".post.csv"):
            continue
        m = os.path.basename(f)[:-4]
        for r in csv.DictReader(open(f)):
            if r.get("marker") != "TILE_LOOP":
                continue
            key = (m,) + tuple(
                (k, v)
                for k, v in r.items()
                if not k.startswith(("mean(", "std(", "TEXT_SIZE")) and k != "marker"
            )
            for c, x in r.items():
                if c.startswith("mean(") and x:
                    out[key + (c,)] = float(x)
    return out


cache = {}


def get(n):
    if n not in cache:
        cache[n] = L(n)
    return cache[n]


def cmp(a, b, label):
    A, B = get(a), get(b)
    if not A or not B:
        print(f"{label:34s} (missing)")
        return
    ks = [k for k in A if k in B and A[k]]
    m05 = sum(abs(B[k] / A[k] - 1) > 0.005 for k in ks)
    m2 = [k for k in ks if abs(B[k] / A[k] - 1) > 0.02]
    mx = max((abs(B[k] / A[k] - 1) for k in ks), default=0)
    by = Counter((k[0], k[-1][5:-1]) for k in m2)
    print(
        f"{label:34s} values {len(ks):5d}  >0.5%: {m05:4d}  >2%: {len(m2):4d}  max {mx*100:5.1f}%  {dict(by.most_common(5))}"
    )


for p in sys.argv[2:]:
    l, ab = p.split("=", 1)
    a, b = ab.split(">")
    cmp(a, b, l)
