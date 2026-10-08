import csv
import glob
import os
import sys


def L(n):
    out = {}
    for f in glob.glob(f"/tmp/cq/{n}/*.csv"):
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


def show(n):
    for k, v in L(n).items():
        print(n, k[0], k[-1], v)


def cmp(a, b, thr=0.005, lim=15):
    A, B = L(a), L(b)
    ks = [k for k in A if k in B and A[k]]
    mv = [(B[k] / A[k] - 1, k) for k in ks if abs(B[k] / A[k] - 1) > thr]
    print(f"{a} -> {b}: values {len(ks)}, moved >{thr*100}%: {len(mv)}")
    for d, k in sorted(mv, key=lambda x: -abs(x[0]))[:lim]:
        cfg = " ".join(
            f"{x}={y}"
            for x, y in k[1:-1]
            if x
            in (
                "formats.input_A",
                "formats.output",
                "math_fidelity",
                "dest_acc",
                "dest_sync",
                "tile_cnt",
                "c_dimm",
                "r_dimm",
                "k_dimm",
                "throttle_level",
                "num_blocks",
            )
        )
        print(f"  {d*100:+6.1f}% {A[k]:9.0f} {B[k]:9.0f} {k[-1]} {cfg}")


for n in sys.argv[1:]:
    if ">" in n:
        a, b = n.split(">")
        cmp(a, b)
    else:
        show(n)
