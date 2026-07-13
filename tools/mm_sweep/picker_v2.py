#!/usr/bin/env python3
# Cost-model v2: adds (a) kb-benefit cap at 4 (training: kb8 won 0/20, kb4 sweet spot), (b) padding-waste
# penalty (v1 had none -> picked 20%-waste configs). Backtest v2 vs v1 on the 3262-config oracle (real
# measurements, device-free). Then show how v2 changes the flagged validation picks.
import json, math, itertools
from collections import defaultdict


def cdiv(a, b):
    return (a + b - 1) // b


def rup(x, y):
    return ((x + y - 1) // y) * y


L1BUDGET = 1440 * 1024
TB = 2048


def plan(M, K, N, Ns, Pk, Sm, kb, nsb):
    Mt, Kt, Nt = M // 32, K // 32, N // 32
    cores = 8 * Pk * Ns * Sm
    if not (16 <= cores <= 104):
        return None
    Ktl = rup(cdiv(Kt, Pk), kb * 8)
    wasteK = Pk * Ktl / Kt - 1
    if wasteK > 0.20:
        return None
    Mblk = cdiv(Mt, Sm)
    Nband = cdiv(Nt, 8)
    Nown = cdiv(Nband, Ns)
    if nsb > Nown:
        return None
    Nbpc = cdiv(Nown, nsb)
    wasteN = 8 * Ns * Nbpc * nsb / Nt - 1
    if wasteN > 0.20:
        return None
    cb0 = Ktl * Mblk * TB
    cb1 = 4 * kb * nsb * TB
    cb2 = 2 * Mblk * nsb * TB
    cb3 = Mblk * nsb * 4096
    cb7 = 2 * Mblk * nsb * TB
    if cb0 + cb1 + cb2 + cb3 + cb7 > L1BUDGET:
        return None
    return dict(cores=cores, Ktl=Ktl, Mblk=Mblk, Nown=Nown, Nbpc=Nbpc, wasteK=wasteK, wasteN=wasteN)


def cost(M, K, N, c, P):
    Ns, Pk, Sm, kb, nsb = c
    g = plan(M, K, N, *c)
    Kt, Nt = K // 32, N // 32
    readT = Kt * Nt / min(g["cores"], P["Csat"])
    comp_pc = g["Mblk"] * g["Nown"] * g["Ktl"]
    area = min(g["Mblk"] * nsb, P["acap"])
    kbe = min(kb, P["kbcap"])  # v2: cap kb benefit (kb>4 no extra, only costs)
    compT = comp_pc / ((kbe / (kbe + P["kk"])) * (area / (area + P["aa"])))
    ovlT = P["ovl"] * comp_pc / g["Nbpc"]
    base = max(readT, compT) + ovlT + P["start"] * g["Ktl"]
    return base * (1 + P["wst"] * (g["wasteK"] + g["wasteN"]))  # v2: padding-waste penalty


# ---- backtest on the 3262-config oracle ----
d = json.load(open("fluxltx_regimeA_sweep.json"))
by = defaultdict(dict)
best = defaultdict(float)
for r in d:
    k = (r["M"], r["K"], r["N"])
    by[k][(r["Ns"], r["Pk"], r["Sm"], r["kb"], r["nsb"])] = r["bwp"]
    best[k] = max(best[k], r["bwp"])
shapes = list(by.keys())


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def pick(shape, P):
    M, K, N = shape
    cfgs = [c for c in by[shape] if c[2] == 1]
    return min(cfgs, key=lambda c: cost(M, K, N, c, P))


def evalP(P):
    regs = [by[s][pick(s, P)] / best[s] for s in shapes]
    return geomean(regs), min(regs)


V1 = dict(Csat=24, kk=0.5, aa=2, acap=6, kbcap=99, ovl=1.0, start=0.0, wst=0.0)  # v1 = no cap, no waste pen
# search v2 params (kbcap + wst added)
grid = dict(
    Csat=[24, 32],
    kk=[0.5, 1],
    aa=[2],
    acap=[6, 8],
    kbcap=[2, 4],
    ovl=[0.6, 1.0],
    start=[0.0, 0.02],
    wst=[0.5, 1.0, 2.0, 4.0],
)
keys = list(grid)
bestP = None
bg = 0
for vals in itertools.product(*[grid[k] for k in keys]):
    P = dict(zip(keys, vals))
    g, _ = evalP(P)
    if g > bg:
        bg = g
        bestP = P
g1, w1 = evalP(V1)
g2, w2 = evalP(bestP)
print(f"v1 (no cap/no-waste): geomean {g1*100:.1f}%  worst {w1*100:.0f}%")
print(f"v2 (kb-cap + waste) : geomean {g2*100:.1f}%  worst {w2*100:.0f}%   params={bestP}")

# ---- how v2 changes the flagged validation outliers (offline pick only; device confirmation pending fw) ----
print("\n=== v2 re-picks on flagged validation outliers (was device-blocked for measurement) ===")


def feasible(M, K, N):
    Nt = N // 32
    out = []
    for Pk in range(1, 13):
        for Ns in range(1, 7):
            Nown = cdiv(cdiv(Nt, 8), Ns)
            for kb in (1, 2, 4, 8):
                for nsb in range(1, Nown + 1):
                    if plan(M, K, N, Ns, Pk, 1, kb, nsb):
                        out.append((Ns, Pk, 1, kb, nsb))
    return out


def pickfree(M, K, N, P):
    fs = feasible(M, K, N)
    return min(fs, key=lambda c: cost(M, K, N, c, P))


OUT = [
    (128, 6144, 2560),
    (128, 6144, 12288),
    (128, 15360, 1536),
    (256, 6144, 6144),
    (256, 5120, 6144),
    (256, 15360, 1536),
    (512, 4096, 1024),
    (32, 20480, 1536),
]
for M, K, N in OUT:
    p1 = pickfree(M, K, N, V1)
    p2 = pickfree(M, K, N, bestP)
    g1p = plan(M, K, N, *p1)
    g2p = plan(M, K, N, *p2)
    ch = "SAME" if p1 == p2 else "CHANGED"
    print(
        f"  {M}x{K}x{N}: v1 {p1[:2]+p1[3:]} wK{g1p['wasteK']:.0%}wN{g1p['wasteN']:.0%}  ->  v2 {p2[:2]+p2[3:]} wK{g2p['wasteK']:.0%}wN{g2p['wasteN']:.0%}  [{ch}]"
    )
