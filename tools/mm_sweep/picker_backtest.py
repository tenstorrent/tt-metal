#!/usr/bin/env python3
# Backtest picker strategies against the FLUX/LTX Regime-A sweep. A picker RANKS the measured feasible
# configs for a shape and selects one; regret = picked_bw / best_bw. Compares: table (oracle), naive
# max-K-split, and a closed-form cost-model scorer (enumerate feasible + score, what a program factory runs).
import json, math
from collections import defaultdict

d = json.load(open("fluxltx_regimeA_sweep.json"))


def cdiv(a, b):
    return (a + b - 1) // b


def rup(x, y):
    return ((x + y - 1) // y) * y


by = defaultdict(dict)
best = defaultdict(float)
bestcfg = {}
for r in d:
    k = (r["M"], r["K"], r["N"])
    c = (r["Ns"], r["Pk"], r["Sm"], r["kb"], r["nsb"])
    by[k][c] = r["bwp"]
    if r["bwp"] > best[k]:
        best[k] = r["bwp"]
        bestcfg[k] = c
shapes = list(by.keys())


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def derived(M, K, N, c):
    Ns, Pk, Sm, kb, nsb = c
    Mt, Kt, Nt = M // 32, K // 32, N // 32
    cores = 8 * Pk * Ns * Sm
    Ktl = rup(cdiv(Kt, Pk), kb * 8)
    Mblk = cdiv(Mt, Sm)
    Nband = cdiv(Nt, 8)
    Nown = cdiv(Nband, Ns)
    Nbpc = cdiv(Nown, nsb)
    return dict(cores=cores, Ktl=Ktl, Mblk=Mblk, Nown=Nown, Nbpc=Nbpc, Pk=Pk, Ns=Ns, nsb=nsb, Kt=Kt, Nt=Nt, Mt=Mt)


# ---- cost model: predict relative time (lower=better). params tuned below. ----
def cost(M, K, N, c, P):
    g = derived(M, K, N, c)
    Ns, Pk, Sm, kb, nsb = c
    readT = g["Kt"] * g["Nt"] / min(g["cores"], P["Csat"])  # in1 read, amortized, saturates
    comp_pc = g["Mblk"] * g["Nown"] * g["Ktl"]  # tile-MACs per core
    area = min(g["Mblk"] * nsb, P["acap"])  # sub-block area, benefit caps at acap
    eff_kb = kb / (kb + P["kk"])  # deeper k-block -> better unpack
    eff_area = area / (area + P["aa"])  # bigger sub-block -> better reuse
    compT = comp_pc / (eff_kb * eff_area)
    ovlT = P["ovl"] * comp_pc / g["Nbpc"]  # few sub-blocks -> poor read/compute overlap
    redT = P["red"] * g["Pk"] * g["Mblk"] * g["Nown"]  # reduction chain depth * out
    fwdT = P["fwd"] * g["Mblk"] * g["Pk"]  # in0 all-gather forward
    startT = P["start"] * g["Ktl"]  # gather startup stall
    return max(readT, compT) + ovlT + redT + fwdT + startT


def pick_cost(shape, P):
    M, K, N = shape
    cfgs = [c for c in by[shape] if c[2] == 1]  # Sm=1 forced (oracle 20/20)
    return min(cfgs, key=lambda c: cost(M, K, N, c, P))


def eval_P(P):
    regs = [by[s][pick_cost(s, P)] / best[s] for s in shapes]
    return geomean(regs), min(regs), regs


# coarse random-ish grid search over the handful of constants
import itertools

bestP = None
bestg = 0
grid = dict(
    Csat=[24, 32, 48],
    kk=[0.5, 1, 2],
    aa=[1, 2, 4],
    acap=[4, 6, 8, 12],
    ovl=[0.0, 0.1, 0.3, 0.6, 1.0],
    red=[0.0, 0.05, 0.1],
    fwd=[0.0, 0.05, 0.1],
    start=[0.0, 0.02, 0.05],
)
keys = list(grid)
import random

random.seed = None
# full product is large; sample
combos = list(itertools.product(*[grid[k] for k in keys]))
print(f"searching {len(combos)} param combos...")
for vals in combos:
    P = dict(zip(keys, vals))
    g, _, _ = eval_P(P)
    if g > bestg:
        bestg = g
        bestP = P
gm, worst, regs = eval_P(bestP)
print(f"\nBEST cost-model params: {bestP}")
print(f"  geomean regret = {gm*100:.1f}%   worst = {worst*100:.0f}%")
print("\n  per-shape:")
order = [
    (32, 2048, 512),
    (32, 2048, 1536),
    (32, 6144, 1536),
    (32, 2048, 2048),
    (32, 6144, 2304),
    (32, 6144, 3072),
    (32, 256, 6144),
    (32, 6144, 6144),
    (32, 6144, 9216),
    (64, 6144, 1536),
    (64, 15360, 1536),
    (64, 6144, 4608),
    (64, 4608, 6144),
    (64, 6144, 9216),
    (128, 6144, 768),
    (128, 15360, 768),
    (128, 6144, 2304),
    (128, 6144, 4608),
    (128, 2304, 6144),
    (512, 6144, 1536),
]
for s in order:
    p = pick_cost(s, bestP)
    got = by[s][p]
    b = best[s]
    bc = bestcfg[s]
    fl = "" if got / b >= 0.97 else ("  <<" if got / b < 0.9 else "  <")
    print(f"  {s[0]}x{s[1]}x{s[2]:>5}: pick{p} {got:.0f}%  | oracle{bc} {b:.0f}%  = {got/b*100:.0f}%{fl}")
