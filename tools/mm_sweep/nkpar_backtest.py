#!/usr/bin/env python3
# Back-test a joint (S, Pk) heuristic against a measured combo matrix from test_flux_sweep.
# Usage: nkpar_backtest.py <run_dir> <manifest.json> [grid_y] [grid_x]
#   run_dir = the tracy -o dir (reads <run_dir>/.logs/cpp_device_perf_report.csv)
import csv, json, math, statistics, sys, os

RUNDIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/flux_runs"
MAN = sys.argv[2] if len(sys.argv) > 2 else "/tmp/flux_manifest.json"
CSV = RUNDIR + "/.logs/cpp_device_perf_report.csv"
G = int(sys.argv[3]) if len(sys.argv) > 3 else 8  # grid.y (row budget): 8 on WH, 10 on BH
GX = int(sys.argv[4]) if len(sys.argv) > 4 else 8  # grid.x
REPS = int(os.environ.get("FC_REPS", "20"))
WARMUP = 3
DROP, CHUNK = 1 + WARMUP, 1 + WARMUP + REPS

man = json.load(open(MAN))
rows = [r for r in csv.DictReader(open(CSV)) if r.get("DEVICE KERNEL DURATION [ns]", "").strip()]
rows.sort(key=lambda r: int(r["GLOBAL CALL COUNT"]))
succ = [m for m in man if m["ok"]]
assert len(rows) == CHUNK * len(succ), (len(rows), CHUNK * len(succ))
i = 0
for m in succ:
    chunk = rows[i : i + CHUNK]
    i += CHUNK
    m["us"] = statistics.median(float(r["DEVICE KERNEL DURATION [ns]"]) for r in chunk[DROP:]) / 1000.0

# per-shape: {(S,Pk): us}, plus auto
shapes = {}
for m in man:
    key = (m["M"], m["K"], m["N"])
    d = shapes.setdefault(key, {"Mt": m["Mt"], "Nt": m["Nt"], "Kt": m["Kt"], "out": m["out_tiles"], "t": {}})
    if m["ok"]:
        d["t"][(m["S"], m["Pk"])] = m["us"]


def pow2_le(x):
    p = 1
    while p * 2 <= x:
        p *= 2
    return max(1, p)


def heuristic(Mt, Nt, Kt, out, P=(224, 96, 20, 256, 8)):
    t8, t4, t2, nwide, capPk = P
    small, big = min(Mt, Nt), max(Mt, Nt)
    if small > 2 * G:
        return (1, 1)
    D = Kt * float(G * GX) / max(1, out)  # K-dominance: deep reduction vs output saturation of the grid
    Pk = 8 if D >= t8 else 4 if D >= t4 else 2 if D >= t2 else 1
    Pk = min(Pk, capPk)
    if Nt >= nwide:  # wide-N: in1 DRAM-bandwidth bound, K-par reduction regresses
        Pk = 1
    while Pk > 1 and (G % Pk != 0 or Kt % Pk != 0 or Kt // Pk < 8):  # fit row budget + deep-K
        Pk //= 2
    return (G // Pk, Pk)


def lookup(d, S, Pk):
    t = d["t"]
    if Pk == 1:
        # auto row (S label is 'auto') or explicit S w/ Pk1 (not measured) -> use auto
        return t.get(("auto", 1))
    return t.get((str(S), Pk))


def evaluate(P):
    sp, regress, regs = [], 0, []
    for key, d in shapes.items():
        auto = d["t"].get(("auto", 1))
        if auto is None:
            continue
        best = min(d["t"].values())
        S, Pk = heuristic(d["Mt"], d["Nt"], d["Kt"], d["out"], P)
        pt = lookup(d, S, Pk)
        if pt is None:
            pt = auto
        sp.append(auto / pt)
        regs.append(pt / best)
        if auto / pt < 0.995:
            regress += 1
    g = math.exp(sum(math.log(x) for x in sp) / len(sp))
    return g, regress, max(regs), sum(regs) / len(regs)


# Grid-search thresholds: hard constraint no regressions, then maximize geomean speedup.
best = None
for t8 in (999, 280, 240):
    for t4 in (40, 56, 72, 96):
        for t2 in (12, 16, 20, 28):
            for nwide in (192, 256, 99999):
                for capPk in (4, 8):
                    P = (t8, t4, t2, nwide, capPk)
                    g, rg, mx, mn = evaluate(P)
                    score = (rg, -round(g, 4), round(mx, 3))
                    if best is None or score < best[0]:
                        best = (score, P, g, rg, mx, mn)
BP = best[1]
print(
    f"BEST PARAMS (t8,t4,t2,nwide,capPk)={BP}  geomean={best[2]:.3f}x regressions={best[3]} maxregret={best[4]:.3f} meanregret={best[5]:.3f}\n"
)

print(
    f"{'shape':>16} {'out':>5}{'Kt':>5} | {'auto us':>8} {'best(S,Pk)':>11}{'best us':>8} | "
    f"{'pred(S,Pk)':>11}{'pred us':>8} {'pred/auto':>9} {'regret':>7}"
)
import math as _m

sp_pred, sp_best, regrets, regress = [], [], [], []
for key, d in shapes.items():
    M, K, N = key
    auto = d["t"].get(("auto", 1))
    if auto is None:
        continue
    best_combo = min(d["t"], key=lambda c: d["t"][c])
    best = d["t"][best_combo]
    S, Pk = heuristic(d["Mt"], d["Nt"], d["Kt"], d["out"], BP)
    pt = lookup(d, S, Pk)
    if pt is None:  # heuristic picked an unmeasured combo -> safe fallback to auto
        S, Pk, pt = "auto", 1, auto
    pred_sp = auto / pt
    regret = pt / best
    sp_pred.append(pred_sp)
    sp_best.append(auto / best)
    regrets.append(regret)
    if pred_sp < 0.995:
        regress.append((f"{M}x{K}x{N}", round(pred_sp, 3)))
    print(
        f"{M}x{K}x{N:>16}"[:16].rjust(16) + f" {d['out']:>5}{d['Kt']:>5} | {auto:>8.1f} "
        f"{str(best_combo):>11}{best:>8.1f} | {str((S,Pk)):>11}{pt:>8.1f} {pred_sp:>8.2f}x {regret:>6.2f}"
    )
geo = lambda xs: _m.exp(sum(_m.log(x) for x in xs) / len(xs))
print(f"\nshapes={len(sp_pred)}  geomean speedup: heuristic={geo(sp_pred):.3f}x  oracle-best={geo(sp_best):.3f}x")
print(
    f"heuristic captures {geo(sp_pred)/geo(sp_best)*100:.1f}% of oracle (geomean)   mean regret={statistics.mean(regrets):.3f}  max regret={max(regrets):.3f}"
)
print(f"regressions (<0.995x vs auto): {regress if regress else 'NONE'}")
