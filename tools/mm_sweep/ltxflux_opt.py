#!/usr/bin/env python3
# LTX/FLUX Mt<=8 cumulative optimization campaign harness.
# - enumerate_feasible(M,K,N): the complete feasible (Ns,Pk,Sm,kb,nsb) space (mirrors device pick_plan()).
# - sweep(M,K,N): measure every feasible config through the REAL production kernel (diag entry, mask 0,
#   forced config) via regime_a_diag_suite.run_one; record wall_us / GB/s / per-RISC; rank by wall_us.
# Perf is value-independent so constant-input measurement is valid; random-operand PCC + config=None are
# validated separately (pytest suite). This harness NEVER reduces work or manipulates timing.
#
# Usage: python ltxflux_opt.py sweep M K N        (sweep one shape, write ltxflux_sweep_MxKxN.json)
#        python ltxflux_opt.py feasible M K N      (list feasible configs only)
import json, os, sys, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)
import regime_a_bench as rb
import regime_a_diag_suite as ds

# Complete planner-supported ranges (deep-investigation mode): Pk 1..12, Ns 1..6, Sm 1..Mt, kb {1,2,4,8}.
PK = list(range(1, 13))
NS = list(range(1, 7))
KB = [1, 2, 4, 8]
kTB = 2048
kL1 = 1440 * 1024


def cdiv(a, b):
    return (a + b - 1) // b


def rup(x, y):
    return cdiv(x, y) * y


def feasible_geo(Mt, Kt, Nt, Ns, Pk, Sm, kb, nsb):
    # Mirror device pick_plan() + build_plan() feasibility (config.cpp + plan.hpp).
    if Sm > Mt or Pk > Kt:  # build_plan rejects empty m-/k-slices
        return None
    cores = 8 * Pk * Ns * Sm
    if not (16 <= cores <= 104):
        return None
    Ktl = rup(cdiv(Kt, Pk), kb * 8)
    if (Pk * Ktl) / Kt - 1.0 > 0.20:
        return None
    Mblk = cdiv(Mt, Sm)
    Nband = cdiv(Nt, 8)
    Nown = cdiv(Nband, Ns)
    if nsb > Nown:
        return None
    Nbpc = cdiv(Nown, nsb)
    if (8 * Ns * Nbpc * nsb) / Nt - 1.0 > 0.20:
        return None
    cb0 = Ktl * Mblk * kTB
    cb1 = 4 * kb * nsb * kTB
    cb2 = 2 * Mblk * nsb * kTB
    cb3 = Mblk * nsb * 4096
    cb7 = 2 * Mblk * nsb * kTB
    if cb0 + cb1 + cb2 + cb3 + cb7 > kL1:
        return None
    W = (Ktl // kb) // 8
    return {"cores": cores, "Ktl": Ktl, "Mblk": Mblk, "Nband": Nband, "Nown": Nown, "Nbpc": Nbpc, "W": W}


# Broad nsb candidate set for wide-N shapes (Nown can be up to ~24). Exhaustive when Nown is small (all
# 1..Nown are <= these); a broad sample + the full-width value when large. Refine around the winner after.
_NSB_BROAD = {1, 2, 3, 4, 6, 8, 9, 12, 16}


def enumerate_feasible(M, K, N, nsb_set=None):
    Mt, Kt, Nt = M // 32, cdiv(K, 32), cdiv(N, 32)
    out = []
    for Pk in PK:
        for Ns in NS:
            for Sm in range(1, Mt + 1):  # Sm = 1..Mt (complete planner range)
                for kb in KB:
                    Nband = cdiv(Nt, 8)
                    Nown = cdiv(Nband, max(Ns, 1))
                    if nsb_set is not None:
                        cands = sorted(n for n in nsb_set if 1 <= n <= Nown)
                    elif Nown <= 10:
                        cands = list(range(1, Nown + 1))  # exhaustive when small
                    else:
                        cands = sorted({n for n in _NSB_BROAD if n <= Nown} | {Nown})  # broad + full-width
                    for nsb in cands:
                        g = feasible_geo(Mt, Kt, Nt, Ns, Pk, Sm, kb, nsb)
                        if g:
                            out.append(((Ns, Pk, Sm, kb, nsb), g))
    return out


def sweep(M, K, N, iters=8):
    feas = enumerate_feasible(M, K, N)
    print(f"[{M}x{K}x{N}] {len(feas)} feasible configs", flush=True)
    auto = tuple(rb.auto_config(M, K, N))
    results = []
    for i, (cfg, g) in enumerate(feas):
        r = ds.run_one(M, K, N, cfg, 0, iters=iters)
        w = r.get("wall_us")
        gb = rb.logical_bytes(M, K, N) / (w / 1e6) / 1e9 if w else None
        pr = r.get("per_risc_us")
        rec = {"cfg": list(cfg), "W": g["W"], "cores": g["cores"], "wall_us": w, "gbps": gb,
               "per_risc": pr, "spread": r.get("spread"), "ok": r.get("ok"), "is_auto": cfg == auto}
        results.append(rec)
        tag = " <AUTO" if cfg == cfg and cfg == auto else ""
        print(f"  [{i+1}/{len(feas)}] cfg={cfg} W{g['W']} c{g['cores']} wall={None if w is None else round(w,2)}us "
              f"gb={None if gb is None else round(gb)}{tag}", flush=True)
        json.dump({"shape": [M, K, N], "auto": list(auto), "results": results},
                  open(f"{HERE}/ltxflux_sweep_{M}x{K}x{N}.json", "w"), indent=2)
    oks = [r for r in results if r["ok"] and r["wall_us"]]
    oks.sort(key=lambda r: r["wall_us"])
    print(f"\n=== TOP 8 (of {len(oks)} ok) for {M}x{K}x{N} (auto={auto}) ===", flush=True)
    for r in oks[:8]:
        a = " <AUTO" if r["is_auto"] else ""
        print(f"  cfg={tuple(r['cfg'])} W{r['W']} wall={r['wall_us']:.2f}us gb={r['gbps']:.0f}{a}", flush=True)
    autor = next((r for r in results if r["is_auto"]), None)
    if autor and oks and autor["wall_us"]:
        best = oks[0]
        gain = (autor["wall_us"] / best["wall_us"] - 1) * 100
        print(f"  AUTO wall={autor['wall_us']:.2f}us  BEST wall={best['wall_us']:.2f}us  "
              f"headroom={gain:.1f}%  best_cfg={tuple(best['cfg'])}", flush=True)
    return results


if __name__ == "__main__":
    mode = sys.argv[1]
    M, K, N = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    if mode == "feasible":
        for cfg, g in enumerate_feasible(M, K, N):
            print(cfg, g)
    elif mode == "sweep":
        sweep(M, K, N)
