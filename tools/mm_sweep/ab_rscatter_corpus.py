#!/usr/bin/env python3
# Reduce-scatter across the full ~60-shape Mt<=8 corpus at the DEPLOYED picker config. For each shape we take
# the picker's (Ns,Pk,Sm,kb,nsb) from regime_a_current_perf.json and, where reduce-scatter is FEASIBLE
# (Pk>1, N_bpc==1, T=M_block*N_sub divisible by Pk and >=Pk), A/B the CHAIN (mask 0) vs reduce-scatter
# (mask 128) at that exact config. Where infeasible, reduce-scatter cannot be enabled at the picker config
# (falls back to chain -> 0 speedup) and we record the reason. Hang-safe run_one; N relaunches, median.
import json, os, statistics, sys

sys.path.insert(0, os.path.dirname(__file__))
import regime_a_diag_suite as ds

N_RELAUNCH = 6
ITERS = 8
CHAIN, RS = 256, 128  # chain baseline uses DIAG_FORCE_CHAIN (mask 0 now selects reduce-scatter for gated shapes)


def cdiv(a, b):
    return (a + b - 1) // b


def geom(M, K, N, cfg):
    Ns, Pk, Sm, kb, nsb = cfg
    Mt, Kt, Nt = cdiv(M, 32), cdiv(K, 32), cdiv(N, 32)
    Mblk = cdiv(Mt, Sm)
    Nband = cdiv(Nt, 8)
    Nown = cdiv(Nband, Ns)
    Nsub = nsb if nsb else Nown
    Nbpc = cdiv(Nown, Nsub)
    T = Mblk * Nsub
    return Mblk, Nsub, Nbpc, T


def feasible(cfg, geo):
    # Matches the kernel guard: Pk>1 and each output SUB-block (T=M_block*N_sub) tile-partitions into Pk
    # chunks. N_bpc>1 now supported (one reduce-scatter per sub-block).
    Ns, Pk, Sm, kb, nsb = cfg
    Mblk, Nsub, Nbpc, T = geo
    if Pk <= 1:
        return False, "Pk==1 (no split-K reduction)"
    if T < Pk or T % Pk != 0:
        return False, f"sub-block T={T} not partitionable into Pk={Pk} chunks"
    return True, "ok"


def med_ab(M, K, N, cfg):
    walls = {CHAIN: [], RS: []}
    for batch, order in enumerate([[CHAIN, RS], [RS, CHAIN]]):
        for i in range(N_RELAUNCH):
            for mask in order:
                r = ds.run_one(M, K, N, cfg, mask, iters=ITERS, timeout=200)
                if r.get("ok") and r.get("wall_us") is not None:
                    walls[mask].append(r["wall_us"])
                else:
                    return None, None, None, f"run fail mask={mask} cls={r.get('cls')}"
    return statistics.median(walls[CHAIN]), statistics.median(walls[RS]), walls, None


def main():
    corpus = json.load(open(f"{ds.HERE}/regime_a_current_perf.json"))["mt8"]
    # Prioritize the shallow-K (K<=2048, exposed-reduction) shapes first, then the rest (deep-K).
    corpus = sorted(corpus, key=lambda r: (r["K"] > 2048, r["K"], r["M"], r["N"]))
    out = {"n_relaunch": N_RELAUNCH, "iters": ITERS, "shapes": []}
    for r in corpus:
        M, K, N, cfg = r["M"], r["K"], r["N"], tuple(r["cfg"])
        geo = geom(M, K, N, cfg)
        feas, reason = feasible(cfg, geo)
        rec = {
            "M": M,
            "K": K,
            "N": N,
            "cfg": list(cfg),
            "feasible": feas,
            "reason": reason,
            "Mblk": geo[0],
            "Nsub": geo[1],
            "Nbpc": geo[2],
            "T": geo[3],
        }
        if feas:
            cw, rw, walls, err = med_ab(M, K, N, cfg)
            if err:
                rec["error"] = err
            else:
                rec["chain_us"] = cw
                rec["rscatter_us"] = rw
                rec["delta_pct"] = (rw - cw) / cw * 100.0
                rec["chain_samples"] = walls[CHAIN]  # raw per-relaunch wall samples (both batches)
                rec["rscatter_samples"] = walls[RS]
            tag = f"chain={cw:.2f} rs={rw:.2f} delta={rec.get('delta_pct',0):+.2f}%" if not err else err
            print(f"[FEAS] {M}x{K}x{N} cfg={cfg} Pk={cfg[1]} Nbpc={geo[2]} T={geo[3]} -> {tag}", flush=True)
        else:
            print(f"[skip] {M}x{K}x{N} cfg={cfg} -> {reason}", flush=True)
        out["shapes"].append(rec)
        json.dump(out, open(f"{ds.HERE}/ab_rscatter_corpus_results.json", "w"), indent=2)
    # summary
    feas = [s for s in out["shapes"] if s["feasible"] and "delta_pct" in s]
    print("\n===== REDUCE-SCATTER @ PICKER CONFIG, 60-shape corpus =====", flush=True)
    print(f"feasible+measured: {len(feas)} / {len(out['shapes'])}", flush=True)
    for s in sorted(feas, key=lambda z: z["delta_pct"]):
        print(
            f"  {s['M']}x{s['K']}x{s['N']:<5} Pk={s['cfg'][1]:<2} chain={s['chain_us']:.2f} rs={s['rscatter_us']:.2f} delta={s['delta_pct']:+.2f}%",
            flush=True,
        )
    if feas:
        best = min(s["delta_pct"] for s in feas)
        wins = [s for s in feas if s["delta_pct"] <= -2.0]
        print(f"  best {best:+.2f}%  ; wins(<=-2%): {len(wins)}", flush=True)
    print(f"wrote {ds.HERE}/ab_rscatter_corpus_results.json", flush=True)


if __name__ == "__main__":
    main()
