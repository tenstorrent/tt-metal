#!/usr/bin/env python3
# 3-way A/B on the PRIMARY 256x2048x1024 at the fixed Pk=4 config (1,4,2,2,4): reduction CHAIN (mask 0) vs
# fan-in-2 TREE (mask 64) vs ring REDUCE-SCATTER (mask 128). Same tensors/ring/placement/compute; only the
# split-K reduction structure differs. Reduce-scatter distributes the reduction adds + output writes across
# all Pk cores and eliminates the single-root receive-wait tail — the orchestrator's strongest remaining
# macro-level reduction lever. Protocol: 2 reversed-order batches x 10 relaunches, median-of-medians,
# hang-safe run_one. All three are PCC-preserving (verified by the gtest); run_one asserts PASS for each.
import json, os, statistics, sys

sys.path.insert(0, os.path.dirname(__file__))
import regime_a_diag_suite as ds

M, K, N = 256, 2048, 1024
CFG = (1, 4, 2, 2, 4)  # Ns,Pk,Sm,kb,nsb -> M_block=4=Pk, N_bpc=1 (reduce-scatter feasible)
VARIANTS = [("chain", 0), ("tree", 64), ("rscatter", 128)]
N_RELAUNCH = 10
ITERS = 8


def main():
    walls = {name: [] for name, _ in VARIANTS}
    fails = []
    orders = [VARIANTS, list(reversed(VARIANTS))]  # 2 reversed batches
    for batch, order in enumerate(orders):
        for i in range(N_RELAUNCH):
            for name, mask in order:
                r = ds.run_one(M, K, N, CFG, mask, iters=ITERS, timeout=180)
                if not r.get("ok") or r.get("wall_us") is None:
                    fails.append({"batch": batch, "i": i, "variant": name, "cls": r.get("cls")})
                    print(f"  [b{batch} i{i}] {name:9s} FAIL cls={r.get('cls')}", flush=True)
                    continue
                walls[name].append(r["wall_us"])
                print(f"  [b{batch} i{i}] {name:9s} wall={r['wall_us']:.3f}us mre={r.get('max_rel_err')}", flush=True)
    med = {name: (statistics.median(w) if w else None) for name, w in walls.items()}
    base = med["chain"]
    out = {
        "shape": [M, K, N],
        "cfg": list(CFG),
        "n_relaunch": N_RELAUNCH,
        "iters": ITERS,
        "fails": fails,
        "variants": {},
    }
    for name, _ in VARIANTS:
        w = walls[name]
        delta = ((med[name] - base) / base * 100.0) if (base and med[name]) else None
        out["variants"][name] = {
            "med_us": med[name],
            "n": len(w),
            "delta_vs_chain_pct": delta,
            "iqr": statistics.quantiles(w, n=4) if len(w) >= 4 else None,
        }
    json.dump(out, open(f"{ds.HERE}/ab_rscatter_results.json", "w"), indent=2)
    print("\n===== 3-WAY SUMMARY (256x2048x1024, cfg (1,4,2,2,4)) =====", flush=True)
    for name, _ in VARIANTS:
        v = out["variants"][name]
        d = f"{v['delta_vs_chain_pct']:+.2f}%" if v["delta_vs_chain_pct"] is not None else "n/a"
        iqr = v["iqr"]
        iqrs = f"[{iqr[0]:.2f},{iqr[2]:.2f}]" if iqr else "n/a"
        print(f"  {name:9s} med={v['med_us']} us  IQR{iqrs}  delta_vs_chain={d}", flush=True)
    print(f"fails={len(fails)}  wrote {ds.HERE}/ab_rscatter_results.json", flush=True)


if __name__ == "__main__":
    main()
