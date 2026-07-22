#!/usr/bin/env python3
# Fixed-config A/B: linear reduction CHAIN (mask 0) vs fan-in-2 reduction TREE (DIAG_REDTREE, mask 64).
# Both are the SAME (Ns,Pk=4,Sm,kb,nsb) config, same tensors/ring/placement/compute — only the split-K
# reduction TOPOLOGY differs (chain depth 3 -> tree depth 2). Output is bit-exact for both (verified by the
# gtest constant-input + random-PCC checks; run_one asserts the PASS for both masks). This measures whether
# depth-2 beats depth-3 on the wall.
#
# Protocol (isolate the topology variable; guard against drift/ordering bias):
#   - 3 shapes, each at a Pk=4 config: the reduction-exposed primary, a small shallow shape, and a deep-K
#     read-bound NEUTRAL control (tree expected ~0 there).
#   - RELAUNCH each (shape,mask) N times (fresh process each -> program-cache miss+hit inside; median wall).
#   - 2 BATCHES with REVERSED mask order per relaunch: batch A = [chain, tree], batch B = [tree, chain].
#     Pool both batches -> median-of-relaunch-medians per (shape,mask). Reversal cancels any warm/cool or
#     thermal drift that would otherwise favour whichever mask ran first.
# Adoption gate (orchestrator): adopt the tree as production default ONLY on a stable >=2% win on the primary
# OR >=1% across multiple shapes with NO regression on any; otherwise chain stays default (tree remains the
# compile-gated diagnostic) and we record a recovery hash.
import json, os, statistics, sys, time

sys.path.insert(0, os.path.dirname(__file__))
import regime_a_diag_suite as ds

# (label, M, K, N, cfg=(Ns,Pk,Sm,kb,nsb)). All Pk=4.
SHAPES = [
    ("primary_256x2048x1024", 256, 2048, 1024, (1, 4, 2, 2, 4)),  # in0-ring-exposed; reduction tail visible
    ("small_32x2048x512", 32, 2048, 512, (1, 4, 1, 2, 2)),  # shallow/tiny; Pk=4 forced
    ("neutral_32x6144x1536", 32, 6144, 1536, (1, 4, 1, 2, 3)),  # deep-K read-bound control (expect ~0)
]
CHAIN, TREE = 256, 64  # chain=FORCE_CHAIN (mask 0 may select reduce-scatter for gated shapes)
N_RELAUNCH = 10
ITERS = 8


def one(M, K, N, cfg, mask):
    r = ds.run_one(M, K, N, cfg, mask, iters=ITERS, timeout=180)
    return r


def main():
    out = {"shapes": [], "meta": {"n_relaunch": N_RELAUNCH, "iters": ITERS, "batches": 2}}
    for label, M, K, N, cfg in SHAPES:
        print(f"\n===== {label}  {M}x{K}x{N}  cfg(Ns,Pk,Sm,kb,nsb)={cfg} =====", flush=True)
        walls = {CHAIN: [], TREE: []}
        fails = []
        for batch, order in enumerate([[CHAIN, TREE], [TREE, CHAIN]]):
            for i in range(N_RELAUNCH):
                for mask in order:
                    r = one(M, K, N, cfg, mask)
                    tag = "chain" if mask == CHAIN else "tree "
                    if not r.get("ok") or r.get("wall_us") is None:
                        fails.append({"batch": batch, "i": i, "mask": mask, "cls": r.get("cls"), "rc": r.get("rc")})
                        print(f"  [b{batch} i{i}] {tag} FAIL cls={r.get('cls')} rc={r.get('rc')}", flush=True)
                        continue
                    walls[mask].append(r["wall_us"])
                    mre = r.get("max_rel_err")
                    print(f"  [b{batch} i{i}] {tag} wall={r['wall_us']:.2f}us mre={mre}", flush=True)
        cw = statistics.median(walls[CHAIN]) if walls[CHAIN] else None
        tw = statistics.median(walls[TREE]) if walls[TREE] else None
        delta = ((tw - cw) / cw * 100.0) if (cw and tw) else None  # negative = tree FASTER
        rec = {
            "label": label,
            "shape": [M, K, N],
            "cfg": list(cfg),
            "chain_med_us": cw,
            "tree_med_us": tw,
            "delta_pct": delta,
            "chain_n": len(walls[CHAIN]),
            "tree_n": len(walls[TREE]),
            "chain_iqr": (statistics.quantiles(walls[CHAIN], n=4) if len(walls[CHAIN]) >= 4 else None),
            "tree_iqr": (statistics.quantiles(walls[TREE], n=4) if len(walls[TREE]) >= 4 else None),
            "fails": fails,
        }
        out["shapes"].append(rec)
        d = f"{delta:+.2f}%" if delta is not None else "n/a"
        print(f"  --> chain={cw} tree={tw}  delta(tree vs chain)={d}  (neg=tree faster)", flush=True)
        json.dump(out, open(f"{ds.HERE}/ab_redtree_results.json", "w"), indent=2)
    print("\n===== SUMMARY =====", flush=True)
    for rec in out["shapes"]:
        d = f"{rec['delta_pct']:+.2f}%" if rec["delta_pct"] is not None else "n/a"
        print(f"  {rec['label']:28s} chain={rec['chain_med_us']}  tree={rec['tree_med_us']}  delta={d}", flush=True)
    print(f"\nwrote {ds.HERE}/ab_redtree_results.json", flush=True)


if __name__ == "__main__":
    main()
