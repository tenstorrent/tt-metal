#!/usr/bin/env python3
# Follow-up to ab_redtree.py: rerun the CHAIN vs TREE A/B on 32x2048x512 at its REAL production picker
# config (2,4,1,2,1) [Ns,Pk,Sm,kb,nsb], not the (1,4,1,2,2) used in the first sweep. The earlier -4.31% tree
# win must be re-established on the deployed config before any selective-adoption claim (orchestrator). Same
# protocol: 2 reversed batches x 10 relaunches, median-of-medians, hang-safe run_one.
import json, os, statistics, sys

sys.path.insert(0, os.path.dirname(__file__))
import regime_a_diag_suite as ds

M, K, N = 32, 2048, 512
CFG = (2, 4, 1, 2, 1)  # production picker's config for this shape (Pk=4)
CHAIN, TREE = 0, 64
N_RELAUNCH = 10
ITERS = 8


def main():
    walls = {CHAIN: [], TREE: []}
    fails = []
    for batch, order in enumerate([[CHAIN, TREE], [TREE, CHAIN]]):
        for i in range(N_RELAUNCH):
            for mask in order:
                r = ds.run_one(M, K, N, CFG, mask, iters=ITERS, timeout=180)
                tag = "chain" if mask == CHAIN else "tree "
                if not r.get("ok") or r.get("wall_us") is None:
                    fails.append({"batch": batch, "i": i, "mask": mask, "cls": r.get("cls")})
                    print(f"  [b{batch} i{i}] {tag} FAIL cls={r.get('cls')}", flush=True)
                    continue
                walls[mask].append(r["wall_us"])
                print(f"  [b{batch} i{i}] {tag} wall={r['wall_us']:.3f}us mre={r.get('max_rel_err')}", flush=True)
    cw = statistics.median(walls[CHAIN]) if walls[CHAIN] else None
    tw = statistics.median(walls[TREE]) if walls[TREE] else None
    delta = ((tw - cw) / cw * 100.0) if (cw and tw) else None
    res = {
        "shape": [M, K, N],
        "cfg": list(CFG),
        "chain_med_us": cw,
        "tree_med_us": tw,
        "delta_pct": delta,
        "chain_n": len(walls[CHAIN]),
        "tree_n": len(walls[TREE]),
        "fails": fails,
        "chain_iqr": statistics.quantiles(walls[CHAIN], n=4) if len(walls[CHAIN]) >= 4 else None,
        "tree_iqr": statistics.quantiles(walls[TREE], n=4) if len(walls[TREE]) >= 4 else None,
    }
    json.dump(res, open(f"{ds.HERE}/ab_redtree_prodcfg_results.json", "w"), indent=2)
    d = f"{delta:+.2f}%" if delta is not None else "n/a"
    print(
        f"\n32x2048x512 cfg{CFG}: chain={cw} tree={tw} delta(tree vs chain)={d} (neg=tree faster) fails={len(fails)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
