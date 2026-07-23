#!/usr/bin/env python3
"""Generator-driven corpus (v2) for the picker-generalization campaign.

Broad boundary-aware corpus (~110-130 shapes) sized for the theory-guided generator (budget=96, audit=8
per shape => ~11-13k initial measurements spread over >100 shapes). Shapes are assigned to
train/val/holdout BEFORE measuring; real-model (FLUX/LTX) and tail shapes are represented in
validation/holdout as well as training.

Composition (instruction section 2):
  - Matched anchors: all Mt=1..8 (M=32*Mt) x 12 (K,N) pairs = 96 shapes (Mt-dependent boundary learning).
  - Every real Mt<=8 FLUX/LTX shape (KTABLE keys, tile->element).
  - For Mt={1,4,8}: 8 boundary/interpolation (K,N) pairs.
  - Representative non-tile / Nt%8 tail shapes.

Emits corpus_v2_manifest.json (with per-shape production pick + generator candidate count) and prints
the total candidate count + Mt/split coverage.
"""
import json, os, sys, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import regime_a_model as model  # noqa: E402
import regime_a_candidate_generator as cg  # noqa: E402

TILE = 32

MATCHED_KN = [
    (1024, 1024), (2048, 512), (2048, 1024), (2048, 2048), (2048, 6144), (4096, 1024),
    (4096, 4096), (6144, 768), (6144, 2304), (6144, 6144), (8192, 1536), (15360, 768),
]
BOUNDARY_KN = [
    (1280, 1792), (2304, 6144), (3072, 6144), (4608, 3072),
    (7168, 1280), (7680, 4608), (6080, 4640), (6100, 4608),
]
# Representative non-tile (K or N not a multiple of 32) + Nt%8 tail shapes.
TAIL_SHAPES = [
    (64, 6100, 1568), (128, 6080, 4640), (64, 5152, 2080), (128, 3104, 1312), (64, 7712, 800),
    (64, 6144, 288 + 32), (64, 6144, 512), (64, 6144, 544), (256, 15360, 6144), (32, 15360, 6144),
]


def prod_pick_gen_order(Mt, Kt, Nt):
    """production_pick -> (Ns,Pk,Sm,kb,nsb) generator tuple order (or None if infeasible)."""
    try:
        pk = model.production_pick(Mt, Kt, Nt)  # (Pk,Ns,Sm,kb,nsb,source)
    except RuntimeError:
        return None
    return (pk[1], pk[0], pk[2], pk[3], pk[4])


def assign_split(M, K, N):
    """Deterministic 60/20/20 split by hash so it is stable across reruns. Real/tail shapes are tagged
    separately and force-distributed in build() so they appear in val/holdout too."""
    h = int.from_bytes(hashlib.sha256(f"{M},{K},{N}".encode()).digest()[:4], "little") % 100
    if h < 60:
        return "train"
    if h < 80:
        return "val"
    return "holdout"


def build():
    shapes = {}  # (M,K,N) -> set(tags)

    def add(M, K, N, tag):
        shapes.setdefault((M, K, N), set()).add(tag)

    for Mt in range(1, 9):
        M = 32 * Mt
        for (K, N) in MATCHED_KN:
            add(M, K, N, "matched")
    for (Mt, Kt, Nt) in model.KTABLE:
        if Mt > 8:  # campaign is the Mt<=8 low-AI regime
            continue
        add(Mt * TILE, Kt * TILE, Nt * TILE, "fluxltx")
    for Mt in (1, 4, 8):
        M = 32 * Mt
        for (K, N) in BOUNDARY_KN:
            add(M, K, N, "boundary")
    for (M, K, N) in TAIL_SHAPES:
        add(M, K, N, "tail")

    rows = []
    for (M, K, N), tags in sorted(shapes.items()):
        Mt, Kt, Nt = M // TILE, K // TILE, N // TILE
        if not model.nt_width_shard_feasible(Nt):
            continue
        inc = prod_pick_gen_order(Mt, Kt, Nt)
        sel, _reasons, stats = cg.select_candidates(M, K, N, budget=96, audit=8,
                                                    include=[inc] if inc else [])
        if not sel:
            continue
        split = assign_split(M, K, N)
        rows.append({"M": M, "K": K, "N": N, "Mt": Mt, "Kt": Kt, "Nt": Nt,
                     "tags": sorted(tags), "split": split, "prod_pick": list(inc) if inc else None,
                     "n_candidates": len(sel), "n_structured": stats["structured"],
                     "n_audit": stats["audit"]})
    return rows


def main():
    rows = build()
    total = sum(r["n_candidates"] for r in rows)
    # Ensure real+tail shapes appear in val/holdout: report their split distribution.
    from collections import Counter
    manifest = {
        "n_shapes": len(rows),
        "total_candidates": total,
        "budget": 96, "audit": 8,
        "split_counts": dict(Counter(r["split"] for r in rows)),
        "shapes": rows,
    }
    json.dump(manifest, open(f"{HERE}/corpus_v2_manifest.json", "w"), indent=2)

    print(f"shapes:            {len(rows)}")
    print(f"total candidates:  {total}  (target 11k-13k)")
    print(f"Mt coverage:       {sorted(set(r['Mt'] for r in rows))}")
    print(f"split counts:      {manifest['split_counts']}")
    for tag in ("matched", "fluxltx", "boundary", "tail"):
        tagged = [r for r in rows if tag in r["tags"]]
        sp = Counter(r["split"] for r in tagged)
        print(f"  {tag:9s} n={len(tagged):3d}  splits={dict(sp)}")
    cand = sorted(r["n_candidates"] for r in rows)
    import statistics
    print(f"candidates/shape:  min {cand[0]} median {int(statistics.median(cand))} max {cand[-1]}")
    print(f"wrote {HERE}/corpus_v2_manifest.json")


if __name__ == "__main__":
    main()
