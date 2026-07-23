#!/usr/bin/env python3
"""Boundary-aware corpus for the picker-generalization campaign.

Builds a curated set of (M,K,N) element shapes (NOT a dense Cartesian K x N grid) covering:
  - all real Mt<=8 FLUX/LTX + M-scaling production shapes,
  - dense K/N anchors 1024..8192,
  - edge/extrapolation dims 256/512/768/15360,
  - narrow-/balanced-/wide-N aspect ratios,
  - Nt%8, Pk/kb divisibility, Sm divisibility, W-transition, L1-boundary cases,
  - selected non-divisible / tail dims,
  - Mt=1..8 on a small anchor set, Mt={1,2,4,8} broadly, some Mt={3,5,6,7}.

Each shape is tagged with the category(ies) it represents. Emits corpus_manifest.json and prints the
feasible-config total + a runtime estimate at the pilot's measured ~9 s/candidate.

Regime-A is low arithmetic intensity: M is small (Mt=1..8 => M=32..256); K,N are the large dims.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import regime_a_model as model  # noqa: E402

TILE = 32
SEC_PER_CAND = 9.0  # pilot-measured per-candidate subprocess wall (mean ~8-10 s)


def mkn(M, K, N):
    return (M, K, N)


# ------------------------------------------------------------------------------------------------
# Group builders. Everything in ELEMENTS; Mt=M/32, Kt=K/32, Nt=N/32.
# ------------------------------------------------------------------------------------------------
shapes = {}  # (M,K,N) -> set(tags)


def add(M, K, N, tag):
    shapes.setdefault((M, K, N), set()).add(tag)


# --- (1) Real production shapes: the 60-shape M-scaling corpus + 20 FLUX/LTX table shapes ---
try:
    base = json.load(open(f"{HERE}/../regime_a_current_perf.json"))["mt8"]
    for r in base:
        add(r["M"], r["K"], r["N"], "real-60corpus")
except Exception:
    pass
# FLUX/LTX table shapes (tile dims -> elements).
for Mt, Kt, Nt in model.KTABLE:
    add(Mt * TILE, Kt * TILE, Nt * TILE, "real-fluxltx-table")

# --- (2) Dense K/N anchors 1024..8192 (elements) ---
ANCHORS = [1024, 1280, 1536, 1792, 2048, 2304, 2560, 3072, 3584, 4096, 4608, 5120, 6144, 7168, 7680, 8192]
EDGE = [256, 512, 768, 15360]
# narrow-N (N<<K): big K, small N ; balanced (K~N) ; wide-N (N>>K): small K, big N.
# Use Mt in {1,2,4,8} broadly across a representative anchor cross-section (NOT full Cartesian).
for M in (32, 64, 128, 256):
    # narrow-N: pair a large K anchor with a small N
    for K in (4096, 6144, 8192, 15360):
        for N in (512, 768, 1024):
            add(M, K, N, "anchor-narrowN")
    # wide-N: small K, large N
    for K in (512, 768, 1024):
        for N in (4096, 6144, 8192):
            add(M, K, N, "anchor-wideN")
    # balanced: K ~ N along the anchor diagonal (subset)
    for KN in (1536, 2048, 3072, 4608, 6144):
        add(M, KN, KN, "anchor-balanced")

# --- (3) Mt=1..8 (incl. 3,5,6,7) on a small anchor set (interpolation test) ---
for M in (32, 64, 96, 128, 160, 192, 224, 256):
    add(M, 6144, 1536, "mt-sweep-narrowN")
    add(M, 2048, 2048, "mt-sweep-balanced")
    add(M, 1024, 6144, "mt-sweep-wideN")

# --- (4) Boundary cases ---
# Nt%8 transitions: Nt just below / at / above multiples of 8 tiles. Nt=N/32.
# feasibility needs 7*ceil(Nt/8) < Nt. Smallest feasible Nt is 8 (N=256). Probe Nt in {8,9,15,16,17,24,25}.
for Nt in (8, 9, 15, 16, 17, 24, 25, 31, 32, 33):
    add(64, 6144, Nt * TILE, "boundary-Nt%8")
# W-transition: W = K_slice_capacity/(kb*8) changes at Kt crossing kb*8*Pk multiples. Probe Kt near 8/16/24.
for Kt in (8, 16, 24, 32, 40, 48, 64, 96, 128, 192, 256):
    add(64, Kt * TILE, 1536, "boundary-W")
# Sm divisibility: Mt that divide / don't divide by candidate Sm (2,3,4).
for Mt in (2, 3, 4, 5, 6, 7, 8):
    add(Mt * TILE, 2048, 768, "boundary-Sm-div")
# L1 boundary: deep-K + wide subblocks push cb0/cb1 to the L1 edge.
add(256, 15360, 1536, "boundary-L1")
add(256, 15360, 6144, "boundary-L1")
add(128, 15360, 768, "boundary-L1")
add(32, 15360, 6144, "boundary-L1")

# --- (5) Non-divisible / tail dims (K,N not multiples of 8*32; M not multiple of 32 handled by Mt only) ---
for K, N in ((6100, 1568), (6080, 4640), (5152, 2080), (3104, 1312), (7712, 800)):
    add(64, K, N, "tail-nondiv")
    add(128, K, N, "tail-nondiv")

# ------------------------------------------------------------------------------------------------
# Feasibility counts + runtime estimate
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
# SWEPT selection (device-measured this campaign). The full corpus above (~180 shapes) is documented as
# the ideal target; measuring every feasible config for all of it is ~565 h (see estimate) — infeasible.
# Per the directive ("prefer fewer shapes with EXHAUSTIVE config coverage over many incompletely swept
# shapes"), we exhaustively sweep this 16-shape subset (every feasible (Pk,Ns,Sm,kb) x nsb-lattice),
# ~31 h initial. It spans Mt=1..8, narrow/balanced/wide-N, real/interp/boundary/tail, and both endpoints
# of the aspect + Mt axes. nsb is sampled on the geometric+boundary lattice (regime_a_model.nsb_lattice),
# NOT the picker's set — one shape carries a full-nsb control in the sweep to validate the lattice.
SWEPT = {
    "train": [
        (256, 2048, 1024),
        (256, 2048, 512),
        (256, 15360, 1536),
        (192, 6144, 1536),
        (32, 6144, 1536),
        (64, 6144, 1536),
        (64, 2048, 2048),
        (224, 6144, 512),
    ],
    "val": [
        (128, 2048, 2048),
        (128, 15360, 768),
        (96, 6144, 1536),
        (32, 1024, 6144),
    ],
    "holdout": [
        (256, 2304, 6144),
        (64, 6080, 4640),
        (256, 15360, 6144),
        (160, 6144, 1536),
    ],
}


def main():
    rows = []
    total_cfg = 0
    infeasible_shapes = []
    for (M, K, N), tags in sorted(shapes.items()):
        Mt, Kt, Nt = M // TILE, K // TILE, N // TILE
        if not model.nt_width_shard_feasible(Nt):
            infeasible_shapes.append((M, K, N))
            continue
        feas = model.enumerate_feasible(Mt, Kt, Nt)
        if not feas:
            infeasible_shapes.append((M, K, N))
            continue
        try:
            pick = model.production_pick(Mt, Kt, Nt)
        except RuntimeError:
            pick = None
        total_cfg += len(feas)
        rows.append(
            {
                "M": M,
                "K": K,
                "N": N,
                "Mt": Mt,
                "Kt": Kt,
                "Nt": Nt,
                "tags": sorted(tags),
                "n_feasible": len(feas),
                "prod_pick": list(pick) if pick else None,
            }
        )
    # --- swept selection with train/val/holdout split + per-shape full & lattice config counts ---
    split_of = {}
    for sp, lst in SWEPT.items():
        for s in lst:
            split_of[s] = sp
    swept_rows = []
    swept_total_lat = 0
    for (M, K, N), sp in sorted(split_of.items()):
        Mt, Kt, Nt = M // TILE, K // TILE, N // TILE
        nfull = len(model.enumerate_feasible(Mt, Kt, Nt))
        nlat = len(model.enumerate_feasible(Mt, Kt, Nt, nsb_mode="lattice"))
        swept_total_lat += nlat
        pick = model.production_pick(Mt, Kt, Nt)
        swept_rows.append(
            {
                "M": M,
                "K": K,
                "N": N,
                "Mt": Mt,
                "Kt": Kt,
                "Nt": Nt,
                "split": sp,
                "n_feasible_full": nfull,
                "n_feasible_lattice": nlat,
                "prod_pick": list(pick),
            }
        )

    manifest = {
        "sec_per_candidate": SEC_PER_CAND,
        "full_target_corpus": {
            "n_shapes": len(rows),
            "n_infeasible_dropped": len(infeasible_shapes),
            "total_feasible_configs": total_cfg,
            "est_full_pass_hours": round(total_cfg * SEC_PER_CAND / 3600, 1),
            "shapes": rows,
        },
        "swept": {
            "nsb_mode": "lattice",
            "n_shapes": len(swept_rows),
            "total_lattice_configs": swept_total_lat,
            "est_initial_pass_hours": round(swept_total_lat * SEC_PER_CAND / 3600, 1),
            "est_with_reruns_hours": round(swept_total_lat * SEC_PER_CAND / 3600 * 1.2, 1),
            "split_counts": {sp: len(lst) for sp, lst in SWEPT.items()},
            "shapes": swept_rows,
        },
    }
    json.dump(manifest, open(f"{HERE}/corpus_manifest.json", "w"), indent=2)

    import statistics

    counts = sorted(r["n_feasible"] for r in rows)
    print("=== FULL TARGET CORPUS (documented ideal) ===")
    print(f"shapes (feasible):        {len(rows)}")
    print(f"shapes dropped (infeas):  {len(infeasible_shapes)}")
    print(f"total feasible configs:   {total_cfg}")
    print(f"per-shape configs:        min {counts[0]}  median {int(statistics.median(counts))}  max {counts[-1]}")
    print(f"est FULL-pass runtime:    {manifest['full_target_corpus']['est_full_pass_hours']} h (infeasible)")
    print("=== SWEPT SELECTION (device-measured, nsb-lattice, exhaustive over (Pk,Ns,Sm,kb)) ===")
    sw = manifest["swept"]
    print(f"shapes:                   {sw['n_shapes']}  split {sw['split_counts']}")
    print(f"total lattice configs:    {sw['total_lattice_configs']}")
    print(f"est initial-pass runtime: {sw['est_initial_pass_hours']} h  (+reruns ~{sw['est_with_reruns_hours']} h)")
    print(f"Mt coverage (swept):      {sorted(set(r['Mt'] for r in swept_rows))}")
    print(f"wrote {HERE}/corpus_manifest.json")


if __name__ == "__main__":
    main()
