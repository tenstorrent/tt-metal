# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The fused head's planner (tt/head_plan.py), host only: the cost model against the measured heads (CAL: MOE_TRACE_DEV
head program times, uniform routing, traced; the layout each run's planner took), its fit (fit_cost - prints the
rates to paste into head_plan.Cost), and a coverage sweep over real models and synthetic dims that flags cliffs
(predicted time jumping between neighbouring shapes) and shapes with no layout at all."""

import math

import pytest
from loguru import logger

from models.demos.moe_block.tt.head_plan import COST, Cost, HeadLayout, HeadShape, cost, describe, k_steps, plan, size

# (H, I_s, E router candidates, K, latent) - I_s split over TP 4
MODELS = {
    "k3": (7168, 6144, 896, 16, 3584),
    "k27": (7168, 2048, 384, 8, None),
    "glmf": (4096, 2048, 288, 8, None),
    "glm53": (6144, 2048, 256, 8, None),
    "m3": (6144, 3072, 128, 4, None),
    "v41": (5120, 2304, 384, 6, None),
}

# (model, mesh: "2x4" line / "ring" of 4, rows a mesh row, GY, MT, NCH, router on the engine, measured head us) - the
# runs' layouts built with P-tile RS zones (before compact zones); bh-lb-09, 200G, traced replays' median
CAL = [
    ("k3", "2x4", 128, 4, 1, 1, True, 316.7),
    ("k3", "2x4", 256, 8, 1, 1, True, 522.0),
    ("k3", "2x4", 512, 8, 2, 1, True, 567.5),
    ("k3", "2x4", 640, 10, 2, 1, True, 639.3),
    ("k3", "2x4", 768, 6, 2, 2, False, 898.0),
    ("k3", "2x4", 1024, 8, 2, 2, False, 1077.8),
    ("k3", "2x4", 1280, 8, 1, 5, False, 1908.0),
    ("k3", "ring", 128, 4, 1, 1, True, 296.3),
    ("k3", "ring", 256, 8, 1, 1, True, 488.0),
    ("k3", "ring", 512, 8, 2, 1, True, 423.4),
    ("k3", "ring", 640, 10, 2, 1, True, 457.9),
    ("k3", "ring", 768, 6, 2, 2, False, 737.9),
    ("k3", "ring", 1024, 8, 2, 2, False, 852.5),
    ("k3", "ring", 1280, 8, 1, 5, False, 1660.8),
    ("glmf", "2x4", 128, 4, 1, 1, True, 130.9),
    ("glmf", "2x4", 256, 8, 1, 1, True, 148.3),
    ("glmf", "2x4", 512, 8, 2, 1, True, 246.9),
    ("glmf", "2x4", 640, 10, 2, 1, True, 281.3),
    ("glmf", "2x4", 768, 6, 2, 2, True, 325.9),
    ("glmf", "2x4", 1024, 8, 2, 2, True, 408.3),
    ("glmf", "2x4", 1280, 8, 1, 5, True, 603.6),
    ("glmf", "ring", 128, 4, 1, 1, True, 114.7),
    ("glmf", "ring", 256, 8, 1, 1, True, 127.2),
    ("glmf", "ring", 512, 8, 2, 1, True, 196.2),
    ("glmf", "ring", 640, 10, 2, 1, True, 212.9),
    ("glmf", "ring", 768, 6, 2, 2, True, 274.7),
    ("glmf", "ring", 1024, 8, 2, 2, True, 335.4),
    ("glmf", "ring", 1280, 8, 1, 5, True, 499.1),
    ("glm53", "2x4", 128, 4, 1, 1, True, 134.6),
    ("glm53", "2x4", 256, 8, 1, 1, True, 207.9),
    ("glm53", "2x4", 512, 8, 2, 1, True, 330.8),
    ("glm53", "2x4", 640, 10, 2, 1, True, 392.6),
    ("glm53", "2x4", 768, 8, 1, 3, False, 476.0),
    ("glm53", "2x4", 1024, 8, 1, 4, False, 639.5),
    ("glm53", "2x4", 1280, 8, 1, 5, False, 845.0),
    ("glm53", "ring", 128, 4, 1, 1, True, 136.2),
    ("glm53", "ring", 256, 8, 1, 1, True, 173.8),
    ("glm53", "ring", 512, 8, 2, 1, True, 275.9),
    ("glm53", "ring", 640, 10, 2, 1, True, 310.7),
    ("glm53", "ring", 768, 6, 2, 2, True, 371.4),
    ("glm53", "ring", 1024, 8, 2, 2, True, 466.2),
    ("glm53", "ring", 1280, 8, 1, 5, False, 664.5),
    ("k27", "2x4", 640, 4, 1, 5, False, 763.2),
    ("k27", "2x4", 128, 4, 1, 1, True, 152.6),
    ("k27", "2x4", 256, 8, 1, 1, True, 241.6),
    ("k27", "2x4", 512, 8, 1, 2, False, 368.2),
    ("k27", "2x4", 768, 8, 1, 3, False, 553.9),
    ("k27", "2x4", 1024, 8, 1, 4, False, 749.8),
    ("k27", "2x4", 1280, 8, 1, 5, False, 944.2),
]
# (the planner's own runs - compact RS zones, the K step it took: model, mesh, rows, GY, MT, NCH, on the engine, KB, us)
CAL_COMPACT = [
    ("k27", "2x4", 512, 8, 2, 1, True, 7, 380.1),
    ("k27", "2x4", 640, 10, 2, 1, True, 7, 444.7),
    ("k27", "2x4", 768, 4, 2, 3, False, 2, 983.1),
    ("k27", "2x4", 1024, 8, 2, 2, True, 7, 679.8),
    ("glmf", "2x4", 768, 6, 2, 2, True, 8, 326.6),
    ("glmf", "2x4", 1280, 4, 2, 5, False, 4, 673.6),
    ("glm53", "2x4", 128, 4, 1, 1, True, 12, 135.8),
    ("glm53", "2x4", 256, 8, 1, 1, True, 12, 209.0),
    ("glm53", "2x4", 512, 8, 2, 1, True, 12, 348.5),
    ("glm53", "2x4", 640, 10, 2, 1, True, 12, 401.9),
    ("glm53", "2x4", 768, 6, 2, 2, True, 8, 477.4),
    ("glm53", "2x4", 1024, 8, 2, 2, True, 8, 607.2),
    ("glm53", "2x4", 1280, 8, 1, 5, False, 12, 804.8),
]


def fold_of(E, N):
    """The router's fold a LoudBox run takes: experts a chip <= 64 for the routing planner (K3 on 8 chips: 8)."""
    f = 1
    while E // f // N > 64:
        f *= 2
    return f


def shape_of(model, rows, mesh, G=4):
    H, Is, E, K, lat = MODELS[model]
    R = 2 if mesh == "2x4" else 1
    return HeadShape(
        rows=rows, H=H, I=Is // G, G=G, E=E, K=K, latent=lat, ring=mesh == "ring", R=R, fold=fold_of(E, R * G)
    )


def cal_layouts():
    """CAL's points with their engines sized as the runs built them: P-tile RS zones and the K step the head took (the
    largest that fit); CAL_COMPACT's with compact zones at their K step."""
    out = []
    pts = [p + (None, False) for p in CAL] + [p[:7] + (p[8], p[7], True) for p in CAL_COMPACT]
    for model, mesh, rows, GY, MT, NCH, on, t, KB, compact in pts:
        sh = shape_of(model, rows, mesh)
        for kb in [KB] if KB else k_steps(sh):
            lay = HeadLayout(GY=GY, MT=MT, NCH=NCH, KB=kb, on_engine=on)
            eng, *_ = size(sh, lay, compact=compact)
            if eng is not None:
                out.append((sh, lay, eng, t, f"{model} {mesh} {rows}{' c' if compact else ''} KB{kb}"))
                break
        else:
            raise AssertionError(f"{model} {mesh} {rows}: the measured layout does not size")
    return out


FIT = (
    "tile",
    "round_us",
    "link_gbs",
    "dram_gbs",
    "rtr_tile",
    "chain_us",
    "chunk_us",
    "base_us",
    "overlap",
    "noc_gbs",
    "chain_ek",
    "xmc_gbs",
)
BOUNDS = dict(
    tile=(20, 120),
    round_us=(0, 6),
    link_gbs=(8, 30),
    dram_gbs=(150, 500),
    rtr_tile=(20, 400),
    chain_us=(30, 400),
    chunk_us=(0, 200),
    base_us=(0, 150),
    overlap=(0, 1),
    noc_gbs=(4, 80),
    chain_ek=(0, 2),
    xmc_gbs=(2, 80),
)


def errors(c: Cost, pts):
    return [(cost(sh, lay, eng, c)[0] / t - 1, tag, t, cost(sh, lay, eng, c)) for sh, lay, eng, t, tag in pts]


def fit_cost(pts):
    """Least squares on log(predicted / measured) over FIT's rates, within BOUNDS."""
    import numpy as np
    from scipy.optimize import least_squares

    lo = [BOUNDS[k][0] for k in FIT]
    hi = [BOUNDS[k][1] for k in FIT]
    x0 = [min(max(getattr(COST, k), a), b) for k, a, b in zip(FIT, lo, hi)]

    def res(x):
        c = Cost(**dict(zip(FIT, x)))
        return [math.log(cost(sh, lay, eng, c)[0] / t) for sh, lay, eng, t, _ in pts]

    best = None
    for seed in range(6):  # (a few starts: the max() terms make it piecewise)
        rng = np.random.default_rng(seed)
        xs = x0 if seed == 0 else [a + (b - a) * rng.random() for a, b in zip(lo, hi)]
        r = least_squares(res, xs, bounds=(lo, hi))
        if best is None or r.cost < best.cost:
            best = r
    return Cost(**dict(zip(FIT, best.x)))


def test_cost_model_calibration():
    """The cost model's error on every measured head (COST as committed); prints a refit for comparison."""
    pts = cal_layouts()
    errs = errors(COST, pts)
    for e, tag, t, (p, parts) in errs:
        logger.info(f"CAL {tag:<16} measured {t:7.1f} predicted {p:7.1f} ({e:+.1%}) {parts}")
    worst = max(abs(e) for e, *_ in errs)
    med = sorted(abs(e) for e, *_ in errs)[len(errs) // 2]
    logger.info(f"CAL error: median {med:.1%}, worst {worst:.1%} over {len(errs)} heads")
    refit = fit_cost(pts)
    e2 = errors(refit, pts)
    logger.info(
        f"CAL refit: median {sorted(abs(e) for e, *_ in e2)[len(e2) // 2]:.1%}, worst {max(abs(e) for e, *_ in e2):.1%}: "
        + ", ".join(f"{k}={getattr(refit, k):.3g}" for k in FIT)
    )
    assert med < 0.12 and worst < 0.35, f"the cost model is off the measurements: median {med:.1%}, worst {worst:.1%}"


SWEEP_H = (2048, 3072, 4096, 5120, 6144, 7168, 8192)
SWEEP_I = (1024, 2048, 3072, 4096, 6144, 8192)
SWEEP_ROWS = (128, 256, 384, 512, 640, 768, 1024, 1280, 1536, 2048)


@pytest.mark.timeout(1800)  # (every shape's whole plan - the routing exchange's paths and buffers too: minutes)
@pytest.mark.parametrize("G", [4, 2])
@pytest.mark.parametrize("mesh", ["2x4", "ring"])
def test_head_plan_sweep(G, mesh):
    """Every (H, I_s, E, K, rows) of the synthetic grid + the real models: the chosen layout and its prediction;
    cliffs - a shape predicted > 1.6x its neighbour with a third more rows (or the same rows and 1 step less H / I) -
    and shapes with no layout, logged (the report: MOE_BLOCK_WORKLOG.md)."""
    if mesh == "ring" and G == 2:
        pytest.skip("a ring of 2 is a line")
    none, cliffs, n = [], [], 0
    for H in SWEEP_H:
        for Is in SWEEP_I:
            for E, K in ((256, 8), (384, 8), (896, 16)):
                prev = None
                for rows in SWEEP_ROWS:
                    if rows % (32 * G) or (H // 32 // G) % 2:
                        continue
                    R = 2 if mesh == "2x4" else 1
                    sh = HeadShape(
                        rows=rows, H=H, I=Is // G, G=G, E=E, K=K, ring=mesh == "ring", R=R, fold=fold_of(E, R * G)
                    )
                    ps = plan(sh)
                    n += 1
                    if not ps:
                        none.append((H, Is, E, rows))
                        prev = None
                        continue
                    t = ps[0].t
                    if prev and t > 1.6 * prev[1] * rows / prev[0]:
                        cliffs.append((H, Is, E, prev[0], rows, round(prev[1]), round(t), describe(ps[0])))
                    prev = (rows, t)
    logger.info(f"SWEEP G {G} {mesh}: {n} shapes, {len(none)} without a layout, {len(cliffs)} cliffs")
    for c in cliffs[:40]:
        logger.info(f"SWEEP cliff H {c[0]} I_s {c[1]} E {c[2]}: rows {c[3]} -> {c[4]}: {c[5]} -> {c[6]} us ({c[7]})")
    for c in none[:40]:
        logger.info(f"SWEEP no layout: H {c[0]} I_s {c[1]} E {c[2]} rows {c[3]}")


@pytest.mark.parametrize("mesh", ["2x4", "ring"])
def test_head_plan_models(mesh):
    """The real models at 128 .. 2048 rows: the chosen layout, the prediction (logged)."""
    for m in MODELS:
        for rows in SWEEP_ROWS:
            ps = plan(shape_of(m, rows, mesh))
            logger.info(f"PLAN {m:<6} {mesh} {rows:5}: " + (describe(ps[0]) if ps else "no layout"))
