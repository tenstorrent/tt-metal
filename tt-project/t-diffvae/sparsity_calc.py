"""CPU tile-sparsity calculator for the DiffVAE neighborhood-attention stages.

Counts, per stage and per boundary regime, the key tiles a query brick multiplies against:
  gathered -- every slot of the chunk's gather (what the op does today),
  live     -- slots holding at least one key in the window of one of the brick's queries
              (what narrowing would iterate),
  exact    -- real queries x window keys / 1024 (the dense-equivalent floor).
Units are tile pairs (one 32x32 QK^T tile), per head and per NA layer.

Windows are per-axis products, so every count factors per axis; the 27 regimes are products of
per-axis Low / Interior / High classes. A brick is Interior on an axis when none of its queries'
windows moved because of the volume edge.

Run: python tt-project/t-diffvae/sparsity_calc.py [--check-planner] [--markdown]
"""

import argparse
import itertools
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from models.tt_dit.layers.neighborhood_reference import context_window_origin, snap_extent  # noqa: E402

STAGES = {
    1: dict(volume=(21, 34, 60), window=(3, 7, 7), depth=4),
    2: dict(volume=(21, 68, 120), window=(3, 7, 7), depth=6),
    3: dict(volume=(41, 68, 120), window=(3, 5, 5), depth=4),
    4: dict(volume=(81, 136, 240), window=(3, 5, 5), depth=2),
    5: dict(volume=(145, 272, 480), window=(11, 11, 11), depth=8),
}
# Production brick at stride 1, chunk (1,1,1): _choose_sharded_brick, pinned in
# test_choose_sharded_brick_regression. Stage 1 runs the linear-order executor instead.
TODAY_BRICK = {2: (8, 4, 1), 3: (16, 2, 1), 4: (8, 2, 2), 5: (8, 2, 2)}

# (label, brick, query_chunk_bricks)
SHAPES = [
    ("(2,4,4)", (2, 4, 4), (1, 1, 1)),
    ("(4,4,4)", (2, 4, 4), (2, 1, 1)),
    ("(1,4,8)", (1, 4, 8), (1, 1, 1)),
    ("(2,2,8)", (2, 2, 8), (1, 1, 1)),
]
STRIDES = [(1, 1, 1), (2, 4, 4)]
REGIMES = "LIH"


def axis_stats(volume, window, stride, brick, chunk_bricks):
    """Per-axis counts, keyed by regime: bricks, live slots, real queries; plus the gather."""
    window = min(window, volume)
    snap = snap_extent(stride, brick)
    origins = [context_window_origin(q // stride, stride, window, volume, snap) for q in range(volume)]
    # Where each window would sit with no volume edge: shift far from 0 by a multiple of both the
    # stride and the brick so grouping and snapping are unchanged.
    shift = math.lcm(stride, brick) * (volume + window)
    free = [context_window_origin((q + shift) // stride, stride, window, 1 << 30, snap) - shift for q in range(volume)]

    bricks = math.ceil(volume / brick)
    stats = {r: [0, 0, 0] for r in REGIMES}
    for b in range(bricks):
        sites = range(b * brick, min(b * brick + brick, volume))
        low = min(origins[q] for q in sites)
        high = max(origins[q] for q in sites) + window - 1
        live = high // brick - low // brick + 1
        moved = [origins[q] - free[q] for q in sites]
        regime = "L" if any(m > 0 for m in moved) else "H" if any(m < 0 for m in moved) else "I"
        stats[regime][0] += 1
        stats[regime][1] += live
        stats[regime][2] += len(sites)

    # Planner gather (neighborhood_plan.cpp): one constant extent per axis, sized for the chunk's
    # stride groups, and as many bricks as the worst-aligned chunk origin needs.
    chunk_sites = chunk_bricks * brick
    extent = min(window + (math.ceil(chunk_sites / stride) - 1) * stride, volume)
    gather = 0
    for c in range(math.ceil(bricks / chunk_bricks)):
        first = c * chunk_sites
        origin = min(context_window_origin(first // stride, stride, window, volume, snap), volume - extent)
        gather = max(gather, math.ceil((origin % brick + extent) / brick))
    gather = min(gather, bricks)
    return stats, gather, window


def config_table(volume, window, stride, brick, chunk):
    axes = [axis_stats(*args) for args in zip(volume, window, stride, brick, chunk)]
    gather = math.prod(a[1] for a in axes)
    keys = math.prod(a[2] for a in axes)
    rows = {}
    for combo in itertools.product(REGIMES, repeat=3):
        per = [a[0][r] for a, r in zip(axes, combo)]
        n = math.prod(p[0] for p in per)
        if n == 0:
            continue
        live = math.prod(p[1] for p in per)
        queries = math.prod(p[2] for p in per)
        rows["".join(combo)] = dict(bricks=n, gathered=n * gather, live=live, exact=queries * keys / 1024)
    return rows, tuple(a[1] for a in axes)


def accepted(stride, brick, chunk):
    """What the op accepts today without DIFFVAE_NA_UNSAFE_CHUNK."""
    multi = math.prod(chunk) > 1
    if multi and tuple(c * b for c, b in zip(chunk, brick)) != tuple(stride):
        return "unsafe-chunk"
    return "ok"


def linear_order_today(volume, window):
    """Stage 1's linear-order executor: dense masked SDPA per plan_na3d tile group."""
    from models.tt_dit.layers.neighborhood_attention_plan import plan_na3d

    plan = plan_na3d(volume, window)
    total = 0
    for group in plan.groups:
        total += len(group.query_slices) * math.ceil(group.n_queries / 32) * math.ceil(group.n_keys / 32)
    return total, plan.tile


def summarize(rows):
    keys = ("bricks", "gathered", "live", "exact")
    return {k: sum(r[k] for r in rows.values()) for k in keys}


def collapse(rows):
    """Group the 27 regimes into interior, face (one edge axis), edge (two) and corner (three)."""
    out = {}
    for name, r in rows.items():
        cls = ("interior", "face", "edge", "corner")[sum(c != "I" for c in name)]
        acc = out.setdefault(cls, dict(bricks=0, gathered=0, live=0, exact=0.0))
        for k in acc:
            acc[k] += r[k]
    return out


def check_planner():
    """Cross-check the gather against ttnn.transformer.neighborhood_plan where the op accepts it."""
    import ttnn

    bad = 0
    for stage, cfg in STAGES.items():
        configs = [("today", TODAY_BRICK.get(stage), (1, 1, 1), (1, 1, 1))]
        configs += [(lbl, b, c, s) for lbl, b, c in SHAPES for s in STRIDES]
        for lbl, brick, chunk, stride in configs:
            if brick is None or accepted(stride, brick, chunk) != "ok":
                continue
            _, ours = config_table(cfg["volume"], cfg["window"], stride, brick, chunk)
            try:
                plan = ttnn.transformer.neighborhood_plan(
                    list(cfg["volume"]), list(cfg["window"]), list(stride), list(brick), query_chunk_bricks=list(chunk)
                )
            except Exception as e:  # geometry the op refuses
                print(f"S{stage} {lbl} stride {stride}: planner refused ({str(e).splitlines()[0][:90]})")
                continue
            theirs = tuple(plan["gather_bricks"])
            flag = "ok" if theirs == ours else "MISMATCH"
            bad += flag != "ok"
            print(f"S{stage} {lbl:8} stride {stride}: ours {ours} planner {theirs} {flag}")
    return bad


def fmt(x):
    return f"{x / 1e3:,.1f}k" if x >= 1e4 else f"{x:,.0f}"


def markdown():
    out = []
    totals = []
    for stage, cfg in STAGES.items():
        vol, win = cfg["volume"], cfg["window"]
        out.append(f"\n### Stage {stage}: volume {vol}, window {win}, {cfg['depth']} NA layers\n")
        configs = []
        if stage in TODAY_BRICK:
            configs.append((f"today {TODAY_BRICK[stage]}", TODAY_BRICK[stage], (1, 1, 1), (1, 1, 1)))
        configs += [(lbl, b, c, s) for lbl, b, c in SHAPES for s in STRIDES]
        out.append(
            "| brick/Q-chunk | stride | gather bricks (t,h,w) | slots | op today | "
            "gathered | live | exact | gath/exact | live/exact | gath/live |"
        )
        out.append("|---|---|---|---|---|---|---|---|---|---|---|")
        if stage == 1:
            lo, tile = linear_order_today(vol, win)
            exact = math.prod(vol) * math.prod(win) / 1024
            out.append(
                f"| today: linear-order, tile {tile} | (1,1,1) | - | - | ok | {fmt(lo)} | - | {fmt(exact)} | "
                f"{lo / exact:.2f} | - | - |"
            )
            totals.append((stage, "today linear-order", lo, None, exact))
        for lbl, brick, chunk, stride in configs:
            rows, g = config_table(vol, win, stride, brick, chunk)
            s = summarize(rows)
            out.append(
                f"| {lbl} | {stride} | {g} | {math.prod(g)} | {accepted(stride, brick, chunk)} | "
                f"{fmt(s['gathered'])} | {fmt(s['live'])} | {fmt(s['exact'])} | "
                f"{s['gathered'] / s['exact']:.2f} | {s['live'] / s['exact']:.2f} | {s['gathered'] / s['live']:.2f} |"
            )
            totals.append((stage, f"{lbl} s{stride}", s["gathered"], s["live"], s["exact"]))
    return "\n".join(out), totals


def regime_markdown(stage, brick, chunk, stride):
    cfg = STAGES[stage]
    rows, g = config_table(cfg["volume"], cfg["window"], stride, brick, chunk)
    lines = [
        f"\nStage {stage}, brick {brick}, chunk {chunk}, stride {stride}, gather {g} = {math.prod(g)} slots\n",
        "| regime (t,h,w) | query bricks | gathered/brick | live/brick | exact/brick | live/gathered |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in sorted(rows.items(), key=lambda kv: -kv[1]["bricks"]):
        n = r["bricks"]
        lines.append(
            f"| {name} | {n:,} | {r['gathered'] / n:.0f} | {r['live'] / n:.1f} | {r['exact'] / n:.1f} | "
            f"{r['live'] / r['gathered']:.2f} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-planner", action="store_true")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--regimes", nargs="*", default=[], help="stage:brick:chunk:stride, e.g. 5:2,4,4:1,1,1:1,1,1")
    args = ap.parse_args()
    if args.check_planner:
        sys.exit(1 if check_planner() else 0)
    if args.markdown:
        print(markdown()[0])
    for spec in args.regimes:
        st, b, c, s = spec.split(":")
        print(regime_markdown(int(st), *(tuple(int(x) for x in v.split(",")) for v in (b, c, s))))
