#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Config space for the MinimalMatmulStridedReduceScatterAsync sweep.

Single source of truth: the orchestrator imports this to budget batches, and the sweep test imports
it to run them. Keeping one enumeration means a batch estimate can never disagree with what the test
actually executes.
"""

TILE = 32
DST_MAX_TILES = 4  # fp32 dest acc: half-sync DST holds 4 tiles

# Blackhole compute grid. RS worker cores are placed starting at row mm_core_grid.y, so MM rows and
# RS rows are a partition of these 10 rows -- the central tradeoff this sweep explores.
BH_GRID_X, BH_GRID_Y = 12, 10

# Bytes available per core for (matmul CBs + the L1-sharded MM output tensor) together. Derived from
# the clash the device reports when the budget is blown: for the 4864x4096x4096 8/8/8 config the CB
# region ended at 1160064 with the sharded output already placed at 1128192, i.e. 31872 bytes of
# overlap. Adding that overlap back to (CB estimate + sharded bytes) for that config pins the ceiling
# here, and it reproduces the reported overshoot exactly. An estimate, not a guarantee: it exists to
# skip configs that would TT_THROW mid-build and leave the device dirty for the next config.
L1_BUDGET = 1444736

BF16_TILE = 2048
FP32_TILE = 4096


def round_up(v, m):
    return ((v + m - 1) // m) * m


def div_up(v, m):
    return (v + m - 1) // m


def pick_subblock(mb_t, nb_t, dst_max=DST_MAX_TILES):
    """Largest (subblock_h, subblock_w) dividing the block and fitting DST; ties favour square."""
    best = (1, 1)
    for sh in range(1, mb_t + 1):
        if mb_t % sh:
            continue
        for sw in range(1, nb_t + 1):
            if nb_t % sw or sh * sw > dst_max:
                continue
            cur, bp = sh * sw, best[0] * best[1]
            if cur > bp or (cur == bp and abs(sh - sw) < abs(best[0] - best[1])):
                best = (sh, sw)
    return best


def mm_per_core(M, N, gx, gy):
    """Tiles of M and N each matmul core owns. The fused op disables core-grid transpose, so M is
    parallelized on grid.y and N on grid.x."""
    Mt, Nt = M // TILE, N // TILE
    return round_up(Mt, gy) // gy, round_up(Nt, gx) // gx


def rs_rows(num_links, workers_per_link, gx, num_directions=2, mux_per_direction=1):
    """Grid rows the reduce-scatter workers occupy. Mirrors reduce_scatter_core_count_per_link:
    per link, each direction needs its mux cores plus its workers."""
    per_link = num_directions * (mux_per_direction + workers_per_link)
    return div_up(num_links * per_link, gx)


def cb_bytes(mb_t, kb_t, nb_t):
    """Per-core matmul CB footprint estimate: in0/in1/out double-buffered bf16, plus the fp32
    accumulation intermediate."""
    return 2 * BF16_TILE * (mb_t * kb_t + kb_t * nb_t + mb_t * nb_t) + FP32_TILE * mb_t * nb_t


def sharded_out_bytes(Mt_pc, Nt_pc):
    """The MM output is block-sharded into the matmul cores' L1 so the RS reader takes it straight
    from L1; that buffer competes with the CBs for the same per-core budget."""
    return Mt_pc * Nt_pc * BF16_TILE


def feasible(cfg):
    """(ok, reason) for one config. Every rejection here is something the device would otherwise
    fail on -- a TT_FATAL at validation or a CB/L1 clash at program build."""
    M, N = cfg["M"], cfg["N"]
    gx, gy = cfg["gx"], cfg["gy"]
    mb, kb, nb = cfg["mb"], cfg["kb"], cfg["nb"]
    Mt, Kt, Nt = M // TILE, cfg["K"] // TILE, N // TILE

    if gx < 2 or gy < 2:
        return False, "grid must be >= 2x2"
    if gx > BH_GRID_X or gy > BH_GRID_Y:
        return False, "grid exceeds device"

    rows = rs_rows(cfg["links"], cfg["workers"], gx)
    if gy + rows > BH_GRID_Y:
        return False, f"gy={gy} + {rows} RS rows > {BH_GRID_Y}"

    if kb > Kt:
        return False, "K block exceeds K"
    Mt_pc, Nt_pc = mm_per_core(M, N, gx, gy)
    if nb > Nt // gx:
        return False, f"N block {nb} > {Nt // gx} tiles of work per core"
    if mb > Mt_pc:
        return False, f"M block {mb} > {Mt_pc} per core"

    if mb % cfg["sbh"] or nb % cfg["sbw"]:
        return False, "block not a subblock multiple"

    n_blocks_pc = div_up(Nt_pc, nb)
    if cfg["chunk"] > n_blocks_pc:
        return False, f"chunk {cfg['chunk']} > {n_blocks_pc} N blocks per core"

    l1 = cb_bytes(mb, kb, nb) + sharded_out_bytes(Mt_pc, Nt_pc)
    if l1 > L1_BUDGET:
        return False, f"L1 {l1} > {L1_BUDGET}"

    return True, ""


def _mk(M, K, N, gx, gy, mb, kb, nb, chunk, links, workers, packet, mode):
    sbh, sbw = pick_subblock(mb, nb)
    return dict(
        M=M,
        K=K,
        N=N,
        gx=gx,
        gy=gy,
        mb=mb,
        kb=kb,
        nb=nb,
        sbh=sbh,
        sbw=sbw,
        chunk=chunk,
        links=links,
        workers=workers,
        packet=packet,
        mode=mode,
    )


CFG_FIELDS = ("M", "K", "N", "gx", "gy", "mb", "kb", "nb", "sbh", "sbw", "chunk", "links", "workers", "packet", "mode")


def cfg_key(rec):
    """Stable identity of a config, from its values alone.

    Resume state is keyed on this rather than on position in the enumeration: editing the space
    reorders indices, and index-keyed state would then mark the wrong configs as already done.
    Accepts a manifest record too, since those carry the config fields plus extras.
    """
    return "|".join(str(rec[f]) for f in CFG_FIELDS)


def _divisors_le(n, lo=2):
    return [d for d in range(lo, n + 1) if n % d == 0]


def enumerate_configs(shape, stage, modes=("fused",)):
    """Feasible configs for a named shape and stage.

    Stage 'structural' walks the grid partition, link/worker counts, chunk width and packet size at
    one fixed sane blocking -- the axes that matter when a shape is overhead-bound. Stage 'blocking'
    walks M/K/N blocks with the structural axes pinned to whatever won stage 1, so the two passes
    stay small enough to finish on a single galaxy.
    """
    M, K, N = SHAPES[shape]
    Nt, Kt = N // TILE, K // TILE
    out = []

    if stage == "structural":
        for gx in [g for g in (2, 4, 6, 8, 12) if Nt % g == 0 or g == 12]:
            for gy in STRUCTURAL_GY[shape]:
                Mt_pc, Nt_pc = mm_per_core(M, N, gx, gy)
                mb = min(Mt_pc, 4)
                nb = min(Nt // gx, 8) or 1
                kb = min(Kt, 8)
                for links in (1, 2, 4):
                    for workers in (1, 2, 3, 4, 5):
                        for chunk in (1, 2):
                            for packet in (4096, 8192):
                                for mode in modes:
                                    c = _mk(M, K, N, gx, gy, mb, kb, nb, chunk, links, workers, packet, mode)
                                    ok, _ = feasible(c)
                                    if ok:
                                        out.append(c)
    elif stage == "blocking":
        pin = BLOCKING_PIN[shape]
        gx, gy = pin["gx"], pin["gy"]
        Mt_pc, Nt_pc = mm_per_core(M, N, gx, gy)
        for mb in range(1, Mt_pc + 1):
            for kb in _divisors_le(Kt) + [Kt]:
                for nb in range(1, min(Nt // gx, 16) + 1):
                    for chunk in (1, 2):
                        for mode in modes:
                            c = _mk(
                                M, K, N, gx, gy, mb, kb, nb, chunk, pin["links"], pin["workers"], pin["packet"], mode
                            )
                            ok, _ = feasible(c)
                            if ok:
                                out.append(c)
    else:
        raise ValueError(f"unknown stage {stage!r}")

    # Dedup: different (mb, nb) can collapse to the same subblock choice, and _divisors_le can repeat Kt.
    seen, uniq = set(), []
    for c in out:
        k = tuple(sorted(c.items()))
        if k not in seen:
            seen.add(k)
            uniq.append(c)
    return uniq


SHAPES = {
    # LTX video FFN ff2, RowParallel reduce-scatter, per device.
    "ff2": (4864, 4096, 4096),
    # LTX single-tile-M shape, per device.
    "small": (32, 2048, 2048),
}

# M is one tile for 'small', so rows past the first only compute padding; 2 is the matmul's floor.
# ff2 has 152 M tiles, so its grid.y is a real axis.
STRUCTURAL_GY = {"ff2": (4, 6, 8), "small": (2, 3, 4)}

# Structural winners, to be replaced with stage-1 results before the blocking pass is meaningful.
BLOCKING_PIN = {
    "ff2": dict(gx=12, gy=8, links=2, workers=3, packet=8192),
    "small": dict(gx=8, gy=2, links=2, workers=3, packet=8192),
}


if __name__ == "__main__":
    import sys

    for shape in sys.argv[1:] or list(SHAPES):
        for stage in ("structural", "blocking"):
            print(f"{shape:6s} {stage:11s} {len(enumerate_configs(shape, stage))} configs")
