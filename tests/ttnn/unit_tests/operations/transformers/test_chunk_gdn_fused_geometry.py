# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-side geometry gates for the fused prep->scan GDN prim (ttnn::prim::chunk_gdn_fused).

No device needed. The fused op's geometry is three pure C++ functions, bound through nanobind:

* ``chunk_gdn_fused_geometry``           — the calibrated cost model (NV, NP, placement, hand-off depth)
                                           the op picks when the fused program config leaves those
                                           fields free;
* ``chunk_gdn_fused_row_local_feasible`` — the row-local feasibility predicate;
* ``chunk_gdn_fused_pool_feasible`` / ``chunk_gdn_fused_pool_home_producers`` — the producer pool's;
* ``chunk_gdn_fused_placement``          — the core map, computed by the SAME function the program
                                           factory calls;
* ``chunk_gdn_fused_item_map``           — the producer map the factory and the three dataflow kernels
                                           share (which producer computes chunk c of head h).

The device tests (test_chunk_gdn_fused.py) can only run the geometry of the attached chip. These
tests take the grid as a parameter, so a geometry tuned for one chip (QB2's 11x10) cannot be baked
in unnoticed: every case runs on 11x10 (QB2), 12x10 (Galaxy2), 13x10 (p150b) and 8x8 (Wormhole).

They check the C++ against an independent Python re-derivation of the same design (inlined below),
and check the factory's core map against the structural predicates the transport relies on:
* every core used at most once, all inside the grid, producers disjoint from receivers;
* each head's receivers a dense rectangle — the multicast destination;
* under row-local placement, no two heads sharing a NoC link (NOC_1 routes -x then -y). A shared
  link is bit-exact and green in every device test but runs 2-3x slower than the model.
"""

import pytest

import ttnn  # noqa: F401  (loads the extension that registers the bindings)
from ttnn._ttnn.operations import transformer as _t

GRIDS = [(11, 10), (12, 10), (13, 10), (8, 8)]
GRID_IDS = [f"{x}x{y}" for x, y in GRIDS]
BHS = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]
VT = 4  # V = 128 -> 4 tiles; invariant across the Qwen GDN family

# ---------------------------------------------------------------------------------------------
# Inlined Python oracle of the fused geometry. Kept self-contained on purpose:
# it is an independent re-derivation of chunk_gdn_device_operation.cpp (choose_fused_geometry,
# fused_row_local_feasible, fused_placement), so a bug has to be made twice to pass.
# Cost-model constants: QB2, re-measured 2026-10-02 (Tracy device time, T=2048 -> NC=64, C=32, K=V=128, medians of
# ops 2-5) on the producer-lever stack.
# ---------------------------------------------------------------------------------------------
_W_P_US = 15.98  # producer item period; flat over BH and the producer count
_T_STEP_US = {1: 3.17, 2: 2.69, 4: 4.45}  # receiver step per V-slice width Vtl, before the per-head slope
_T_STEP_POOL_VTL1_US = 4.25  # Vtl=1 is round-trip bound; a pool's extras lengthen the round trip
_CHAIN_SLOPE_US = 0.012  # receiver chunk period growth per head (shared DRAM / NoC)
_SKEW_A_US, _SKEW_B_US = 4.0, 0.5  # slowest chain behind the median chain: A + B * BH
_TAIL_US = 4.0  # last chunk consumed -> kernel end
_FILL_A_US, _FILL_B_US, _FILL_C_US = 19.5, 0.72, 1.0 / 144.0  # fill = A + min(BH, 24) * (B + C * producers)
_ROW_MAJOR_PACE_US = 8.4  # link-bound chunk period of the row-major placement
# Balance bumps: two bounds within `width` of each other expose each side's jitter to the other, up to `peak`
# when equal. Chain vs producers at Vtl <= 2 (narrower at depth >= 3); home vs extra producers of a pool.
_BALANCE_PEAK, _BALANCE_WIDTH_D2, _BALANCE_WIDTH_D3 = 0.08, 0.25, 0.15
_POOL_BALANCE_PEAK, _POOL_BALANCE_WIDTH = 0.25, 0.25
_POOL_START_US, _POOL_START_SHARE = 20.0, 0.25  # pool, Vtl <= 2: the home producers' first items end later
_POOL_SKEW_US = 4.0  # extra chain skew of a pool
_POOL_CREDIT_D2_US = 0.025  # pool at depth 2, Vtl <= 2: credit round trip exposed per step, per head
_DEPTHS = (2, 3)  # hand-off depths the model chooses between
_NV_CANDIDATES = (1, 2, 4, 8)  # receivers per head the model considers (those dividing Vt)
_HANDOFF_TILES, _PRODUCER_PREP_TILES, _TILE_BYTES, _L1_BUDGET_BYTES = 19, 48, 4096, 1400 * 1024
# Measured phased device time (prep + scan) at NC=64, interpolated linearly in BH; beyond the table scaled by BH/48.
_PHASED_TABLE = ((4, 359.8), (8, 461.5), (12, 603.0), (16, 769.4), (32, 1278.8), (48, 1907.9))


def _fill(bh, producers):
    return _FILL_A_US + min(bh, 24) * (_FILL_B_US + _FILL_C_US * producers)


def _t_step(vtl, pooled=False):
    return _T_STEP_POOL_VTL1_US if pooled and vtl == 1 else _T_STEP_US[vtl]


def _balance(a, b, width, peak):
    if a <= 0 or b <= 0:
        return 0.0
    m = max(a, b)
    return peak * max(0.0, 1.0 - abs(a - b) / (width * m))


def _handoff_fits_l1(vtl, depth):
    tiles = _HANDOFF_TILES * depth + 4 + max(20 * vtl + 1, _PRODUCER_PREP_TILES)
    return tiles * _TILE_BYTES <= _L1_BUDGET_BYTES


def _extras_before(bh, num, den, h, c):
    """Extra chunks of head h in [0, c) under the shared item map (chunk_gdn_fused_map.hpp)."""
    return (c * num * bh + (bh - 1 - h) * den) // (den * bh)


def _pool_load(bh, nc, nph, nx, num, den):
    """Items of the busiest home producer and of the busiest extra: the home chunks of a head are dealt round-robin
    to its NPH home producers, the extra items round-robin to the NX extras."""
    if nx == 0:
        return -(-nc // nph), 0
    nh = max(nc - _extras_before(bh, num, den, h, nc) for h in range(bh))
    ne = (nc * num * bh) // den
    return -(-nh // nph), -(-ne // nx)


def _t_fused(bh, nc, nv, np_, depth, placement, nph=None, num=0, den=1):
    """Device time of one geometry: the pipeline fill, then the slowest of the chain (NC-1 steps plus the head
    skew), the busiest home producer's remaining items and the busiest extra's, stretched by the balance bumps,
    then the tail. placement 2: np_ is the pool size P with nph home producers per head and the extras' share
    num / den; a pool without extras is the per-head geometry it is."""
    vtl = VT // nv
    producers = np_ if placement == 2 else bh * np_
    nph = nph if placement == 2 else np_
    nx = np_ - bh * nph if placement == 2 else 0
    pooled = nx > 0
    share = num / den if pooled else 0.0
    n_home, n_extra = _pool_load(bh, nc, nph, nx, num, den)
    pace = _t_step(vtl, pooled) + _CHAIN_SLOPE_US * bh
    if pooled and vtl <= 2 and depth <= 2:
        pace += _POOL_CREDIT_D2_US * bh
    H = (n_home - 1) * _W_P_US
    X = (n_extra - 1) * _W_P_US if n_extra else 0.0
    if placement == 0 and max(H, X) < 2.0 * (nc - 1) * pace:  # row-major: link-bound unless well production-bound
        pace = max(pace, _ROW_MAJOR_PACE_US)
    C = (nc - 1) * pace + _SKEW_A_US + _SKEW_B_US * bh + (_POOL_SKEW_US if pooled else 0.0)
    pen = start = 0.0
    if vtl <= 2:
        pen = _balance(C, H, _BALANCE_WIDTH_D2 if depth <= 2 else _BALANCE_WIDTH_D3, _BALANCE_PEAK)
        if pooled:  # the home/extra balance matters only where the producers bound the run
            m = max(C, H, X)
            g = max(0.0, 1.0 - (m - max(H, X)) / (_POOL_BALANCE_WIDTH * m))
            pen = max(pen, g * _balance(H, X, _POOL_BALANCE_WIDTH, _POOL_BALANCE_PEAK))
            start = _POOL_START_US * min(1.0, share / _POOL_START_SHARE)
    return _fill(bh, producers) + max(C, H + start, X) * (1.0 + pen) + _TAIL_US


def _t_phased_us(bh, nc):
    pts = _PHASED_TABLE
    x = min(bh, pts[-1][0])
    i = 0
    while i + 2 < len(pts) and x > pts[i + 1][0]:
        i += 1
    (x0, y0), (x1, y1) = pts[i], pts[i + 1]
    t = y0 + (x - x0) / (x1 - x0) * (y1 - y0)
    if bh > pts[-1][0]:
        t *= bh / pts[-1][0]
    return t * nc / 64.0


def _row_major_feasible(gx, gy, bh, nv):
    return nv in (1, 2, 4) and VT % nv == 0 and bh >= 1 and gx // nv >= 1 and bh <= (gx // nv) * gy


def _row_local_feasible(gx, gy, bh, nv, np_):
    L = nv + np_
    if L > gx or nv < 1 or np_ < 1:
        return False
    k = gx // L
    if bh <= k * gy:
        return True
    rem, wl = bh - k * gy, gx - k * L
    if wl < 1:
        return False
    rw = min(nv, wl)
    if nv % rw:
        return False
    return rem * (nv // rw + -(-np_ // wl)) <= gy


def _pool_home_producers(gx, gy, bh, nv, P):
    """Placement 2: home producers per head — the largest NPH with BH*NPH <= P that has a row-local layout."""
    for nph in range(min(P // bh, gx), 0, -1):
        if _row_local_feasible(gx, gy, bh, nv, nph):
            return nph
    return 0


def _pool_feasible(gx, gy, bh, nv, P):
    return nv >= 1 and P >= 1 and bh * nv + P <= gx * gy and _pool_home_producers(gx, gy, bh, nv, P) >= 1


PER_HEAD, POOL, BOTH = 0, 1, 2  # chunk_gdn_fused_geometry's `candidates`


def _choose_geometry(grid, bh, nc, fixed_nv=0, fixed_np=0, fixed_nbuf=0, candidates=BOTH, fixed_share=-1.0):
    """Per-head candidates: over NV | Vt, NP with a feasible row-local layout and depth in {2, 3} (a pinned
    depth as is), minimise T_fused; with no row-local layout, fall back to row-major placement (link-bound).
    Pool candidates: the same NV set, P = every core the receivers leave (or the pinned size), at most BH*NC,
    with the extras' share pinned or the best of every share from the balanced NX / P down to 0 (ties -> the
    larger share); a pool with no extras is the per-head geometry (NV, NPH). Ties -> fewer cores in total,
    then smaller NV, then the shallower ring."""
    gx, gy = grid
    depths = (fixed_nbuf,) if fixed_nbuf else _DEPTHS
    best = None

    def consider(nv, np_, placement, nph=None, num=0, den=1):
        nonlocal best
        cores = bh * nv + (np_ if placement == 2 else bh * np_)
        for depth in depths:
            if not fixed_nbuf and not _handoff_fits_l1(VT // nv, depth):
                continue
            key = (_t_fused(bh, nc, nv, np_, depth, placement, nph, num, den), cores, nv, depth)
            if best is None or key < best[0]:
                best = (key, nv, np_, placement, depth, num if num else 0, den if num else 1)

    def consider_pool(nv, P, nph):
        nx = P - bh * nph
        nums = [int(fixed_share * P + 0.5)] if fixed_share >= 0 else range(nx, -1, -1)
        for depth in depths:
            if not fixed_nbuf and not _handoff_fits_l1(VT // nv, depth):
                continue
            t_best, num_best = None, None
            for num in nums:
                t = _t_fused(bh, nc, nv, P, depth, 2, nph, num, P)
                if t_best is None or t < t_best:
                    t_best, num_best = t, num
            key = (t_best, bh * nv + P, nv, depth)
            nonlocal best
            if best is None or key < best[0]:
                best = (key, nv, P, 2, depth, num_best, P)

    def nv_ok(nv):
        return VT % nv == 0 and (VT // nv) in _T_STEP_US and (not fixed_nv or nv == fixed_nv)

    if candidates != POOL:
        for nv in _NV_CANDIDATES:
            if not nv_ok(nv):
                continue
            for np_ in range(1, gx - nv + 1):
                np_eff = min(np_, nc)
                if bh * (nv + np_eff) > gx * gy:
                    break
                if fixed_np and np_eff != min(fixed_np, nc):
                    continue
                if _row_local_feasible(gx, gy, bh, nv, np_eff):
                    consider(nv, np_eff, 1)
        if best is None:
            for nv in _NV_CANDIDATES:
                if not nv_ok(nv) or not _row_major_feasible(gx, gy, bh, nv) or gx * gy - bh * nv < 1:
                    continue
                np_ = min(fixed_np, nc) if fixed_np else min((gx * gy - bh * nv) // bh, nc)
                if np_ >= 1 and bh * (nv + np_) <= gx * gy:
                    consider(nv, np_, 0)
    if candidates != PER_HEAD:
        for nv in _NV_CANDIDATES:
            if not nv_ok(nv) or bh * nv >= gx * gy:
                continue
            P = min(fixed_np if fixed_np else gx * gy - bh * nv, bh * nc)
            if not _pool_feasible(gx, gy, bh, nv, P):
                continue
            nph = _pool_home_producers(gx, gy, bh, nv, P)
            if P == bh * nph:  # no extras: the per-head geometry (NV, NPH) itself
                consider(nv, nph, 1) if candidates == BOTH else consider(nv, P, 2, nph)
            else:
                consider_pool(nv, P, nph)
    t_ph = _t_phased_us(bh, nc)
    none = {
        "nv": None,
        "np": None,
        "placement": None,
        "nbuf": None,
        "num": None,
        "T_fused": None,
        "T_phased": t_ph,
        "fused_pays": False,
    }
    if best is None:
        return none
    (t, _, _, _), nv, np_, pl, depth, num, den = best
    return {
        "nv": nv,
        "np": np_,
        "placement": pl,
        "nbuf": depth,
        "num": num,
        "T_fused": t,
        "T_phased": t_ph,
        "fused_pays": t < t_ph,
    }


def _placement_row_major(gx, gy, bh, nv, np_):
    """Placement 0: head h's 1xNV receivers at row h // HPR, columns (h % HPR)*NV ..; the producers on
    the remaining cores row-major from row 0, producer p serving head p // NP."""
    hpr = gx // nv
    rcv = [None] * (bh * nv)
    for h in range(bh):
        y0, x0 = h // hpr, (h % hpr) * nv
        for v in range(nv):
            rcv[h * nv + v] = (x0 + v, y0)
    taken = set(rcv)
    free = [(x, y) for y in range(gy) for x in range(gx) if (x, y) not in taken]
    return rcv, free[: bh * np_]


def _placement_row_local(gx, gy, bh, nv, np_):
    """Placement 1: k = grid_x // (NV+NP) heads per row, each in its own column segment (receivers
    first, producers east of them); heads beyond k*grid_y as vertical blocks in the leftover columns
    (an rw x rh receiver rectangle on top, the producers row-major below it)."""
    L = nv + np_
    k = gx // L
    rcv = [None] * (bh * nv)
    prod = [None] * (bh * np_)
    n_row_heads = min(bh, k * gy)
    for h in range(n_row_heads):
        row, xs = h // k, (h % k) * L
        for v in range(nv):
            rcv[h * nv + v] = (xs + v, row)
        for j in range(np_):
            prod[h * np_ + j] = (xs + nv + j, row)
    rem, xl = bh - n_row_heads, k * L
    if rem:
        wl = gx - xl
        rw = min(nv, wl)
        rh = nv // rw
        block_h = rh + -(-np_ // wl)
        for kk in range(rem):
            h, y0 = n_row_heads + kk, kk * block_h
            for v in range(nv):
                rcv[h * nv + v] = (xl + v % rw, y0 + v // rw)
            for j in range(np_):
                prod[h * np_ + j] = (xl + j % wl, y0 + rh + j // wl)
    return rcv, prod


def _placement_pool(gx, gy, bh, nv, P):
    """Placement 2: the row-local map of NPH = _pool_home_producers home producers per head, then the
    P - BH*NPH extras on the remaining cores row-major."""
    nph = _pool_home_producers(gx, gy, bh, nv, P)
    rcv, prod = _placement_row_local(gx, gy, bh, nv, nph)
    taken = set(rcv) | set(prod)
    extras = [(x, y) for y in range(gy) for x in range(gx) if (x, y) not in taken]
    return rcv, prod + extras[: P - bh * nph], nph


def _rect(cores):
    xs, ys = [c[0] for c in cores], [c[1] for c in cores]
    return min(xs), min(ys), max(xs), max(ys)


def _shared_links(rcv, prod, nv, np_):
    """NOC_1 routes -x then -y. Every producer must lie east of (or in) its head's receiver columns and
    in a row >= its receivers' rows; its horizontal leg in its own row and its vertical leg in the
    receiver column must not overlap another head's legs. Returns the violations (empty == pass)."""
    bad, hlegs, vlegs = [], {}, {}
    for p, (px, py) in enumerate(prod):
        h = p // np_
        x0, y0, x1, y1 = _rect(rcv[h * nv : (h + 1) * nv])
        if px < x0 or py < y0:
            bad.append(f"head {h} producer {p} at {(px, py)} is west of / above its receivers {(x0, y0, x1, y1)}")
            continue
        for x in range(x1 + 1, px + 1):
            prev = hlegs.setdefault((x, py), h)
            if prev != h:
                bad.append(f"heads {prev} and {h} share horizontal link cell {(x, py)}")
        for y in range(y1 + 1, py + 1):
            prev = vlegs.setdefault((min(px, x1), y), h)
            if prev != h:
                bad.append(f"heads {prev} and {h} share vertical link cell {(min(px, x1), y)}")
    return bad


def _feasible_layouts(gx, gy, bh):
    """Every (NV, NP, placement) the factory accepts for this grid and BH."""
    out = []
    for nv in (1, 2, 4):
        for np_ in range(1, gx):
            if bh * (nv + np_) > gx * gy:
                break
            if _row_major_feasible(gx, gy, bh, nv):
                out.append((nv, np_, 0))
            if _row_local_feasible(gx, gy, bh, nv, np_):
                out.append((nv, np_, 1))
    return out


# ---------------------------------------------------------------------------------------------
# The cost model and the feasibility predicate mirror the oracle
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("candidates", [PER_HEAD, POOL, BOTH], ids=["per_head", "pool", "both"])
@pytest.mark.parametrize("nc", [8, 64])
@pytest.mark.parametrize("bh", BHS)
@pytest.mark.parametrize("grid", GRIDS, ids=GRID_IDS)
def test_cost_model_mirror(grid, bh, nc, candidates):
    """The C++ cost model and the oracle pick the same (NV, NP, placement, depth) and agree on T_fused and
    T_phased — so the op's default dispatch (both candidate kinds) is the documented model on every grid,
    and so are the geometries producer_pool=False / True resolve to."""
    nv, np_, pl, nbuf, t_f, t_ph, pays, num = _t.chunk_gdn_fused_geometry(
        grid[0], grid[1], bh, nc, VT, candidates=candidates
    )
    o = _choose_geometry(grid, bh, nc, candidates=candidates)
    if o["nv"] is None:
        assert nv == 0, f"oracle: no fused geometry fits; C++ picked NV={nv} NP={np_}"
        assert not pays
        return
    assert (nv, np_, pl, nbuf, num) == (
        o["nv"],
        o["np"],
        o["placement"],
        o["nbuf"],
        o["num"],
    ), f"C++ {(nv, np_, pl, nbuf, num)} vs oracle {o}"
    assert abs(t_f - o["T_fused"]) < 0.5 and abs(t_ph - o["T_phased"]) < 0.5, (t_f, t_ph, o)
    assert pays == o["fused_pays"]


@pytest.mark.parametrize("bh", BHS)
@pytest.mark.parametrize("grid", GRIDS, ids=GRID_IDS)
def test_row_local_feasibility_mirror(grid, bh):
    """The C++ row-local feasibility predicate mirrors the oracle for every (NV, NP)."""
    for nv in (1, 2, 4):
        for np_ in range(1, grid[0]):
            got = bool(_t.chunk_gdn_fused_row_local_feasible(grid[0], grid[1], bh, nv, np_))
            assert got == _row_local_feasible(grid[0], grid[1], bh, nv, np_), (nv, np_, got)


@pytest.mark.parametrize("bh", BHS)
@pytest.mark.parametrize("grid", GRIDS, ids=GRID_IDS)
def test_pool_feasibility_mirror(grid, bh):
    """The C++ pool predicates (feasible, home producers per head) mirror the oracle for every pool size."""
    gx, gy = grid
    for nv in (1, 2, 4):
        for P in range(1, gx * gy - bh * nv + 1):
            nph = _t.chunk_gdn_fused_pool_home_producers(gx, gy, bh, nv, P)
            assert nph == _pool_home_producers(gx, gy, bh, nv, P), (nv, P, nph)
            got = bool(_t.chunk_gdn_fused_pool_feasible(gx, gy, bh, nv, P))
            assert got == _pool_feasible(gx, gy, bh, nv, P), (nv, P, got)
            assert got == (nph >= 1), (nv, P, got, nph)


@pytest.mark.parametrize(
    "fixed",
    [
        (1, 0, 0),
        (2, 0, 0),
        (4, 0, 0),
        (0, 1, 0),
        (0, 2, 0),
        (0, 5, 0),
        (0, 8, 0),
        (2, 2, 0),
        (4, 5, 0),
        (0, 0, 2),
        (0, 0, 3),
        (0, 0, 4),
        (2, 7, 2),
        (4, 5, 1),
        (1, 4, 3),
    ],
    ids=lambda v: "nv%d_np%d_d%d" % v,
)
@pytest.mark.parametrize("bh", [4, 12, 16, 32])
@pytest.mark.parametrize("grid", [(11, 10), (12, 10)], ids=["11x10", "12x10"])
def test_constrained_choice_mirror(grid, bh, fixed):
    """A partially pinned fused config (any of num_receivers / num_producers / handoff_depth) fixes those
    fields; C++ and the oracle fill in the rest identically, agree on T_fused, and the result always fits
    the grid."""
    fnv, fnp, fnb = fixed
    nv, np_, pl, nbuf, t_f, _, _, num = _t.chunk_gdn_fused_geometry(grid[0], grid[1], bh, 16, VT, fnv, fnp, fnb)
    o = _choose_geometry(grid, bh, 16, fixed_nv=fnv, fixed_np=fnp, fixed_nbuf=fnb)
    if o["nv"] is None:
        assert nv == 0
        return
    assert (nv, np_, pl, nbuf, num) == (o["nv"], o["np"], o["placement"], o["nbuf"], o["num"]), (
        (nv, np_, pl, nbuf, num),
        o,
    )
    assert abs(t_f - o["T_fused"]) < 0.5, (t_f, o)
    assert bh * nv + (np_ if pl == 2 else bh * np_) <= grid[0] * grid[1]
    if fnv:
        assert nv == fnv
    if fnp:
        assert np_ == min(fnp, 16)
    if fnb:
        assert nbuf == fnb


# ---------------------------------------------------------------------------------------------
# The factory's own core map, on every grid
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("bh", BHS)
@pytest.mark.parametrize("grid", GRIDS, ids=GRID_IDS)
def test_factory_placement(grid, bh):
    """For every layout the factory accepts, its core map (a) equals the oracle's and (b) satisfies the
    structural predicates the hand-off relies on. Overlapping cores would alias CBs silently; a
    receiver set that is not a dense rectangle is not a valid multicast destination."""
    gx, gy = grid
    layouts = _feasible_layouts(gx, gy, bh)
    if not layouts:
        pytest.skip(f"no fused layout fits BH={bh} on {gx}x{gy}")
    for nv, np_, pl in layouts:
        tag = f"NV={nv} NP={np_} placement={pl}"
        rcv, prod = _t.chunk_gdn_fused_placement(gx, gy, bh, nv, np_, pl)
        rcv, prod = [tuple(c) for c in rcv], [tuple(c) for c in prod]
        oracle = (_placement_row_major if pl == 0 else _placement_row_local)(gx, gy, bh, nv, np_)
        assert (rcv, prod) == oracle, f"{tag}: the factory's core map differs from the oracle's"

        assert len(rcv) == bh * nv and len(prod) == bh * np_, tag
        assert all(0 <= x < gx and 0 <= y < gy for x, y in rcv + prod), f"{tag}: core outside the grid"
        assert len(set(rcv + prod)) == len(rcv) + len(prod), f"{tag}: a core is used twice"
        for h in range(bh):
            cores = rcv[h * nv : (h + 1) * nv]
            x0, y0, x1, y1 = _rect(cores)
            assert (x1 - x0 + 1) * (y1 - y0 + 1) == nv, f"{tag}: head {h} receivers {cores} not a dense rectangle"
            if pl == 0:
                assert y0 == y1, f"{tag}: head {h} receivers {cores} cross a grid row"


@pytest.mark.parametrize("bh", BHS)
@pytest.mark.parametrize("grid", GRIDS, ids=GRID_IDS)
def test_pool_placement(grid, bh):
    """Placement 2 for every NV and a spread of pool sizes: the factory's core map equals the oracle's
    (the row-local map of the home producers, then the extras row-major), uses every core at most once,
    keeps each head's receivers a dense rectangle, and its first BH*NPH producers ARE the row-local
    map's — the home producers keep their head's traffic in its own row."""
    gx, gy = grid
    tested = 0
    for nv in (1, 2, 4):
        free = gx * gy - bh * nv
        for P in sorted({bh, bh + 1, 2 * bh, free // 2, free - 1, free}):
            if P < 1 or P > free or not _pool_feasible(gx, gy, bh, nv, P):
                continue
            tag = f"NV={nv} P={P}"
            rcv, prod = _t.chunk_gdn_fused_placement(gx, gy, bh, nv, P, 2)
            rcv, prod = [tuple(c) for c in rcv], [tuple(c) for c in prod]
            o_rcv, o_prod, nph = _placement_pool(gx, gy, bh, nv, P)
            assert (rcv, prod) == (o_rcv, o_prod), f"{tag}: the factory's core map differs from the oracle's"
            assert nph == _t.chunk_gdn_fused_pool_home_producers(gx, gy, bh, nv, P)
            assert len(rcv) == bh * nv and len(prod) == P, tag
            assert all(0 <= x < gx and 0 <= y < gy for x, y in rcv + prod), f"{tag}: core outside the grid"
            assert len(set(rcv + prod)) == len(rcv) + len(prod), f"{tag}: a core is used twice"
            rl_rcv, rl_prod = _placement_row_local(gx, gy, bh, nv, nph)
            assert rcv == rl_rcv and prod[: bh * nph] == rl_prod, f"{tag}: home producers are not the row-local map"
            for h in range(bh):
                cores = rcv[h * nv : (h + 1) * nv]
                x0, y0, x1, y1 = _rect(cores)
                assert (x1 - x0 + 1) * (y1 - y0 + 1) == nv, f"{tag}: head {h} receivers {cores} not a dense rectangle"
            tested += 1
    if not tested:
        pytest.skip(f"no producer pool fits BH={bh} on {gx}x{gy}")


def _item_map_properties(bh, nc, nph, nx, num, den):
    """Every (head, chunk) served exactly once, each producer's chunks non-decreasing, owner[h][c] the
    producer whose list holds (h, c). Returns the per-producer item counts."""
    items, owner = _t.chunk_gdn_fused_item_map(bh, nc, nph, nx, num, den)
    assert len(items) == bh * nph + nx and len(owner) == bh and all(len(row) == nc for row in owner)
    seen = {}
    for p, lst in enumerate(items):
        chunks = [c for _, c in lst]
        assert chunks == sorted(chunks), f"producer {p}: chunks {chunks} not non-decreasing"
        for h, c in lst:
            assert 0 <= h < bh and 0 <= c < nc, (p, h, c)
            assert (h, c) not in seen, f"({h}, {c}) served by producers {seen[(h, c)]} and {p}"
            seen[(h, c)] = p
            assert owner[h][c] == p, f"owner[{h}][{c}] = {owner[h][c]}, but producer {p} lists it"
    assert len(seen) == bh * nc, f"{bh * nc - len(seen)} items unserved"
    return [len(lst) for lst in items]


@pytest.mark.parametrize(
    "bh, nc, nph, nx, num, den",
    [
        (16, 64, 3, 30, 30, 78),  # BH=16 NV=2 pool of 78 on 11x10, balanced share 30/78
        (16, 64, 3, 30, 26, 78),  # same pool, the extras at 1/3
        (12, 64, 7, 2, 2, 86),  # BH=12 NV=2 pool of 86: two extras
        (8, 64, 9, 22, 22, 94),  # BH=8 NV=2 pool of 94
        (48, 64, 1, 14, 14, 62),  # BH=48 NV=1 pool of 62
        (16, 9, 3, 30, 30, 78),  # NC not a multiple of anything
        (16, 64, 3, 30, 78, 78),  # every chunk to the extras: home producers idle
        (16, 64, 3, 30, 0, 78),  # no chunk to the extras: they idle
        (16, 2, 3, 30, 30, 78),  # fewer chunks than home producers: idle producers of both kinds
        (3, 5, 2, 7, 7, 13),  # small odd sizes
    ],
)
def test_pool_item_map(bh, nc, nph, nx, num, den):
    """The shared producer map (one formula in the factory and the three kernels): a partition of the
    items into non-decreasing chunk lists with a consistent owner function, and loads balanced to within
    one item per producer kind."""
    counts = _item_map_properties(bh, nc, nph, nx, num, den)
    home, extras = counts[: bh * nph], counts[bh * nph :]
    assert max(home) - min(home) <= 1, f"home loads {min(home)}..{max(home)}"
    if extras:
        assert max(extras) - min(extras) <= 1, f"extra loads {min(extras)}..{max(extras)}"
        n_extra_items = (nc * num * bh) // den
        assert sum(extras) == n_extra_items and sum(home) == nc * bh - n_extra_items


def test_pool_item_map_extras_step_chunks():
    """An extra's successive items step through increasing chunks (phased per head) instead of the same
    chunk of BH/NX heads: with two extras at BH=12 (pool of 86) each item is ~7 chunks after the previous,
    so a chunk needed by every head at once is never queued BH/NX deep on one core."""
    items, _ = _t.chunk_gdn_fused_item_map(12, 64, 7, 2, 2, 86)
    for lst in items[84:]:
        chunks = [c for _, c in lst]
        assert len(chunks) >= 8 and all(5 <= b - a <= 9 for a, b in zip(chunks, chunks[1:])), chunks


@pytest.mark.parametrize("bh, nc, np_", [(12, 8, 3), (12, 64, 7), (16, 1, 1), (4, 9, 4)])
def test_item_map_per_head_form(bh, nc, np_):
    """NX = 0 is the per-head form: producer h*NP + j owns chunks c = j, j+NP, ... of head h — the mapping
    placements 0 and 1 have always used."""
    items, owner = _t.chunk_gdn_fused_item_map(bh, nc, np_, 0, 0, 1)
    for h in range(bh):
        for j in range(np_):
            assert items[h * np_ + j] == [(h, c) for c in range(j, nc, np_)], (h, j)
        assert [owner[h][c] for c in range(nc)] == [h * np_ + c % np_ for c in range(nc)]


@pytest.mark.parametrize("bh", BHS)
@pytest.mark.parametrize("grid", GRIDS, ids=GRID_IDS)
def test_chosen_layout_shares_no_links(grid, bh):
    """The layout the cost model picks (at the production chunk count) keeps each head's hand-off
    traffic on its own NoC links. Checked on the factory's actual core map."""
    nv, np_, pl = _t.chunk_gdn_fused_geometry(grid[0], grid[1], bh, 64, VT)[:3]
    if nv == 0 or pl == 0:
        pytest.skip("no row-local geometry for this (grid, BH)")
    rcv, prod = _t.chunk_gdn_fused_placement(grid[0], grid[1], bh, nv, np_, pl)
    # A pool's home producers are the row-local map; its extras serve every head from wherever they sit.
    nph = _t.chunk_gdn_fused_pool_home_producers(grid[0], grid[1], bh, nv, np_) if pl == 2 else np_
    bad = _shared_links([tuple(c) for c in rcv], [tuple(c) for c in prod[: bh * nph]], nv, nph)
    assert not bad, f"NV={nv} NP={np_} placement={pl}: {bad[:5]}"


@pytest.mark.parametrize(
    "grid, bh, nv, np_, placement, message",
    [
        ((8, 8), 12, 1, 8, 0, r"R\+P = "),  # 12*9 = 108 cores > 64
        ((11, 10), 21, 4, 1, 0, "receiver rectangles do not fit"),  # 105 cores fit; 2 heads/row x 10 rows < 21
        ((11, 10), 16, 4, 2, 1, "leftover heads need"),  # 96 cores fit; 6 leftover heads need 12 rows > 10
        ((11, 10), 11, 4, 4, 1, "not a multiple of the leftover width"),  # 88 cores fit; 3 leftover columns
        ((11, 10), 16, 2, 79, 2, r"R\+P = "),  # pool: 32 + 79 = 111 cores > 110
        ((11, 10), 16, 2, 15, 2, "no row-local layout with a home producer per head"),  # pool smaller than BH
    ],
)
def test_infeasible_placement_raises(expect_error, grid, bh, nv, np_, placement, message):
    """A layout that does not fit must FATAL (as the factory would), never return a truncated map."""
    with expect_error(RuntimeError, message):
        _t.chunk_gdn_fused_placement(grid[0], grid[1], bh, nv, np_, placement)


# ---------------------------------------------------------------------------------------------
# Calibration anchors: the shipped model reproduces the QB2 measurements it was fitted to (2026-10-02, the
# producer-lever stack; Tracy device time at T=2048 -> NC=64, median of ops 2-5)
# ---------------------------------------------------------------------------------------------


def test_phased_model_matches_measurements():
    """T_phased reproduces the measured phased device-time points (prep + scan) within 2 %."""
    for bh, meas in ((4, 359.8), (8, 461.5), (12, 603.0), (16, 769.4), (32, 1278.8), (48, 1907.9)):
        t_ph = _t.chunk_gdn_fused_geometry(11, 10, bh, 64, VT)[5]
        assert abs(t_ph - meas) / meas < 0.02, (bh, t_ph, meas)
    t24 = _t.chunk_gdn_fused_geometry(11, 10, 24, 64, VT)[5]
    assert 1000 <= t24 <= 1050, t24  # interpolated between BH=16 and 32


# Per-head fused device time: (BH, NV, NP, handoff_depth, placement, us, tolerance). Vtl=1 at the balance (BH=12
# NV=4 NP=5) runs ~15 % above the model: its round-trip-bound step is more fragile than the Vtl=2 bump says, and
# the model never picks NV=4.
_FUSED_MEASUREMENTS = [
    (4, 2, 9, 2, 1, 203.5, 0.05),
    (4, 2, 9, 3, 1, 204.1, 0.05),
    (4, 2, 7, 2, 1, 203.5, 0.05),
    (4, 4, 7, 2, 1, 237.9, 0.05),
    (4, 4, 7, 3, 1, 238.3, 0.05),
    (8, 2, 9, 2, 1, 218.6, 0.05),
    (8, 2, 9, 3, 1, 219.0, 0.05),
    (8, 2, 7, 2, 1, 213.4, 0.05),
    (8, 2, 6, 2, 1, 212.4, 0.05),
    (8, 2, 5, 2, 1, 237.4, 0.05),
    (8, 4, 7, 2, 1, 255.3, 0.05),
    (8, 4, 7, 3, 1, 256.0, 0.05),
    (12, 2, 7, 2, 1, 228.8, 0.05),
    (12, 2, 7, 3, 1, 229.8, 0.05),
    (12, 2, 6, 2, 1, 237.1, 0.05),
    (12, 2, 6, 3, 1, 225.3, 0.05),
    (12, 2, 5, 2, 1, 246.9, 0.05),
    (12, 2, 5, 3, 1, 247.0, 0.05),
    (12, 4, 5, 2, 1, 306.3, 0.16),
    (12, 4, 5, 3, 1, 306.0, 0.16),
    (16, 1, 4, 2, 1, 348.1, 0.05),
    (16, 1, 4, 3, 1, 347.9, 0.05),
    (16, 1, 3, 2, 1, 377.5, 0.05),
    (16, 2, 3, 2, 1, 372.5, 0.05),
    (16, 2, 3, 3, 1, 372.6, 0.05),
    (16, 2, 4, 2, 0, 575.6, 0.05),
    (16, 1, 5, 2, 0, 591.5, 0.05),
    (16, 4, 2, 2, 0, 543.5, 0.05),
    (24, 1, 3, 2, 1, 393.1, 0.05),
    (24, 1, 3, 3, 1, 393.7, 0.05),
    (24, 2, 2, 2, 1, 544.2, 0.05),
    (32, 1, 2, 2, 1, 556.1, 0.05),
    (32, 2, 1, 2, 1, 1046.1, 0.05),
    (48, 1, 1, 2, 1, 1059.5, 0.05),
]


@pytest.mark.parametrize("bh, nv, np_, depth, placement, meas, tol", _FUSED_MEASUREMENTS, ids=lambda v: str(v))
def test_fused_model_matches_measurements(bh, nv, np_, depth, placement, meas, tol):
    """T_fused at a pinned (NV, NP, depth) reproduces the measured fused device time; the row-major rows
    (no row-local layout at BH=16 for them) go through the fallback and its link-bound pace."""
    got_nv, got_np, pl, nbuf, t_f = _t.chunk_gdn_fused_geometry(11, 10, bh, 64, VT, nv, np_, depth)[:5]
    assert (got_nv, got_np, pl, nbuf) == (nv, np_, placement, depth)
    assert abs(t_f - meas) / meas < tol, f"model {t_f:.1f} vs measured {meas} ({(t_f - meas) / meas:+.1%})"


def test_qb2_operating_points():
    """The Qwen3.6-27B TP-4 shape (BH=12) on QB2 picks NV=2 per-head at hand-off depth 3, 6 producers per head
    (measured 225 us; 7 per head 229, the pool 231), ~227 us vs phased 603; BH=4 and 8 the same NV=2 NP=7
    geometry; BH=48 pays with the pool of 62 (one home producer per head + 14 extras, ~838 vs the phased 1908);
    BH=64 needs >= 128 cores, so no fused geometry exists and the op must dispatch phased."""
    nv, np_, pl, nbuf, t_f, t_ph, pays, _ = _t.chunk_gdn_fused_geometry(11, 10, 12, 64, VT)
    assert (nv, np_, pl, nbuf) == (2, 6, 1, 3) and pays
    assert 222 <= t_f <= 232 and 595 <= t_ph <= 610
    for bh in (4, 8):
        nv, np_, _, nbuf = _t.chunk_gdn_fused_geometry(11, 10, bh, 64, VT)[:4]
        assert (nv, np_, nbuf) == (2, 7, 3), (bh, nv, np_, nbuf)
    nv, np_, pl, nbuf, t_f, t_ph, pays48, num = _t.chunk_gdn_fused_geometry(11, 10, 48, 64, VT)
    assert (nv, np_, pl, nbuf, num) == (1, 62, 2, 2, 14) and pays48 and t_f < 0.5 * t_ph, (nv, np_, pl, nbuf, t_f, t_ph)
    nv, _, _, _, _, _, pays, _ = _t.chunk_gdn_fused_geometry(11, 10, 64, 64, VT)
    assert nv == 0 and not pays


# Pooled fused device time: (BH, NV, P, handoff_depth, extras' share, us, tolerance). The share sweeps at the pool
# picks (BH=16: NV=2 P=78, balanced 30/78; 24: NV=1 P=86, 14/86; 32: NV=1 P=78, 14/78; 48: NV=1 P=62, 14/62)
# and the pools of BH 4-12 at their balanced share. Around the BH=16 optimum (shares 0.30-0.34, 292-294 us) the
# model steps with the busiest home producer's item count; the staggered first items smooth that in reality,
# so the share-0.30 row (16 home items in the map, 15 in effect) sits 8 % under the model.
_POOL_MEASUREMENTS = [
    (16, 2, 78, 2, 30 / 78, 318.0, 0.06),
    (16, 2, 78, 3, 30 / 78, 320.8, 0.06),
    (16, 2, 78, 2, 0.36, 301.5, 0.06),
    (16, 2, 78, 2, 0.34, 292.6, 0.06),
    (16, 2, 78, 2, 0.32, 292.5, 0.06),
    (16, 2, 78, 2, 0.30, 293.5, 0.10),
    (16, 2, 78, 2, 0.26, 310.5, 0.06),
    (16, 2, 78, 2, 0.22, 321.3, 0.06),
    (16, 2, 78, 2, 0.179, 330.5, 0.06),
    (16, 2, 78, 2, 0.14, 341.8, 0.06),
    (16, 2, 78, 2, 0.10, 347.3, 0.06),
    (16, 2, 78, 2, 0.06, 358.2, 0.06),
    (16, 2, 78, 2, 0.03, 364.8, 0.06),
    (16, 2, 78, 2, 0.0, 383.8, 0.06),
    (16, 2, 78, 3, 0.34, 294.2, 0.06),
    (16, 2, 78, 3, 0.30, 292.7, 0.06),
    (16, 2, 78, 3, 0.26, 308.6, 0.06),
    (16, 1, 94, 2, 30 / 94, 365.1, 0.06),
    (12, 2, 86, 2, 2 / 86, 250.3, 0.06),
    (12, 2, 86, 3, 2 / 86, 230.5, 0.06),
    (12, 2, 86, 2, 0.0, 249.5, 0.06),
    (12, 2, 86, 3, 0.0, 230.9, 0.06),
    (8, 2, 94, 2, 22 / 94, 237.9, 0.06),
    (8, 2, 94, 3, 22 / 94, 230.5, 0.08),
    (4, 2, 102, 2, 66 / 102, 209.1, 0.06),
    (4, 2, 102, 3, 66 / 102, 205.9, 0.06),
    (4, 4, 94, 2, 66 / 94, 303.1, 0.06),
    (24, 1, 86, 2, 14 / 86, 368.5, 0.06),
    (24, 1, 86, 3, 14 / 86, 367.8, 0.06),
    (24, 1, 86, 2, 0.10, 372.1, 0.06),
    (24, 1, 86, 2, 0.05, 389.8, 0.06),
    (24, 1, 86, 2, 0.0, 397.6, 0.06),
    (32, 1, 78, 2, 14 / 78, 474.3, 0.06),
    (32, 1, 78, 3, 14 / 78, 472.9, 0.06),
    (32, 1, 78, 2, 0.10, 514.0, 0.06),
    (32, 1, 78, 2, 0.05, 540.1, 0.06),
    (32, 1, 78, 2, 0.0, 559.9, 0.06),
    (48, 1, 62, 2, 14 / 62, 836.5, 0.06),
    (48, 1, 62, 3, 14 / 62, 836.4, 0.06),
    (48, 1, 62, 2, 0.10, 969.4, 0.06),
    (48, 1, 62, 2, 0.05, 1018.2, 0.06),
    (48, 1, 62, 2, 0.0, 1063.2, 0.06),
]


@pytest.mark.parametrize("bh, nv, P, depth, share, meas, tol", _POOL_MEASUREMENTS, ids=lambda v: str(round(v, 3)))
def test_pool_model_matches_measurements(bh, nv, P, depth, share, meas, tol):
    """T_fused of the pool at a pinned (NV, P, depth, share) reproduces the measured pooled device time."""
    got_nv, got_np, pl, nbuf, t_f, _, _, num = _t.chunk_gdn_fused_geometry(11, 10, bh, 64, VT, nv, P, depth, POOL, share)
    assert (got_nv, got_np, pl, nbuf, num) == (nv, P, 2, depth, int(share * P + 0.5))
    assert abs(t_f - meas) / meas < tol, f"model {t_f:.1f} vs measured {meas} ({(t_f - meas) / meas:+.1%})"


def test_qb2_pool_operating_points():
    """The producer pool on QB2 (what producer_pool=True resolves to with the geometry left free): every core the
    receivers leave, in the row-local map of the largest NPH the pool allows plus the extras, at the model's
    share. BH=16 NV=2: 78 producers = 3 per head + 30 extras at 27/78 (measured 294 us; the balanced 30/78 318);
    BH=24 NV=1: 86 = 3 per head + 14 extras, balanced; BH=32: 78 = 2 per head + 14, balanced; BH=48: 62 = 1 per
    head + 14, balanced. The pool is the default pick from BH=16 up (16: 293 vs the per-head 347; 24: 374 vs 388;
    32: 473 vs 547; 48: 838 vs 1056) and loses to the per-head geometry at BH <= 12."""
    for bh, nv, P, nph, num, nbuf, wins in (
        (16, 2, 78, 3, 27, 2, True),
        (12, 2, 86, 7, 2, 3, False),
        (8, 2, 94, 9, 22, 3, False),
        (4, 2, 102, 9, 66, 3, False),
        (24, 1, 86, 3, 14, 2, True),
        (32, 1, 78, 2, 14, 2, True),
        (48, 1, 62, 1, 14, 2, True),
    ):
        got_nv, got_np, pl, got_nbuf, t_pool, _, pays, got_num = _t.chunk_gdn_fused_geometry(
            11, 10, bh, 64, VT, nv, candidates=POOL
        )
        assert (got_nv, got_np, pl, got_nbuf, got_num) == (nv, P, 2, nbuf, num), (bh, got_nv, got_np, pl, got_nbuf, got_num)
        assert _t.chunk_gdn_fused_pool_home_producers(11, 10, bh, nv, P) == nph, (bh, nv, P)
        assert pays
        t_head = _t.chunk_gdn_fused_geometry(11, 10, bh, 64, VT, candidates=PER_HEAD)[4]
        assert (t_pool < t_head) == wins, (bh, t_pool, t_head)
    nv, P, pl, nbuf, t_pool, _, _, num = _t.chunk_gdn_fused_geometry(11, 10, 16, 64, VT, candidates=POOL)
    assert (nv, P, pl, nbuf, num) == (2, 78, 2, 2, 27) and 283 <= t_pool <= 298, (nv, P, pl, nbuf, num, t_pool)
    t_head = _t.chunk_gdn_fused_geometry(11, 10, 16, 64, VT, candidates=PER_HEAD)[4]
    assert 340 <= t_head <= 352, t_head
    for bh, expect_pool in (
        (1, False),
        (2, False),
        (3, False),
        (4, False),
        (6, False),
        (8, False),
        (12, False),
        (16, True),
        (24, True),
        (32, True),
        (48, True),
    ):
        nv, np_, pl = _t.chunk_gdn_fused_geometry(11, 10, bh, 64, VT)[:3]
        assert (pl == 2) == expect_pool, f"BH={bh}: default pick NV={nv} NP={np_} placement={pl}"
    # A pool without extras is the per-head geometry: under both candidate kinds it is reported as such.
    nv, np_, pl, _, t_pin, _, _, num = _t.chunk_gdn_fused_geometry(11, 10, 16, 64, VT, 2, 48)
    assert (nv, np_, pl, num) == (2, 3, 1, 0), (nv, np_, pl, num)
    nv, np_, pl, _, t_pool, _, _, num = _t.chunk_gdn_fused_geometry(11, 10, 16, 64, VT, 2, 48, 0, POOL)
    assert (nv, np_, pl, num, t_pool) == (2, 48, 2, 0, t_pin), (nv, np_, pl, num, t_pool, t_pin)


def test_pool_share_choice():
    """The model's share for the BH=16 pool follows the measured sweep: a little under the balance (27/78 = 0.346,
    the home producers carry 15 % more than the extras) is the optimum; the balance itself costs the ordering
    penalty, lower shares make the home producers the bottleneck. A pinned share is taken as is, and the
    oracle agrees with the C++ at every share."""
    t_at = {}
    for num in range(31):
        got = _t.chunk_gdn_fused_geometry(11, 10, 16, 64, VT, 2, 78, 2, POOL, num / 78)
        o = _choose_geometry((11, 10), 16, 64, fixed_nv=2, fixed_np=78, fixed_nbuf=2, candidates=POOL, fixed_share=num / 78)
        assert got[7] == num == o["num"] and abs(got[4] - o["T_fused"]) < 0.5, (num, got, o)
        t_at[num] = got[4]
    assert min(t_at, key=t_at.get) == 27, t_at
    assert t_at[30] > t_at[27] * 1.08 and t_at[0] > t_at[27] * 1.25, t_at
    free = _t.chunk_gdn_fused_geometry(11, 10, 16, 64, VT, 2, 78, 2, POOL)
    assert free[7] == 27 and free[4] == t_at[27]
