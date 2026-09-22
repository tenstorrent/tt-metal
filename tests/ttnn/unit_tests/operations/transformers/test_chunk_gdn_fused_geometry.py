# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-side geometry gates for the fused prep->scan GDN prim (ttnn::prim::chunk_gdn_fused).

No device needed. The fused op's geometry is three pure C++ functions, bound through nanobind:

* ``chunk_gdn_fused_geometry``           — the calibrated cost model (NV, NP, placement) the op picks
                                           when no QWEN_GDN_NV / QWEN_GDN_NP / QWEN_GDN_PLACEMENT knob is set;
* ``chunk_gdn_fused_row_local_feasible`` — the row-local feasibility predicate;
* ``chunk_gdn_fused_placement``          — the core map, computed by the SAME function the program
                                           factory calls.

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
# it is an independent re-derivation of chunk_gdn_fused.cpp (choose_fused_geometry,
# fused_row_local_feasible, fused_placement), so a bug has to be made twice to pass.
# Cost-model constants: QB2, measured 2026-09-21 (wall-op and per-RISC device zones).
# ---------------------------------------------------------------------------------------------
_W_P_US = 34.0  # producer us per (head, chunk) item at >= 84 concurrent producers
_W_P_LOW_US, _W_P_LOW_PRODUCERS, _W_P_HIGH_PRODUCERS = 26.0, 28, 84  # lighter load -> faster, linear
_T_STEP_US = {1: 3.5, 2: 4.9, 4: 7.5}  # receiver period at V-slice width Vtl (tiles)
_FILL_US = 65.0
_PHASED_A_MS, _PHASED_B_MS, _PHASED_BH48_MS, _PHASED_LINEAR_TO_BH = 0.0358, 0.310, 2.475, 32


def _w_p_at(producers):
    f = min(1.0, max(0.0, (producers - _W_P_LOW_PRODUCERS) / float(_W_P_HIGH_PRODUCERS - _W_P_LOW_PRODUCERS)))
    return _W_P_LOW_US + (_W_P_US - _W_P_LOW_US) * f


def _t_phased_us(bh, nc):
    lin = _PHASED_LINEAR_TO_BH
    t_lin = (_PHASED_A_MS * lin + _PHASED_B_MS) * 1000.0
    if bh <= lin:
        t = (_PHASED_A_MS * bh + _PHASED_B_MS) * 1000.0
    else:  # DRAM-saturated: interpolate to the measured BH=48 point, then scale linearly
        t = t_lin + (_PHASED_BH48_MS * 1000.0 - t_lin) * (min(bh, 48) - lin) / float(48 - lin)
        if bh > 48:
            t *= bh / 48.0
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


def _choose_geometry(grid, bh, nc, fixed_nv=0, fixed_np=0):
    """Over NV | Vt and NP with a feasible row-local layout, minimise
    T_fused = NC * max(w_p / NP, t_step(Vt / NV)) + fill; ties -> fewer cores, then smaller NV. With no
    row-local layout, fall back to row-major placement with the same formula."""
    gx, gy = grid
    best = None
    for nv in (1, 2, 4, 8):
        if VT % nv or (fixed_nv and nv != fixed_nv) or (VT // nv) not in _T_STEP_US:
            continue
        for np_ in range(1, gx - nv + 1):
            np_eff = min(np_, nc)
            if bh * (nv + np_eff) > gx * gy:
                break
            if fixed_np and np_eff != min(fixed_np, nc):
                continue
            if not _row_local_feasible(gx, gy, bh, nv, np_eff):
                continue
            t = nc * max(_w_p_at(bh * np_eff) / np_eff, _T_STEP_US[VT // nv]) + _FILL_US
            key = (t, nv + np_eff, nv)
            if best is None or key < best[0]:
                best = (key, nv, np_eff, 1)
    if best is None:
        for nv in (1, 2, 4, 8):
            if VT % nv or (VT // nv) not in _T_STEP_US or (fixed_nv and nv != fixed_nv):
                continue
            if not _row_major_feasible(gx, gy, bh, nv) or gx * gy - bh * nv < 1:
                continue
            np_ = min(fixed_np, nc) if fixed_np else min((gx * gy - bh * nv) // bh, nc)
            if np_ < 1 or bh * (nv + np_) > gx * gy:
                continue
            t = nc * max(_w_p_at(bh * np_) / np_, _T_STEP_US[VT // nv]) + _FILL_US
            key = (t, nv + np_, nv)
            if best is None or key < best[0]:
                best = (key, nv, np_, 0)
    t_ph = _t_phased_us(bh, nc)
    if best is None:
        return {"nv": None, "np": None, "placement": None, "T_fused": None, "T_phased": t_ph, "fused_pays": False}
    (t, _, _), nv, np_, pl = best
    return {"nv": nv, "np": np_, "placement": pl, "T_fused": t, "T_phased": t_ph, "fused_pays": t < t_ph}


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


@pytest.mark.parametrize("nc", [8, 64])
@pytest.mark.parametrize("bh", BHS)
@pytest.mark.parametrize("grid", GRIDS, ids=GRID_IDS)
def test_cost_model_mirror(grid, bh, nc):
    """The C++ cost model and the oracle pick the same (NV, NP, placement) and agree on T_fused and
    T_phased — so the op's default dispatch is the documented model on every grid."""
    nv, np_, pl, t_f, t_ph, pays = _t.chunk_gdn_fused_geometry(grid[0], grid[1], bh, nc, VT)
    o = _choose_geometry(grid, bh, nc)
    if o["nv"] is None:
        assert nv == 0, f"oracle: no fused geometry fits; C++ picked NV={nv} NP={np_}"
        assert not pays
        return
    assert (nv, np_, pl) == (o["nv"], o["np"], o["placement"]), f"C++ {(nv, np_, pl)} vs oracle {o}"
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


@pytest.mark.parametrize("fixed", [(1, 0), (2, 0), (4, 0), (0, 1), (0, 2), (0, 5), (0, 8), (2, 2), (4, 5)])
@pytest.mark.parametrize("bh", [4, 12, 16, 32])
@pytest.mark.parametrize("grid", [(11, 10), (12, 10)], ids=["11x10", "12x10"])
def test_constrained_choice_mirror(grid, bh, fixed):
    """A partial env override (QWEN_GDN_NV or QWEN_GDN_NP alone) pins that field; C++ and the oracle
    fill in the other one identically, and the result always fits the grid."""
    fnv, fnp = fixed
    nv, np_, pl, _, _, _ = _t.chunk_gdn_fused_geometry(grid[0], grid[1], bh, 16, VT, fnv, fnp)
    o = _choose_geometry(grid, bh, 16, fixed_nv=fnv, fixed_np=fnp)
    if o["nv"] is None:
        assert nv == 0
        return
    assert (nv, np_, pl) == (o["nv"], o["np"], o["placement"]), ((nv, np_, pl), o)
    assert bh * (nv + np_) <= grid[0] * grid[1]
    if fnv:
        assert nv == fnv
    if fnp:
        assert np_ == min(fnp, 16)


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
def test_chosen_layout_shares_no_links(grid, bh):
    """The layout the cost model picks (at the production chunk count) keeps each head's hand-off
    traffic on its own NoC links. Checked on the factory's actual core map."""
    nv, np_, pl, _, _, _ = _t.chunk_gdn_fused_geometry(grid[0], grid[1], bh, 64, VT)
    if nv == 0 or pl != 1:
        pytest.skip("no row-local geometry for this (grid, BH)")
    rcv, prod = _t.chunk_gdn_fused_placement(grid[0], grid[1], bh, nv, np_, pl)
    bad = _shared_links([tuple(c) for c in rcv], [tuple(c) for c in prod], nv, np_)
    assert not bad, f"NV={nv} NP={np_}: {bad[:5]}"


@pytest.mark.parametrize(
    "grid, bh, nv, np_, placement, message",
    [
        ((8, 8), 12, 1, 8, 0, r"R\+P = "),  # 12*9 = 108 cores > 64
        ((11, 10), 21, 4, 1, 0, "receiver rectangles do not fit"),  # 105 cores fit; 2 heads/row x 10 rows < 21
        ((11, 10), 16, 4, 2, 1, "leftover heads need"),  # 96 cores fit; 6 leftover heads need 12 rows > 10
        ((11, 10), 11, 4, 4, 1, "not a multiple of the leftover width"),  # 88 cores fit; 3 leftover columns
    ],
)
def test_infeasible_placement_raises(expect_error, grid, bh, nv, np_, placement, message):
    """A layout that does not fit must FATAL (as the factory would), never return a truncated map."""
    with expect_error(RuntimeError, message):
        _t.chunk_gdn_fused_placement(grid[0], grid[1], bh, nv, np_, placement)


# ---------------------------------------------------------------------------------------------
# Calibration anchors: the shipped model reproduces the QB2 measurements it was fitted to
# ---------------------------------------------------------------------------------------------


def test_phased_model_matches_measurements():
    """T_phased reproduces the measured phased wall-op points (QB2, NC=64) within 5 %."""
    for bh, meas in ((4, 453), (8, 593), (12, 706), (16, 883), (32, 1449), (48, 2475)):
        t_ph = _t.chunk_gdn_fused_geometry(11, 10, bh, 64, VT)[4]
        assert abs(t_ph - meas) / meas < 0.05, (bh, t_ph, meas)


def test_fused_model_matches_measurements():
    """T_fused at the model's own picks reproduces the measured fused wall-op points (QB2, NC=64, wall
    minus the 115 us of host glue) within 10 %."""
    for bh, meas in ((4, 284), (8, 326), (12, 364), (16, 581), (32, 1178)):
        nv, np_, _, t_f, _, _ = _t.chunk_gdn_fused_geometry(11, 10, bh, 64, VT)
        assert abs(t_f - meas) / meas < 0.10, (bh, nv, np_, t_f, meas)


def test_qb2_operating_points():
    """The Qwen3.6-27B TP-4 shape (BH=12) on QB2 picks NV=2, NP=7 row-local, ~390 us vs phased ~706;
    BH=64 needs >= 128 cores, so no fused geometry exists and the op must dispatch phased."""
    nv, np_, pl, t_f, t_ph, pays = _t.chunk_gdn_fused_geometry(11, 10, 12, 64, VT)
    assert (nv, np_, pl) == (2, 7, 1) and pays
    assert 360 <= t_f <= 400 and 690 <= t_ph <= 745
    nv, _, _, _, _, pays = _t.chunk_gdn_fused_geometry(11, 10, 64, 64, VT)
    assert nv == 0 and not pays
