# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The compiled host setup (host/native_setup.cpp) builds exactly what the numpy definitions build: the matrix of a
level (reference/problem.py) and the per-core plan of a launch (sparse_rows/plan.py). Host only, no device."""

import numpy as np
import pytest

from models.experimental.hpcg.host import native
from models.experimental.hpcg.reference.problem import Level
from models.experimental.hpcg.tt.custom_ops.sparse_rows.plan import SparseRowPlan
from models.experimental.hpcg.tt.hpcg import slot_stride

CASES = [(13, 1), (16, 1), (26, 1), (48, 5), (64, 10), (60, 5)]


def _levels(nx, bands):
    """The same level built by the compiled helper and by numpy."""
    if native.lib() is None:
        pytest.skip("no C++ compiler")
    a, b = Level(nx, bands, slot_stride), Level.__new__(Level)
    b.nx, b.bands, b.n = nx, bands, nx**3
    b.P, b.starts, b.bstarts = b._layout()
    b.stride = slot_stride(b.P)
    b._build_numpy()
    return a, b


@pytest.mark.parametrize("nx, bands", CASES)
def test_level_matches_numpy(nx, bands):
    a, b = _levels(nx, bands)
    assert (a.P, a.starts, a.bstarts, a.nnz) == (b.P, b.starts, b.bstarts, b.nnz)
    for f in ("pos", "nat_at", "V_rows", "COL_rows", "B"):
        assert np.array_equal(getattr(a, f), getattr(b, f)), f
    assert np.array_equal(a.real, a.nat_at >= 0) and a.nnz == (3 * nx - 2) ** 3


@pytest.mark.parametrize("nx, bands", CASES)
@pytest.mark.parametrize("exchange", [True, False])
def test_plan_matches_numpy(nx, bands, exchange):
    L, _ = _levels(nx, bands)
    blocks = [(L.starts[c], L.starts[c + 1]) for c in range(8)]
    noc = [(i % 13, i // 13) for i in range(130)]
    p = SparseRowPlan(L.COL_rows, blocks, blocks, 130, noc, exchange, bands=L.bstarts if bands > 1 else None, pout=L.P)
    ng = L.P // 16 + 2
    got = SparseRowPlan._plan_native(native.lib(), L.COL_rows, p.own, exchange, ng, 0, noc)
    want = SparseRowPlan._plan_numpy(L.COL_rows, L.P, p.own, exchange, ng, 0)

    def per_core(f):  # ([nc, maxp] lo, len, l1, pieces per core) -> per-core (lo, len, l1)
        lo, ln, l1, n = f
        return [(lo[k, : n[k]], ln[k, : n[k]], l1[k, : n[k]]) for k in range(len(n))]

    assert np.array_equal(got[0][3], want[0][3])
    for (gl, gn, g1), (wl, wn, w1) in zip(per_core(got[0]), per_core(want[0])):
        assert np.array_equal(gl, wl) and np.array_equal(gn, wn) and np.array_equal(g1, w1)
    assert np.array_equal(got[1], want[1]), "offsets"
    assert np.array_equal(got[2], want[2]), "own L1 offsets"
    w_rows, w_at, w_nbr = SparseRowPlan._group_exchange(want[3], p.nc, 8, noc)
    assert np.array_equal(got[3][0], w_rows) and np.array_equal(got[3][1], w_at), "exchange"
    assert all(np.array_equal(g, w) for g, w in zip(got[3][2], w_nbr)), "neighbours"


@pytest.mark.parametrize("nx, bands", CASES)
def test_link_matches_numpy(nx, bands):
    if native.lib() is None:
        pytest.skip("no C++ compiler")
    F, C = Level(nx, bands, slot_stride), Level(nx // 2, 1, slot_stride)
    F._link_native(native.lib(), C)
    got = (F.R, F.Rw, F.Pm, F.Pw)
    F._link_numpy(C)
    for name, g, w in zip(("R", "Rw", "Pm", "Pw"), got, (F.R, F.Rw, F.Pm, F.Pw)):
        assert g.dtype == w.dtype and np.array_equal(g, w), name


@pytest.mark.parametrize("nx, bands", CASES)
def test_stencil_windows_match_column_windows(nx, bands):
    """The window pieces planned from the storage-order formula equal the ones planned from the column table."""
    L, _ = _levels(nx, bands)
    blocks = [(L.starts[c], L.starts[c + 1]) for c in range(8)]
    noc = [(i % 13, i // 13) for i in range(130)]
    bs = L.bstarts if bands > 1 else None
    want = SparseRowPlan(L.COL_rows, blocks, blocks, 130, noc, True, bands=bs, pout=L.P)
    got = SparseRowPlan(None, blocks, blocks, 130, noc, True, bands=bs, pout=L.P, stencil=L)
    assert got.nc == want.nc and np.array_equal(got.own, want.own)
    for (gl, gn, g1), (wl, wn, w1) in zip(got.pieces, want.pieces):
        assert np.array_equal(gl, wl) and np.array_equal(gn, wn) and np.array_equal(g1, w1)
    assert np.array_equal(got.own_l1, want.own_l1) and got.exch == want.exch and got.win_b == want.win_b
    assert got.off is None


@pytest.mark.parametrize("nx, bands", [(26, 1), (64, 10), (60, 5)])
def test_meta_matches_python(nx, bands):
    """The compiled per-core metadata rows equal the Python ones, for Gauss-Seidel, residual and SpMV phase lists."""
    L, _ = _levels(nx, bands)
    blocks = [(L.starts[c], L.starts[c + 1]) for c in range(8)]
    noc = [(1 + i % 13, 2 + i // 13) for i in range(130)]
    p = SparseRowPlan(L.COL_rows, blocks, blocks, 130, noc, True, bands=L.bstarts if bands > 1 else None, pout=L.P)
    gs = list(range(7, -1, -1)) + list(range(8))
    for phases in (gs, gs + [8 + b for b in range(8)], [8 + b for b in range(8)], [0]):
        for tpw, align in ((1024, 16), (256, 16), (1024, 1024)):
            got, want = p.meta(phases, noc, tpw, align), p._meta_python(phases, noc, tpw, align)
            assert np.array_equal(got[0], want[0]) and list(got[1:]) == list(want[1:]), (phases[:3], tpw, align)


@pytest.mark.parametrize("nx, bands", [(16, 1), (26, 1), (48, 5), (64, 10), (60, 5)])
def test_transfer_plans_match_link(nx, bands):
    """Restriction and prolongation planned from the storage-order formulas equal the plans built from Level.link's
    R / Pm tables: window pieces, offsets and weights."""
    if native.lib() is None:
        pytest.skip("no C++ compiler")
    F, C = Level(nx, bands, slot_stride), Level(nx // 2, max(1, bands // 2), slot_stride)
    F.link(C)
    noc = [(i % 13, i // 13) for i in range(130)]
    nb0 = F.starts[1]
    cblocks = [(C.starts[c], C.starts[c + 1]) for c in range(8)]
    cases = [
        ("restrict", F.R, F.Rw, cblocks, [(0, nb0)], C.bstarts if C.bands > 1 else None, C.P),
        ("prolong", F.Pm, F.Pw, [(0, nb0)], cblocks, [F.bstarts[0]] if F.bands > 1 else None, nb0),
    ]
    for kind, table, weights, out_blocks, src, bs, pout in cases:
        want = SparseRowPlan(table[None, :], out_blocks, src, 130, noc, False, bands=bs)
        got = SparseRowPlan(None, out_blocks, src, 130, noc, False, bands=bs, pout=pout, transfer=(kind, F, C))
        assert got.nc == want.nc and np.array_equal(got.own, want.own), kind
        for (gl, gn, g1), (wl, wn, w1) in zip(got.pieces, want.pieces):
            assert np.array_equal(gl, wl) and np.array_equal(gn, wn) and np.array_equal(g1, w1), kind
        assert np.array_equal(got.off, want.off), f"{kind} offsets"
        assert np.array_equal(got.weights.reshape(-1), weights), f"{kind} weights"


@pytest.mark.parametrize("nc", [1, 7, 64, 96, 130])
def test_ownership_matches_numpy(nc):
    """Row ownership from the compiled helper equals the numpy definition, with and without y-bands."""
    rng = np.random.default_rng(nc)
    for _ in range(20):
        starts = np.cumsum(rng.integers(0, 50000, 9))
        blocks = [(int(starts[b]), int(starts[b + 1])) for b in range(8)]
        for B in (1, 2, 5, 10):
            if nc % B or nc < B:
                continue
            bands = None
            if B > 1:
                bands = [[s] + sorted(int(x) for x in rng.integers(s, e + 1, B - 1)) + [e] for s, e in blocks]
            got = SparseRowPlan._ownership(native.lib(), blocks, bands, nc)
            want = SparseRowPlan._ownership_numpy(blocks, bands, nc)
            assert np.array_equal(got, want), (nc, B)


@pytest.mark.parametrize("nx, bands", CASES)
def test_deferred_exchange_matches(nx, bands):
    """A stencil plan whose exchange lists are completed later equals one planned in one go."""
    L, _ = _levels(nx, bands)
    blocks = [(L.starts[c], L.starts[c + 1]) for c in range(8)]
    noc = [(i % 13, i // 13) for i in range(130)]
    bs = L.bstarts if bands > 1 else None
    want = SparseRowPlan(None, blocks, blocks, 130, noc, True, bands=bs, pout=L.P, stencil=L)
    got = SparseRowPlan(None, blocks, blocks, want.nc, noc, True, bands=bs, pout=L.P, stencil=L, defer_exchange=True)
    assert got.complete_exchange()
    assert got.exch == want.exch and got.nbr == want.nbr and np.array_equal(got.own_l1, want.own_l1)


@pytest.mark.parametrize("nx, bands", CASES)
def test_segment_counts_match_meta(nx, bands):
    """The per-core segment counts of a launch, computed from ownership, equal the ones its metadata rows give."""
    L, _ = _levels(nx, bands)
    blocks = [(L.starts[c], L.starts[c + 1]) for c in range(8)]
    noc = [(i % 13, i // 13) for i in range(130)]
    p = SparseRowPlan(L.COL_rows, blocks, blocks, 130, noc, True, bands=L.bstarts if bands > 1 else None, pout=L.P)
    resid = [8 + b for b in range(8)]
    for phases in ([7, 6, 5, 4, 3, 2, 1, 0], list(range(8)) + resid, resid):
        for tpw in (256, 1024):
            assert list(p.meta(phases, noc, tpw)[1:4]) == list(p.segment_counts(phases, tpw)), (phases, tpw)
    # a post-smoother sharing the pre-smoother's table (SparseRowOp's table_phases): its counts are the table's
    # Gauss-Seidel counts, with no RESTF segments
    gs16 = [7, 6, 5, 4, 3, 2, 1, 0] + list(range(8))
    for tpw in (256, 1024):
        _, _, table_gs, _, _ = p.meta(gs16 + resid, noc, tpw)
        assert list(p.segment_counts(gs16, tpw)) == [table_gs, table_gs, [0] * p.nc], tpw
