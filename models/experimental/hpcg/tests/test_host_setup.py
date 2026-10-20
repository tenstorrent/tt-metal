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
    lib = native.lib()
    if lib is None:
        pytest.skip("no C++ compiler")
    a, b = Level.__new__(Level), Level.__new__(Level)
    for L in (a, b):
        L.nx, L.bands, L.n = nx, bands, nx**3
    a._build_native(lib, slot_stride)
    b._build_numpy(slot_stride)
    return a, b


@pytest.mark.parametrize("nx, bands", CASES)
def test_level_matches_numpy(nx, bands):
    a, b = _levels(nx, bands)
    assert (a.P, a.starts, a.bstarts, a.nnz) == (b.P, b.starts, b.bstarts, b.nnz)
    for f in ("pos", "nat_at", "V_rows", "COL_rows", "B"):
        assert np.array_equal(getattr(a, f), getattr(b, f)), f


@pytest.mark.parametrize("nx, bands", CASES)
@pytest.mark.parametrize("exchange", [True, False])
def test_plan_matches_numpy(nx, bands, exchange):
    L, _ = _levels(nx, bands)
    blocks = [(L.starts[c], L.starts[c + 1]) for c in range(8)]
    noc = [(i % 13, i // 13) for i in range(130)]
    p = SparseRowPlan(L.COL_rows, blocks, blocks, 130, noc, exchange, bands=L.bstarts if bands > 1 else None, pout=L.P)
    ng = L.P // 16 + 2
    got = SparseRowPlan._plan_native(native.lib(), L.COL_rows, p.own, exchange, ng, 0)
    want = SparseRowPlan._plan_numpy(L.COL_rows, L.P, p.own, exchange, ng, 0)
    assert len(got[0]) == len(want[0])
    for (gl, gn, g1), (wl, wn, w1) in zip(got[0], want[0]):
        assert np.array_equal(gl, wl) and np.array_equal(gn, wn) and np.array_equal(g1, w1)
    assert np.array_equal(got[1], want[1]), "offsets"
    assert np.array_equal(got[2], want[2]), "own L1 offsets"
    assert got[3] == want[3], "exchange entries"


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
