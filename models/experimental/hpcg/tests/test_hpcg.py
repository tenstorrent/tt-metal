# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The device-generated matrix against the host-built one, every device op against the numpy reference of the same
algorithm (reference/problem.py), then a few CG iterations against the fp64 reference.

SpMV, restriction and prolongation must be bit-exact in fp32. A Gauss-Seidel update differs from numpy by at most
a couple of fp32 roundings in the divide, so smoothers are checked to 5e-7 absolute (values are O(1)).
"""

import numpy as np
import pytest
import ttnn
from loguru import logger

from models.experimental.hpcg.reference import problem as ref
from models.experimental.hpcg.tt.custom_ops.sparse_rows import SparseRowPlan
from models.experimental.hpcg.tt.custom_ops.sparse_rows.op import noc_coords
from models.experimental.hpcg.tt.hpcg import HPCG, slot_stride
from models.experimental.hpcg.tt.layout import from_device, to_device

GS_ATOL = 5e-7


@pytest.mark.parametrize(
    "nx, bands",
    [
        (64, None),  # default ownership (64^3: no y-bands)
        (64, (5, 5, 5, 5)),  # y-bands on every level, including empty bands on the 8^3 level
        (160, None),  # default bands 10 / 5 / 1 / 1 (slow: numpy reference at 4.1 M rows)
    ],
)
def test_hpcg_ops_and_cg(device, nx, bands):
    h = HPCG(device, nx, bands=bands)
    order = h.order
    # the host-built levels: the reference matrix (the device generated its own)
    levels = ref.build(nx, [L.bands for L in h.levels], slot_stride)
    g = device.compute_with_storage_grid_size()
    for l, (d, L) in enumerate(zip(h.lv, levels)):
        blocks = [(L.starts[c], L.starts[c + 1]) for c in range(8)]
        bs = L.bstarts if L.bands > 1 else None
        hp = SparseRowPlan(L.COL_rows, blocks, blocks, g.x * g.y, noc_coords(device), True, bands=bs, pout=L.P)
        vals = ttnn.to_torch(d.values).numpy().reshape(27, -1)[:, : L.P]
        offs = ttnn.to_torch(d.offsets).numpy().view(np.int32).reshape(27, -1)[:, : L.P]
        assert np.array_equal(vals.view(np.int32), L.V.view(np.int32)), f"level {l} generated values"
        assert np.array_equal(offs, hp.off[:, : L.P]), f"level {l} generated offsets"
        if l + 1 == len(levels):
            continue
        # grid transfer tables: against the plans and weights of Level.link's R / Pm tables
        C, nb0 = levels[l + 1], L.starts[1]
        cblocks = [(C.starts[c], C.starts[c + 1]) for c in range(8)]
        cb = C.bstarts if C.bands > 1 else None
        fb = [L.bstarts[0]] if L.bands > 1 else None
        nc = g.x * g.y
        rp = SparseRowPlan(L.R[None, :], cblocks, [(0, nb0)], nc, noc_coords(device), False, bands=cb)
        pp = SparseRowPlan(L.Pm[None, :], [(0, nb0)], cblocks, nc, noc_coords(device), False, bands=fb)
        for name, t, o, p, w in (("restriction", d.Rt, d.Roff, rp, L.Rw), ("prolongation", d.Pt, d.Poff, pp, L.Pw)):
            got_w = ttnn.to_torch(t).numpy().reshape(-1)[: w.size]
            got_o = ttnn.to_torch(o).numpy().view(np.int32).reshape(-1)[: w.size]
            assert np.array_equal(got_w.view(np.int32), np.asarray(w, np.float32).view(np.int32)), f"{l} {name} w"
            assert np.array_equal(got_o, p.off.reshape(-1)[: w.size]), f"level {l} generated {name} offsets"
    assert np.array_equal(from_device(h.b), levels[0].B.astype(np.float64)), "generated b"
    assert not np.any(from_device(h.x)) and not np.any(from_device(h.p)), "generated x = 0 and p = 0"

    rng = np.random.default_rng(0)
    for l, d in enumerate(h.lv):
        L = levels[l]
        xs = np.where(L.real, rng.uniform(0.5, 1.5, L.P), 0.0).astype(np.float32)
        ttnn.copy(to_device(device, xs), d.z)
        h.spmv(l)()
        want = ref.spmv(L, xs, np.float32).astype(np.float64)
        assert np.array_equal(from_device(d.Az), want), f"level {l} SpMV is not bit-exact"

        rs = np.where(L.real, rng.uniform(-1, 1, L.P), 0.0).astype(np.float32)
        ttnn.copy(to_device(device, rs), d.r)
        ttnn.copy(to_device(device, xs), d.z)
        h.ops[("gs", l)]()
        x32 = xs.copy()
        ref.symgs(L, x32, rs, np.float32, order)
        err = np.max(np.abs(from_device(d.z) - x32))
        logger.info(f"level {l} SYMGS max abs diff {err:.2e}")
        assert err <= GS_ATOL, f"level {l} SYMGS differs by {err:.2e}"

        # pre-smoother: z holds garbage on entry and must be treated as 0; then Az = A z, block 0 = r - A z
        h.ops[("gs_pre", l)]()
        x32 = np.zeros_like(xs)
        ref.symgs(L, x32, rs, np.float32, order)
        got = from_device(d.z)
        err = np.max(np.abs(got - x32))
        assert err <= GS_ATOL, f"level {l} pre-smoother differs by {err:.2e}"
        if l < 3:
            a32 = ref.spmv(L, got.astype(np.float32), np.float32).astype(np.float64)
            nb0 = L.starts[1]
            a32[:nb0] = (rs[:nb0] - a32[:nb0].astype(np.float32)).astype(np.float64)
            assert np.array_equal(from_device(d.Az), a32), f"level {l} residual SpMV is not bit-exact"

    for l in range(3):
        d, L, dc = h.lv[l], levels[l], h.lv[l + 1]
        nb0 = L.starts[1]
        t = np.zeros(L.P, np.float32)
        t[:nb0] = rng.uniform(-1, 1, nb0)
        ttnn.copy(to_device(device, t), d.Az)  # the restriction reads r - A z from block 0 of Az
        h.ops[("restrict", l)]()
        assert np.array_equal(from_device(dc.r), (t[:nb0][L.R] * L.Rw).astype(np.float64)), f"level {l} restriction"
        zc = rng.uniform(-1, 1, dc.P).astype(np.float32)
        zf = rng.uniform(-1, 1, d.P).astype(np.float32)
        ttnn.copy(to_device(device, zc), dc.z)
        ttnn.copy(to_device(device, zf), d.z)
        h.ops[("prolong", l)]()
        want = zf.astype(np.float64)
        want[:nb0] = (zf[:nb0] + (zc[L.Pm] * L.Pw).astype(np.float32)).astype(np.float64)
        assert np.array_equal(from_device(d.z), want), f"level {l} prolongation"

    L0 = levels[0]
    rs = np.where(L0.real, rng.uniform(-1, 1, L0.P), 0.0).astype(np.float32)
    ttnn.copy(to_device(device, rs), h.r)
    h.mg(0)
    ref64 = ref.mg(levels, 0, rs.astype(np.float64), np.float64, order)
    rel = np.max(np.abs(from_device(h.lv[0].z) - ref64)) / np.max(np.abs(ref64))
    logger.info(f"multigrid V-cycle relative error vs fp64 {rel:.2e}")
    assert rel < 1e-6

    iters = 3
    for _ in range(2):  # reset() and start() must give the same state when run again (the demo repeats them)
        h.reset()
        nr0 = float(ttnn.to_torch(h.start()).reshape(-1)[0])
    _, hist64 = ref.cg(levels, iters, np.float64, order)
    hist = []
    for k in range(iters):
        nr = float(ttnn.to_torch(h.iteration()).reshape(-1)[0]) / nr0
        hist.append(nr)
        logger.info(f"CG {k + 1}: device {nr:.6e}  numpy fp64 {hist64[k]:.6e}")
        assert abs(nr - hist64[k]) <= 1e-5 * hist64[k], f"CG iteration {k + 1}"

    # a diverged solve leaves NaN in x and p: reset() writes zeros (x - x would stay NaN), so the next solve gives
    # the same residuals
    nan = np.full(L0.P, np.nan, np.float32)
    ttnn.copy(to_device(device, nan), h.x)
    ttnn.copy(to_device(device, nan), h.p)
    h.reset()
    assert not np.any(from_device(h.x)) and not np.any(from_device(h.p)), "reset() writes x = 0 and p = 0"
    nr0 = float(ttnn.to_torch(h.start()).reshape(-1)[0])
    again = [float(ttnn.to_torch(h.iteration()).reshape(-1)[0]) / nr0 for _ in range(iters)]
    assert again == hist, f"CG after a NaN x and p: {again} != {hist}"
