# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Every device op against the numpy reference of the same algorithm (reference/problem.py), then a few CG
iterations against the fp64 reference.

SpMV, restriction and prolongation must be bit-exact in fp32. A Gauss-Seidel update differs from numpy by at most
a couple of fp32 roundings in the divide, so smoothers are checked to 5e-7 absolute (values are O(1)).
"""

import numpy as np
import pytest
import ttnn
from loguru import logger

from models.experimental.hpcg.reference import problem as ref
from models.experimental.hpcg.tt.hpcg import HPCG
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
    rng = np.random.default_rng(0)
    for l, d in enumerate(h.lv):
        L = d.L
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
        d, L, dc = h.lv[l], h.levels[l], h.lv[l + 1]
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

    L0 = h.levels[0]
    rs = np.where(L0.real, rng.uniform(-1, 1, L0.P), 0.0).astype(np.float32)
    ttnn.copy(to_device(device, rs), h.r)
    h.mg(0)
    ref64 = ref.mg(h.levels, 0, rs.astype(np.float64), np.float64, order)
    rel = np.max(np.abs(from_device(h.lv[0].z) - ref64)) / np.max(np.abs(ref64))
    logger.info(f"multigrid V-cycle relative error vs fp64 {rel:.2e}")
    assert rel < 1e-6

    iters = 3
    h.reset()
    nr0 = float(ttnn.to_torch(h.start()).reshape(-1)[0])
    _, hist64 = ref.cg(h.levels, iters, np.float64, order)
    for k in range(iters):
        nr = float(ttnn.to_torch(h.iteration()).reshape(-1)[0]) / nr0
        logger.info(f"CG {k + 1}: device {nr:.6e}  numpy fp64 {hist64[k]:.6e}")
        assert abs(nr - hist64[k]) <= 1e-5 * hist64[k], f"CG iteration {k + 1}"
