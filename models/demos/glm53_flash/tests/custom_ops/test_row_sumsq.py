# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``custom_ops/row_sumsq`` vs the prefill mHC projection's ``ttnn.sum(ttnn.multiply(x, x), dim=-1, keepdim=True)``
(tt-blaze G53F-0209), on the projection's input shape ``[1, 1, T, 16384]`` fp32 (the 4 residual streams per chip).

For each T in ``RS_TS`` rows: random rows with a per-row scale over ~6 decades, replicated to every chip; the fused
op's row sums must equal the composite's bitwise on every chip (both run on the device), and so must the whole
projection (``RMSNorm(x) @ fn_T``, the composite ``_project`` vs the same with the fused sums). Then both forms of
the sums are timed (``RS_REPS`` back-to-back calls, one synchronize).

    pytest models/demos/glm53_flash/tests/custom_ops/test_row_sumsq.py -s
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import _project as composite_project
from models.demos.glm53_flash.custom_ops.row_sumsq.op import RowSumSq
from models.demos.glm53_flash.tt.mesh import mesh_device_params

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "8,4").split(","))
W = 16384
TS = [int(v) for v in os.environ.get("RS_TS", "2048,1024,64").split(",")]
REPS = int(os.environ.get("RS_REPS", "10"))
EPS = 1e-6


def _fused_project(x, fn_T, ckc, sumsq):
    """``composite_project`` (tp_sum None) with the fused row sums: the same ops otherwise, in the same order."""
    mixes_un = ttnn.matmul(x, fn_T, compute_kernel_config=ckc)
    ss = sumsq(x)
    ms = ttnn.multiply(ss, 1.0 / x.shape[-1])
    return ttnn.multiply(mixes_un, ttnn.rsqrt(ttnn.add(ms, EPS)))


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH, trace_region_size=0)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_row_sumsq(mesh_device):
    g = torch.Generator().manual_seed(9)
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    op = RowSumSq(mesh_device)
    ckc = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
    )
    fn_T = ttnn.from_torch(
        torch.randn(W, 24, generator=g) * 0.01,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=rep,
    )
    ok = True
    for T in TS:
        xh = torch.randn(1, 1, T, W, generator=g) * torch.exp(3 * torch.randn(1, 1, T, 1, generator=g))
        x = ttnn.from_torch(xh, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)
        composite = lambda: ttnn.sum(ttnn.multiply(x, x), dim=-1, keepdim=True)
        ref, got = composite(), op(x)
        assert list(got.shape) == list(ref.shape), (got.shape, ref.shape)
        rows = lambda t: [ttnn.to_torch(c).reshape(T, 1).view(torch.int32) for c in ttnn.get_device_tensors(t)]
        r_c, g_c = rows(ref), rows(got)
        eq = sum(torch.equal(a, b) for a, b in zip(r_c, g_c))
        diff = int((r_c[0] != g_c[0]).sum())
        host = (xh.double() ** 2).sum(-1).reshape(T, 1)
        rel = float(((g_c[0].view(torch.float32).double() - host).abs() / host).max())
        p_ref = composite_project(x, fn_T, EPS, ckc)
        p_got = _fused_project(x, fn_T, ckc, op)
        mix = lambda t: [ttnn.to_torch(c).reshape(T, 24).view(torch.int32) for c in ttnn.get_device_tensors(t)]
        peq = sum(torch.equal(a, b) for a, b in zip(mix(p_ref), mix(p_got)))
        for t in (ref, got, p_ref, p_got):
            ttnn.deallocate(t)

        def timed(fn):
            for _ in range(2):
                ttnn.deallocate(fn())
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            for _ in range(REPS):
                ttnn.deallocate(fn())
            ttnn.synchronize_device(mesh_device)
            return (time.perf_counter() - t0) / REPS * 1e3

        ms_c, ms_f = timed(composite), timed(lambda: op(x))
        logger.info(
            f"[row-sumsq] T={T}: sums == composite on {eq} / {len(r_c)} chips ({diff} of {T} rows differ on chip 0; "
            f"max rel vs float64 {rel:.2e}); projection == composite on {peq} / {len(r_c)} chips; "
            f"composite square + sum {ms_c:.3f} ms, fused {ms_f:.3f} ms per call"
        )
        ok &= eq == len(r_c) and peq == len(r_c)
        ttnn.deallocate(x)
    assert ok, "the fused row sums differ from the composite's"
