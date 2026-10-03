# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused mHC collapse + norm kernels (tt/mhc_collapse.py) vs the exact (fp64) result and vs layer's rms_norm path."""
import math

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.test_mhc_opt import chain_ms, first, pcc
from models.demos.blackhole.deepseek_v41_flash.tt.mhc_collapse import mhc_collapse, mhc_norm_apply

D, T = 5120, 4


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 100_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@torch.no_grad()
def test_mhc_collapse_kernel(mesh_device):
    md = mesh_device
    torch.manual_seed(0)
    rep = ttnn.ReplicateTensorToMesh(md)
    up = lambda t, dt=ttnn.float32: ttnn.from_torch(
        t, device=md, dtype=dt, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep
    )
    x = torch.randn(T, 1, 4, D) + 0.5
    x[..., 3] *= 200
    pre = torch.rand(T, 1, 1, 4) * 1.5
    w = torch.rand(1, 1, 1, D) + 0.5
    eps = 1e-20
    tx, tpre = up(x), up(pre)
    w4 = up((w * math.sqrt(D)).expand(1, 1, T, D).contiguous())
    tw = up(w)
    h = (pre.double().reshape(T, 4, 1) * x.double().reshape(T, 4, D)).sum(1)
    hb = h.to(torch.bfloat16).double()
    exact = hb * torch.rsqrt((hb * hb).mean(-1, keepdim=True) + eps) * w.double().reshape(1, D)

    th, tpart = mhc_collapse(tx, tpre)
    print("MB h PCC", pcc(first(th).reshape(T, D), h), "partial0", first(tpart)[0, 0, :4, :2].flatten().tolist())
    out = first(mhc_norm_apply(th, tpart, w4, eps)).reshape(T, D)
    print(
        "MB norm PCC vs exact",
        pcc(out, exact),
        "max rel err",
        float(((out - exact).abs() / exact.abs().clamp_min(1e-3)).max()),
    )
    ckc = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
    )
    ref = lambda: ttnn.rms_norm(
        ttnn.typecast(
            ttnn.reshape(
                ttnn.matmul(tpre, tx, compute_kernel_config=ckc, core_grid=ttnn.CoreGrid(y=4, x=8)), [1, 1, T, D]
            ),
            ttnn.bfloat16,
        ),
        epsilon=eps,
        weight=tw,
    )
    rout = first(ref()).reshape(T, D)
    print("MB layer-style rms_norm PCC vs exact", pcc(rout, exact))
    print(f"MB TIME ref collapse+norm {chain_ms(md, ref):.3f} ms")
    print(f"MB TIME op1 collapse {chain_ms(md, lambda: mhc_collapse(tx, tpre)):.3f} ms")
    print(f"MB TIME op1+op2 {chain_ms(md, lambda: mhc_norm_apply(*mhc_collapse(tx, tpre), w4, eps)):.3f} ms")
    assert pcc(out, exact) > 0.9999
