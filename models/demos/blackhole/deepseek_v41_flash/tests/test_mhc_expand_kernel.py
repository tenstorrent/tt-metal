# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused mHC expand kernel (tt/mhc_expand.py) vs the composite (matmul + addcmul)."""
import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.test_mhc_opt import chain_ms, first, pcc
from models.demos.blackhole.deepseek_v41_flash.tt.mhc_expand import mhc_expand

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
def test_mhc_expand_kernel(mesh_device):
    md = mesh_device
    torch.manual_seed(0)
    rep = ttnn.ReplicateTensorToMesh(md)
    up = lambda t: ttnn.from_torch(
        t,
        device=md,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep,
    )
    x, y = torch.randn(T, 1, 4, D) * 3, torch.randn(T, 1, 1, D)
    post, comb = torch.rand(T, 1, 4, 1) * 2, torch.softmax(torch.randn(T, 1, 4, 4), -1)
    tx, ty, tp, tc = up(x), up(y), up(post), up(comb)
    exact = torch.einsum("tij,tid->tjd", comb[:, 0].double(), x[:, 0].double()) + post[:, 0].double() * y[:, 0].double()
    out = first(mhc_expand(ty, tx, tp, tc))
    print(
        "MB expand PCC vs exact",
        pcc(out.reshape(T, 4, D), exact),
        "max abs err",
        float((out.reshape(T, 4, D) - exact).abs().max()),
    )
    ckc = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
    )
    comp = lambda: ttnn.addcmul(
        ttnn.matmul(tc, tx, transpose_a=True, compute_kernel_config=ckc, core_grid=ttnn.CoreGrid(y=4, x=8)), tp, ty
    )
    cout = first(comp())
    print("MB composite max abs err", float((cout.reshape(T, 4, D) - exact).abs().max()))
    print(f"MB TIME composite {chain_ms(md, comp):.3f} ms")
    for nc in (20, 32, 40, 80):
        print(f"MB TIME fused cores<={nc} {chain_ms(md, lambda: mhc_expand(ty, tx, tp, tc, nc)):.3f} ms")
    assert pcc(out.reshape(T, 4, D), exact) > 0.999999
