# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""RmsNormDecode (custom op, N cores exchanging partial sums) vs ttnn.rms_norm and the torch golden."""

import pytest
import torch

import ttnn
from models.demos.gemma4.custom_ops.rms_norm_decode import RmsNormDecode

SINGLE_CHIP = [pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-chip")]
EPS = 1e-6


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


@pytest.mark.parametrize(
    "R, W, gamma, scalar, mem",
    [
        (32, 2816, True, 1.0, "dram"),
        (1, 2816, True, 1.0, "dram"),
        (32, 2816, False, 2816**-0.5, "dram"),
        (32, 2816, True, 1.0, "l1"),
        (8, 1024, True, 1.0, "dram"),
    ],
    ids=["b32_gamma", "b1_gamma", "b32_router_scalar", "b32_l1", "b8_w1024"],
)
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_rms_norm_decode_matches_rms_norm(mesh_device, device_params, R, W, gamma, scalar, mem):
    torch.manual_seed(0)
    x = (torch.randn(1, 1, R, W) * 4).to(torch.bfloat16)
    g = (torch.rand(W) + 0.5).to(torch.bfloat16) if gamma else None
    golden = RmsNormDecode.golden(x, g, EPS, scalar)
    memcfg = ttnn.DRAM_MEMORY_CONFIG if mem == "dram" else ttnn.L1_MEMORY_CONFIG
    tx = ttnn.from_torch(x, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=memcfg)
    tg = (
        ttnn.from_torch(g.reshape(1, 1, 1, W), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        if gamma
        else None
    )
    # the shipped op: rms_norm with the ROW_MAJOR gamma (decode path), then the scalar
    g_rm = (
        ttnn.from_torch(g.reshape(1, 1, -1, 32), device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
        if gamma
        else None
    )
    ref = ttnn.rms_norm(tx, weight=g_rm, epsilon=EPS) if gamma else ttnn.rms_norm(tx, epsilon=EPS)
    if scalar != 1.0:
        ref = ttnn.mul(ref, scalar)
    ref = ttnn.to_torch(ref).float()
    out = ttnn.to_torch(RmsNormDecode.op(tx, tg, EPS, scalar)).float()
    assert torch.isfinite(out).all()
    for name, other in (("golden", golden), ("rms_norm", ref)):
        pcc = _pcc(other, out)
        print(
            f"[R{R} W{W} gamma={gamma} scalar={scalar:.4f} {mem} N={RmsNormDecode.num_cores(W // 32)}] pcc vs {name}: {pcc:.6f} max|diff| {(other - out).abs().max():.4f}"
        )
        assert pcc > 0.9999, (name, pcc)
    # program-cache hits on fresh buffers, twice (semaphores must be reset between runs)
    for _ in range(2):
        tx2 = ttnn.from_torch(x, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=memcfg)
        out2 = ttnn.to_torch(RmsNormDecode.op(tx2, tg, EPS, scalar)).float()
        assert torch.equal(out2, out), f"cache-hit call differs: max|diff| {(out2 - out).abs().max()}"
