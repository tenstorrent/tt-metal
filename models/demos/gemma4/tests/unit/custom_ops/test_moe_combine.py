# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""MoeCombine (custom op) vs the composite ``permute, mul, fast_reduce_nc`` and the torch golden."""

import pytest
import torch

import ttnn
from models.demos.gemma4.custom_ops.moe_combine import MoeCombine

SINGLE_CHIP = [pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-chip")]


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def _routing(E, R, k=8, seed=0):
    """[1, 1, E, 32] bf16 table: every user picks k of the E local experts with normalized random weights."""
    g = torch.Generator().manual_seed(seed)
    w = torch.zeros(1, 1, E, 32)
    for u in range(R):
        idx = torch.randperm(E, generator=g)[:k]
        vals = torch.rand(k, generator=g) + 0.1
        w[0, 0, idx, u] = vals / vals.sum()
    return w.to(torch.bfloat16)


def _composite(down, w_tab):
    """the shipped path: [1,E,R,1] weights (permute of the [1,1,R,E] dense routing) * down, then fast_reduce_nc."""
    E, R = int(down.shape[1]), int(down.shape[2])
    dense = ttnn.to_torch(w_tab).float()[0, 0, :, :R].t().reshape(1, 1, R, E)  # [1,1,R,E]
    routing = ttnn.from_torch(
        dense.to(torch.bfloat16), device=down.device(), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )
    routing_3d = ttnn.permute(routing, (0, 3, 2, 1))  # [1, E, R, 1]
    weighted = ttnn.mul(down, routing_3d)
    return ttnn.unsqueeze_to_4D(ttnn.experimental.fast_reduce_nc(weighted, dims=[1]))


@pytest.mark.parametrize(
    "E, R, H, k, mem",
    [
        (1, 32, 256, 1, "dram"),
        (2, 32, 256, 2, "dram"),
        (3, 32, 256, 3, "dram"),
        (4, 32, 256, 4, "dram"),
        (8, 32, 256, 8, "dram"),
        (8, 32, 256, 2, "dram"),
        (16, 32, 256, 16, "dram"),
        (32, 32, 2816, 8, "dram"),
        (32, 1, 2816, 2, "l1"),
        (32, 32, 2816, 32, "dram"),
        (32, 8, 2816, 0, "dram"),
    ],
    ids=[
        "one_expert",
        "two_experts",
        "three",
        "four",
        "eight",
        "eight_two_active",
        "sixteen",
        "b32_top8",
        "b1_two_experts_l1",
        "b32_all_experts",
        "b8_none_selected",
    ],
)
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_moe_combine_matches_composite(mesh_device, device_params, E, R, H, k, mem):
    torch.manual_seed(0)
    down = torch.randn(1, E, R, H).to(torch.bfloat16)
    w = _routing(E, R, k=k)
    golden = MoeCombine.golden(down.float(), w.float())
    memcfg = ttnn.DRAM_MEMORY_CONFIG if mem == "dram" else ttnn.L1_MEMORY_CONFIG
    t_down = ttnn.from_torch(
        down, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=memcfg
    )
    t_w = ttnn.from_torch(w, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
    out = ttnn.to_torch(MoeCombine.op(t_down, t_w)).float()
    assert torch.isfinite(out).all()
    assert list(out.shape) == [1, 1, R, H]
    d = (out - golden).abs()[0, 0]  # [R, H]
    print(
        f"[E{E} R{R} k{k}] max|diff| per 16-col half of tile 0: {[round(float(d[:, i*16:(i+1)*16].max()), 3) for i in range(2)]}, per row (first 8): {[round(float(d[r].max()), 3) for r in range(min(R, 8))]}, golden max {float(golden.abs().max()):.3f}"
    )
    if k == 0:
        assert out.abs().max() == 0, "no selected expert must give zeros"
        return
    ref = ttnn.to_torch(_composite(t_down, t_w)).float()[:, :, :R, :]  # the shipped path keeps the padded 32 rows
    for name, other in (("golden", golden), ("composite", ref)):
        pcc = _pcc(other, out)
        print(f"[E{E} R{R} k{k} {mem}] pcc vs {name}: {pcc:.6f} max|diff| {(other - out).abs().max():.4f}")
        assert pcc > 0.9999, f"MoeCombine diverges from the {name}: pcc={pcc}"
    # program-cache hit on fresh buffers (new addresses re-applied as runtime args)
    t_down2 = ttnn.from_torch(
        down, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=memcfg
    )
    t_w2 = ttnn.from_torch(w, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
    out2 = ttnn.to_torch(MoeCombine.op(t_down2, t_w2)).float()
    assert torch.equal(out2, out), f"cache-hit call differs: max|diff| {(out2 - out).abs().max()}"
