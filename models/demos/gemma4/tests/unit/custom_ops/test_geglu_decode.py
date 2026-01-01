# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""GegluDecode (custom op) vs the composite ``apply_geglu_fused`` (slice, slice, gelu, mul) and the torch golden."""

import pytest
import torch

import ttnn
from models.demos.gemma4.custom_ops.geglu_decode import GegluDecode
from models.demos.gemma4.tt.experts.prefill import apply_geglu_fused

SINGLE_CHIP = [pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-chip")]


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


@pytest.mark.parametrize(
    "E, R, I, mem",
    [
        (32, 32, 704, "dram"),
        (1, 32, 544, "dram"),
        (32, 1, 704, "l1"),
        (4, 64, 704, "dram"),
        (1, 128, 96, "dram"),
        (1, 128, 288, "dram"),
        # dense MLP per-chip halves: 31B tp4 5376 / tp8 2688, 12B tp4 3840 / tp8 1920
        (1, 32, 5376, "dram"),
        (1, 1, 5376, "dram"),
        (1, 32, 2688, "dram"),
        (1, 32, 3840, "dram"),
        (1, 32, 1920, "l1"),
    ],
    ids=[
        "experts_b32",
        "shared_mlp_b32",
        "experts_b1_l1",
        "two_row_tiles",
        "lanes_mlp_8x4",
        "lanes_mlp_2x4",
        "dense31b_tp4",
        "dense31b_tp4_b1",
        "dense31b_tp8",
        "dense12b_tp4",
        "dense12b_tp8_l1",
    ],
)
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_geglu_decode_matches_composite(mesh_device, device_params, E, R, I, mem):
    torch.manual_seed(0)
    x = (torch.randn(1, E, R, 2 * I) * 2).to(torch.bfloat16)
    golden = GegluDecode.golden(x.float())
    memcfg = ttnn.DRAM_MEMORY_CONFIG if mem == "dram" else ttnn.L1_MEMORY_CONFIG

    def dev(t):
        return ttnn.from_torch(
            t, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=memcfg
        )

    tx = dev(x)
    ref = ttnn.to_torch(apply_geglu_fused(tx)).float()
    out = ttnn.to_torch(GegluDecode.op(tx)).float()
    assert torch.isfinite(out).all()
    for name, other in (("golden", golden), ("composite", ref)):
        pcc = _pcc(other, out)
        print(f"[E{E} R{R} I{I} {mem}] pcc vs {name}: {pcc:.6f} max|diff| {(other - out).abs().max():.4f}")
        assert pcc > 0.9999, f"GegluDecode diverges from the {name}: pcc={pcc}"
    # program-cache hit on freshly allocated inputs (new buffer addresses must be re-applied as runtime args)
    out2 = ttnn.to_torch(GegluDecode.op(dev(x))).float()
    assert torch.equal(out2, out), f"cache-hit call differs: max|diff| {(out2 - out).abs().max()}"


@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_geglu_decode_perf_vs_composite(mesh_device, device_params):
    """Host wall per call (dispatch-bound outside tracy); device time from a profiler run of this selector."""
    import time

    torch.manual_seed(0)
    x = torch.randn(1, 32, 32, 2 * 704, dtype=torch.bfloat16)
    tx = ttnn.from_torch(x, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    ITERS = 20
    for name, fn in (("composite", lambda: apply_geglu_fused(tx)), ("geglu_decode", lambda: GegluDecode.op(tx))):
        for _ in range(3):
            fn()
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for _ in range(ITERS):
            out = fn()
        ttnn.synchronize_device(mesh_device)
        print(f"[perf E32 R32 I704] {name}: host {(time.perf_counter() - t0) / ITERS * 1e6:.1f} us/call", flush=True)
        ttnn.deallocate(out)
