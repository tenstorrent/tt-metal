# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""GegluDecode (custom op) vs the composite ``apply_geglu_fused`` (slice, slice, gelu, mul) and the torch golden;
GegluPrefill (the prefill rows of the shared MLP) bitwise against the composite."""

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


@pytest.mark.parametrize(
    "R, I",
    [(4096, 3840), (2048, 3840), (96, 3840), (2048, 5376), (4096, 544)],
    ids=["dense12b_tp4_4k", "dense12b_tp4_2k", "dense12b_tp4_96rows", "dense31b_tp4_2k", "shared26b_tp4_4k"],
)
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_geglu_prefill_bitwise_composite(mesh_device, device_params, R, I):
    """GegluPrefill is bitwise the composite slice, slice, gelu, mul (and its input-activation form)."""
    from models.demos.gemma4.custom_ops.geglu_prefill import GegluPrefill

    torch.manual_seed(0)
    x = (torch.randn(1, 1, R, 2 * I) * 2).to(torch.bfloat16)

    def dev(t):
        return ttnn.from_torch(t, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    tx = dev(x)
    up = ttnn.slice(tx, [0, 0, 0, 0], [1, 1, R, I])
    gate = ttnn.slice(tx, [0, 0, 0, I], [1, 1, R, 2 * I])
    ref = ttnn.to_torch(ttnn.mul(ttnn.gelu(gate, variant=ttnn.GeluVariant.Accurate), up)).float()
    act = [ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 0.0)]
    ref_act = ttnn.to_torch(ttnn.mul(gate, up, input_tensor_a_activations=act)).float()
    out = ttnn.to_torch(GegluPrefill.op(tx)).float()
    print(f"[R{R} I{I}] bitwise composite {torch.equal(out, ref)} activation form {torch.equal(ref_act, ref)}")
    assert torch.equal(ref_act, ref), f"activation form max|diff| {(ref_act - ref).abs().max()}"
    assert torch.equal(out, ref), f"GegluPrefill max|diff| vs composite {(out - ref).abs().max()}"
    # program-cache hit on fresh buffers
    assert torch.equal(ttnn.to_torch(GegluPrefill.op(dev(x))).float(), out)
