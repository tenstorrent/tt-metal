# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision matrix for scaled_dot_product_attention (Refinement 1).

Single authoritative precision characterization across the numerical surface
opened by Refinement 1: dtype ∈ {bfloat16, float32, bfloat8_b} × fp32_dest_acc_en
∈ {True, False} × input distribution ∈ {normal, uniform}, on tile-aligned shapes
that fit L1 (D ≤ 128 — the large-head-dim OOM cells are Refinement 2's budget).

PCC is the sole gate; all other metrics are printed for observability. Thresholds
mirror the golden TOLERANCES map (helpers.py). The op-side EXCLUSION
{float32, fp32_dest_acc_en=False} is skipped (legal-but-lossy, refused).

DO NOT DELETE — this is the precision baseline the numerical refinements defend.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc


# (dtype, fp32_dest_acc_en) → PCC gate for the `normal` distribution. Mirrors
# helpers.TOLERANCES (the golden gate is calibrated on randn / fa_rand).
_PCC = {
    (ttnn.float32, True): 0.999,
    (ttnn.bfloat16, True): 0.995,
    (ttnn.bfloat16, False): 0.99,
    (ttnn.bfloat8_b, True): 0.99,
    (ttnn.bfloat8_b, False): 0.99,
}

# Uniform [0,1] is a documented stress distribution: the small, all-positive
# dynamic range flattens the softmax so relative error is amplified (the golden
# `test_uniform_input` regression exercises the same effect). fp32 still nails
# it (~0.9999); the block-float / 16-bit-DEST cells bottom out near 0.98, so the
# uniform gate is a floor, not the tight randn tolerance.
_PCC_UNIFORM_FLOOR = 0.98

# Tile-aligned shapes that fit L1 at every dtype (D ≤ 128).
SHAPES = [
    (1, 1, 32, 32),  # single tile
    (1, 1, 128, 64),  # multi-KV-chunk
    (1, 4, 256, 64),  # multi-head, deeper sequence
    (2, 4, 128, 128),  # batched, wider head_dim
]

# torch has no native bfloat8_b — quantization happens on device; reference in bf16.
_TORCH_DTYPE = {
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
    ttnn.bfloat8_b: torch.bfloat16,
}


def _reference(Q, K, V):
    return torch.nn.functional.scaled_dot_product_attention(Q.float(), K.float(), V.float())


@pytest.mark.parametrize("distribution", ["normal", "uniform"])
@pytest.mark.parametrize(
    "fp32_acc",
    [pytest.param(True, id="fp32_acc"), pytest.param(False, id="bf16_acc")],
)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(ttnn.bfloat16, id="bf16"),
        pytest.param(ttnn.float32, id="fp32"),
        pytest.param(ttnn.bfloat8_b, id="bfp8"),
    ],
)
@pytest.mark.parametrize("shape", SHAPES, ids=[f"B{s[0]}_H{s[1]}_S{s[2]}_D{s[3]}" for s in SHAPES])
def test_precision_matrix(device, shape, dtype, fp32_acc, distribution):
    from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

    if dtype == ttnn.float32 and not fp32_acc:
        pytest.skip("op-side EXCLUSION: {float32, fp32_dest_acc_en=False} is legal-but-lossy, refused")

    torch.manual_seed(42)
    torch_dtype = _TORCH_DTYPE[dtype]
    if distribution == "uniform":
        Q = torch.rand(shape, dtype=torch_dtype)
        K = torch.rand(shape, dtype=torch_dtype)
        V = torch.rand(shape, dtype=torch_dtype)
    else:
        Q = torch.randn(shape, dtype=torch_dtype)
        K = torch.randn(shape, dtype=torch_dtype)
        V = torch.randn(shape, dtype=torch_dtype)

    expected = _reference(Q, K, V)

    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,  # op clamps bf16/bf8b to HiFi2 internally (#38306)
        fp32_dest_acc_en=fp32_acc,
        math_approx_mode=False,
    )
    to_dev = lambda t: ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = ttnn.to_torch(
        scaled_dot_product_attention(to_dev(Q), to_dev(K), to_dev(V), compute_kernel_config=cfg)
    ).float()

    abs_err = (out - expected).abs()
    rel_rms = (torch.sqrt((abs_err**2).mean()) / expected.float().pow(2).mean().sqrt().clamp(min=1e-10)).item()
    pcc_gate = _PCC_UNIFORM_FLOOR if distribution == "uniform" else _PCC[(dtype, fp32_acc)]
    passed, msg = check_with_pcc(expected, out, pcc_gate)
    print(
        f"\n[precision-matrix] {shape} dtype={dtype} acc={fp32_acc} dist={distribution}: "
        f"{msg} rel_rms={rel_rms:.5f} max_abs={abs_err.max().item():.5f}"
    )
    assert passed, msg
