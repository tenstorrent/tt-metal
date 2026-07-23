# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 3 — perf-flagged profile (K/V reuse multicast).

Device-ns measurement + correctness for the mandatory perf target shape
(feature_spec.LOOSE_CASES): B=1, H=10, S=9472, D=128 bf16 @ fp32_dest_acc_en=False,
mask none, auto scale, self/MHA. Run with `run_safe_pytest.sh --profile` to get the
per-op DEVICE KERNEL DURATION [ns] from Tracy. Correctness gate: PCC >= 0.997.

DO NOT DELETE — this is the perf baseline/target harness for Refinements 3 & 5.
"""
import math

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

# The flagged shape (feature_spec.LOOSE_CASES).
FLAG_SHAPE = (1, 10, 9472, 128)


def _fa_rand(*shape, seed=1234):
    """Heavy-tailed flash-attention input distribution (matches the golden harness)."""
    torch.manual_seed(seed)
    normal = torch.randn(shape)
    # heavy tail: scale a random subset up
    bernoulli = torch.bernoulli(torch.full(shape, 0.001))
    return normal + bernoulli * torch.randn(shape) * 10.0


def _reference(Q, K, V, scale):
    return torch.nn.functional.scaled_dot_product_attention(
        Q.float(), K.float(), V.float(), attn_mask=None, is_causal=False, scale=scale
    )


@pytest.mark.parametrize("shape", [FLAG_SHAPE])
def test_flagged_shape_perf(device, shape):
    B, H, S, D = shape
    scale = 1.0 / math.sqrt(D)

    torch_q = _fa_rand(B, H, S, D)
    torch_k = _fa_rand(B, H, S, D, seed=1235)
    torch_v = _fa_rand(B, H, S, D, seed=1236)

    q = ttnn.from_torch(torch_q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    k = ttnn.from_torch(torch_k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    v = ttnn.from_torch(torch_v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_dest_acc_en=False,
        math_approx_mode=False,
    )

    out = scaled_dot_product_attention(q, k, v, compute_kernel_config=cfg)
    result = ttnn.to_torch(out).float()

    ref = _reference(torch_q, torch_k, torch_v, scale)
    assert_with_pcc(ref, result, 0.997)
