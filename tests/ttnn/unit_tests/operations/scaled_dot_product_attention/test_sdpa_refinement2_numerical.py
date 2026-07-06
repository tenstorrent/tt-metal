# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 2 — Numerical configurability expansion tests.

Tests the new dtype (float32, bfloat8_b) and fp32_dest_acc_en=False
configurations directly, across multiple shapes and mask/scale modes.
"""

from __future__ import annotations

import math

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


def _make_inputs(q_shape, k_shape, v_shape, torch_dtype, device, dtype, layout):
    torch.manual_seed(42)
    Q = torch.randn(q_shape, dtype=torch_dtype)
    K = torch.randn(k_shape, dtype=torch_dtype)
    V = torch.randn(v_shape, dtype=torch_dtype)
    ttnn_Q = ttnn.from_torch(Q, dtype=dtype, layout=layout, device=device)
    ttnn_K = ttnn.from_torch(K, dtype=dtype, layout=layout, device=device)
    ttnn_V = ttnn.from_torch(V, dtype=dtype, layout=layout, device=device)
    return Q, K, V, ttnn_Q, ttnn_K, ttnn_V


def _run_sdpa(ttnn_Q, ttnn_K, ttnn_V, device, dtype, fp32_dest_acc_en, attn_mask=None, scale=None):
    compute_kernel_config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=fp32_dest_acc_en,
        math_approx_mode=False,
    )
    from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

    return scaled_dot_product_attention(
        ttnn_Q,
        ttnn_K,
        ttnn_V,
        attn_mask=attn_mask,
        scale=scale,
        compute_kernel_config=compute_kernel_config,
    )


def _torch_ref(Q, K, V, attn_mask=None, scale=None):
    Qf = Q.to(torch.float32)
    Kf = K.to(torch.float32)
    Vf = V.to(torch.float32)
    am = attn_mask.to(torch.float32) if attn_mask is not None else None
    return torch.nn.functional.scaled_dot_product_attention(Qf, Kf, Vf, attn_mask=am, scale=scale)


# ---- Test shapes ----
SHAPES = [
    ((1, 1, 32, 32), (1, 1, 32, 32), (1, 1, 32, 32)),
    ((1, 1, 128, 64), (1, 1, 128, 64), (1, 1, 128, 64)),
    ((1, 1, 256, 64), (1, 1, 256, 64), (1, 1, 256, 64)),
    ((1, 1, 128, 128), (1, 1, 128, 128), (1, 1, 128, 128)),
    ((1, 8, 128, 64), (1, 8, 128, 64), (1, 8, 128, 64)),
    ((2, 4, 128, 64), (2, 4, 128, 64), (2, 4, 128, 64)),
]


@pytest.mark.parametrize(
    "q_shape, k_shape, v_shape", SHAPES, ids=[f"Q{'x'.join(str(d) for d in s[0])}" for s in SHAPES]
)
@pytest.mark.parametrize("fp32_dest_acc_en", [True, False], ids=["fp32_acc", "bf16_acc"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b], ids=["bf16", "fp32", "bf8b"])
def test_sdpa_dtype_and_acc(device, q_shape, k_shape, v_shape, dtype, fp32_dest_acc_en):
    """Test all dtype × fp32_dest_acc_en combinations (excluding fp32+False EXCLUSION)."""
    if dtype == ttnn.float32 and not fp32_dest_acc_en:
        pytest.skip("fp32 + fp32_dest_acc_en=False is an op-side EXCLUSION")

    torch_dtype = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32, ttnn.bfloat8_b: torch.bfloat16}[dtype]
    Q, K, V, ttnn_Q, ttnn_K, ttnn_V = _make_inputs(
        q_shape, k_shape, v_shape, torch_dtype, device, dtype, ttnn.TILE_LAYOUT
    )

    expected = _torch_ref(Q, K, V)
    output = _run_sdpa(ttnn_Q, ttnn_K, ttnn_V, device, dtype, fp32_dest_acc_en)
    result = ttnn.to_torch(output)

    pcc_threshold = 0.99 if dtype == ttnn.bfloat8_b else 0.999
    assert_with_pcc(expected.float(), result.float(), pcc=pcc_threshold)


@pytest.mark.parametrize(
    "q_shape, k_shape, v_shape", SHAPES[:3], ids=[f"Q{'x'.join(str(d) for d in s[0])}" for s in SHAPES[:3]]
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b], ids=["bf16", "fp32", "bf8b"])
def test_sdpa_with_mask(device, q_shape, k_shape, v_shape, dtype):
    """Test mask application across dtypes."""
    torch_dtype = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32, ttnn.bfloat8_b: torch.bfloat16}[dtype]
    Q, K, V, ttnn_Q, ttnn_K, ttnn_V = _make_inputs(
        q_shape, k_shape, v_shape, torch_dtype, device, dtype, ttnn.TILE_LAYOUT
    )

    B, _, S_q, _ = q_shape
    S_kv = k_shape[2]
    mask = torch.zeros(B, 1, S_q, S_kv, dtype=torch_dtype)
    mask.masked_fill_(torch.triu(torch.ones(S_q, S_kv, dtype=torch.bool), diagonal=1), float("-inf"))
    ttnn_mask = ttnn.from_torch(mask, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    expected = _torch_ref(Q, K, V, attn_mask=mask)
    output = _run_sdpa(ttnn_Q, ttnn_K, ttnn_V, device, dtype, True, attn_mask=ttnn_mask)
    result = ttnn.to_torch(output)

    pcc_threshold = 0.99 if dtype == ttnn.bfloat8_b else 0.999
    assert_with_pcc(expected.float(), result.float(), pcc=pcc_threshold)


@pytest.mark.parametrize(
    "q_shape, k_shape, v_shape", SHAPES[:3], ids=[f"Q{'x'.join(str(d) for d in s[0])}" for s in SHAPES[:3]]
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b], ids=["bf16", "fp32", "bf8b"])
def test_sdpa_explicit_scale(device, q_shape, k_shape, v_shape, dtype):
    """Test explicit scale across dtypes."""
    torch_dtype = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32, ttnn.bfloat8_b: torch.bfloat16}[dtype]
    Q, K, V, ttnn_Q, ttnn_K, ttnn_V = _make_inputs(
        q_shape, k_shape, v_shape, torch_dtype, device, dtype, ttnn.TILE_LAYOUT
    )

    scale = 0.125
    expected = _torch_ref(Q, K, V, scale=scale)
    output = _run_sdpa(ttnn_Q, ttnn_K, ttnn_V, device, dtype, True, scale=scale)
    result = ttnn.to_torch(output)

    pcc_threshold = 0.99 if dtype == ttnn.bfloat8_b else 0.999
    assert_with_pcc(expected.float(), result.float(), pcc=pcc_threshold)


def test_sdpa_fp32_false_exclusion(device):
    """Verify that fp32 + fp32_dest_acc_en=False raises ExcludedCell."""
    Q = torch.randn(1, 1, 128, 64, dtype=torch.float32)
    K = torch.randn(1, 1, 128, 64, dtype=torch.float32)
    V = torch.randn(1, 1, 128, 64, dtype=torch.float32)
    ttnn_Q = ttnn.from_torch(Q, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_K = ttnn.from_torch(K, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_V = ttnn.from_torch(V, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    compute_kernel_config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=False,
        math_approx_mode=False,
    )
    from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

    with pytest.raises(NotImplementedError):
        scaled_dot_product_attention(
            ttnn_Q,
            ttnn_K,
            ttnn_V,
            compute_kernel_config=compute_kernel_config,
        )
