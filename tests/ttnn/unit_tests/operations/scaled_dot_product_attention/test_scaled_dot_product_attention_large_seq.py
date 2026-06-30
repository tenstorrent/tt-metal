# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 6 — Large sequence causal attention tests.

Tests that the causal block skip optimization (skipping fully-future KV-blocks
above the causal diagonal) produces correct results for very large sequence
lengths without timing out.

The target cell test_sdpa_tt_large_seq__nightly[1-8-1-131072-128-k128-q128-bf16]
lives in eval/golden_tests/scaled_dot_product_attention/test_translated.py.
These tests exercise the same causal block skip logic at smaller scales for
fast verification, plus a large-scale test that would time out without the skip.
"""
import math
import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


# Causal self-attention reference implementation
def pytorch_causal_sdpa(Q, K, V, scale=None):
    if scale is not None:
        # Use scale via the scaled_dot_product_attention API
        # torch doesn't directly support custom scale in older versions,
        # so we compute manually
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    else:
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.shape[-1])
    # Causal mask: keep lower triangle (col <= row), mask upper (col > row)
    S_q, S_kv = Q.shape[-2], K.shape[-2]
    causal_mask = torch.tril(torch.ones(S_q, S_kv, dtype=torch.bool), diagonal=0)
    scores = scores.masked_fill(~causal_mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, V)


@pytest.mark.parametrize("S", [1024, 4096, 8192], ids=["S1024", "S4096", "S8192"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["bf16", "bfp8"])
def test_causal_block_skip_correctness(device, S, dtype):
    """Verify causal block skip produces correct results for large S.

    Without the skip, S=8192 would be slow; with it, fully-future KV-blocks
    are skipped entirely. The output must match the PyTorch reference.
    """
    torch.manual_seed(42)
    B, H, D = 1, 4, 128
    Q = torch.randn(B, H, S, D, dtype=torch.float32)
    K = torch.randn(B, H, S, D, dtype=torch.float32)
    V = torch.randn(B, H, S, D, dtype=torch.float32)

    expected = pytorch_causal_sdpa(Q, K, V)

    tt_Q = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_K = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_V = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    compute_kernel_config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
    )

    tt_out = scaled_dot_product_attention(tt_Q, tt_K, tt_V, is_causal=True, compute_kernel_config=compute_kernel_config)
    result = ttnn.to_torch(tt_out)

    assert_with_pcc(result.float(), expected.float(), pcc=0.99)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["bf16", "bfp8"])
def test_causal_vs_non_causal_large(device, dtype):
    """Causal and non-causal attention should produce different outputs for
    large S — this verifies the causal block skip doesn't accidentally
    skip too many blocks (which would make causal look like non-causal).
    """
    torch.manual_seed(42)
    B, H, S, D = 1, 2, 1024, 128
    Q = torch.randn(B, H, S, D, dtype=torch.float32)
    K = torch.randn(B, H, S, D, dtype=torch.float32)
    V = torch.randn(B, H, S, D, dtype=torch.float32)

    tt_Q = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_K = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_V = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    compute_kernel_config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
    )

    tt_causal = scaled_dot_product_attention(
        tt_Q, tt_K, tt_V, is_causal=True, compute_kernel_config=compute_kernel_config
    )
    tt_non_causal = scaled_dot_product_attention(
        tt_Q, tt_K, tt_V, is_causal=False, compute_kernel_config=compute_kernel_config
    )

    causal_result = ttnn.to_torch(tt_causal).float()
    non_causal_result = ttnn.to_torch(tt_non_causal).float()

    # They must differ — causal masking changes the output
    max_diff = (causal_result - non_causal_result).abs().max().item()
    assert max_diff > 0.01, f"Causal and non-causal outputs too similar (max_diff={max_diff})"


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
def test_large_causal_seq_completes(device, dtype):
    """Large sequence causal attention must complete without timeout.

    S=32768 is large enough to timeout without the causal block skip,
    but small enough to run in a reasonable test time. With the skip,
    only ~half the KV-blocks are processed.
    """
    torch.manual_seed(42)
    B, H, S, D = 1, 8, 32768, 128
    Q = torch.randn(B, H, S, D, dtype=torch.float32)
    K = torch.randn(B, H, S, D, dtype=torch.float32)
    V = torch.randn(B, H, S, D, dtype=torch.float32)

    expected = pytorch_causal_sdpa(Q, K, V)

    tt_Q = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_K = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_V = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    compute_kernel_config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
    )

    tt_out = scaled_dot_product_attention(tt_Q, tt_K, tt_V, is_causal=True, compute_kernel_config=compute_kernel_config)
    result = ttnn.to_torch(tt_out)

    assert_with_pcc(result.float(), expected.float(), pcc=0.99)


@pytest.mark.parametrize("S", [256, 512, 1024], ids=["S256", "S512", "S1024"])
def test_causal_block_skip_small_shapes(device, S):
    """Causal block skip must produce exact results for small shapes too.

    These shapes have few KV-blocks, so the skip logic's boundary conditions
    (diagonal-straddling blocks) are exercised.
    """
    torch.manual_seed(42)
    B, H, D = 1, 1, 64
    Q = torch.randn(B, H, S, D, dtype=torch.float32)
    K = torch.randn(B, H, S, D, dtype=torch.float32)
    V = torch.randn(B, H, S, D, dtype=torch.float32)

    expected = pytorch_causal_sdpa(Q, K, V)

    tt_Q = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_K = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_V = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    compute_kernel_config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
    )

    tt_out = scaled_dot_product_attention(tt_Q, tt_K, tt_V, is_causal=True, compute_kernel_config=compute_kernel_config)
    result = ttnn.to_torch(tt_out)

    assert_with_pcc(result.float(), expected.float(), pcc=0.99)


def test_causal_single_tile_diagonal(device):
    """Single-tile causal attention (S=32): the single KV-block is the diagonal
    block itself — the causal skip must NOT skip it.
    """
    torch.manual_seed(42)
    B, H, S, D = 1, 1, 32, 32
    Q = torch.randn(B, H, S, D, dtype=torch.float32)
    K = torch.randn(B, H, S, D, dtype=torch.float32)
    V = torch.randn(B, H, S, D, dtype=torch.float32)

    expected = pytorch_causal_sdpa(Q, K, V)

    tt_Q = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_K = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_V = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    compute_kernel_config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
    )

    tt_out = scaled_dot_product_attention(tt_Q, tt_K, tt_V, is_causal=True, compute_kernel_config=compute_kernel_config)
    result = ttnn.to_torch(tt_out)

    assert_with_pcc(result.float(), expected.float(), pcc=0.99)
