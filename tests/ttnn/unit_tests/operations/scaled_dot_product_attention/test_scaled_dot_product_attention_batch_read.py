# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 7 — Batch NoC reads + large sequence verification.

Tests that the batch NoC read optimization in the reader kernel produces
correct results across all dtypes, mask modes, and attention kinds, and
that the large sequence test completes within a reasonable time.
"""
import math
import pytest
import torch
import ttnn


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32])
@pytest.mark.parametrize("is_causal", [True, False])
def test_batch_read_correctness(device, dtype, is_causal):
    """Verify batch NoC reads produce correct results for all dtypes and mask modes."""
    B, H, S, D = 1, 4, 128, 64
    torch.manual_seed(42)
    Q = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    K = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    V = torch.randn(B, H, S, D, dtype=torch.bfloat16)

    expected = torch.nn.functional.scaled_dot_product_attention(
        Q.float(), K.float(), V.float(), is_causal=is_causal
    ).to(torch.bfloat16)

    tt_Q = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_K = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_V = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    tt_out = ttnn.operations.scaled_dot_product_attention.scaled_dot_product_attention(
        tt_Q, tt_K, tt_V, is_causal=is_causal
    )
    result = ttnn.to_torch(tt_out)

    pcc = torch.corrcoef(torch.stack([result.float().flatten(), expected.float().flatten()]))[0, 1].item()
    assert pcc > 0.99, f"PCC={pcc} for dtype={dtype}, is_causal={is_causal}"


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("S", [256, 512, 1024])
def test_batch_read_large_seq(device, dtype, S):
    """Verify batch reads work correctly on larger sequences (multi-KV-block)."""
    B, H, D = 1, 4, 64
    torch.manual_seed(42)
    Q = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    K = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    V = torch.randn(B, H, S, D, dtype=torch.bfloat16)

    expected = torch.nn.functional.scaled_dot_product_attention(Q.float(), K.float(), V.float(), is_causal=True).to(
        torch.bfloat16
    )

    tt_Q = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_K = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_V = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    tt_out = ttnn.operations.scaled_dot_product_attention.scaled_dot_product_attention(tt_Q, tt_K, tt_V, is_causal=True)
    result = ttnn.to_torch(tt_out)

    pcc = torch.corrcoef(torch.stack([result.float().flatten(), expected.float().flatten()]))[0, 1].item()
    assert pcc > 0.99, f"PCC={pcc} for dtype={dtype}, S={S}"


def test_batch_read_mqa(device):
    """Verify batch reads work with MQA (multiple Q heads, 1 KV head)."""
    B, H_q, H_kv, S, D = 1, 8, 1, 256, 64
    torch.manual_seed(42)
    Q = torch.randn(B, H_q, S, D, dtype=torch.bfloat16)
    K = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16)
    V = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16)

    # PyTorch reference: broadcast K/V across heads
    K_expanded = K.expand(B, H_q, S, D)
    V_expanded = V.expand(B, H_q, S, D)
    expected = torch.nn.functional.scaled_dot_product_attention(
        Q.float(), K_expanded.float(), V_expanded.float(), is_causal=True
    ).to(torch.bfloat16)

    tt_Q = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_K = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_V = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    tt_out = ttnn.operations.scaled_dot_product_attention.scaled_dot_product_attention(tt_Q, tt_K, tt_V, is_causal=True)
    result = ttnn.to_torch(tt_out)

    pcc = torch.corrcoef(torch.stack([result.float().flatten(), expected.float().flatten()]))[0, 1].item()
    assert pcc > 0.99, f"PCC={pcc} for MQA"


def test_batch_read_custom_mask(device):
    """Verify batch reads work with custom additive mask."""
    B, H, S, D = 1, 4, 128, 64
    torch.manual_seed(42)
    Q = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    K = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    V = torch.randn(B, H, S, D, dtype=torch.bfloat16)

    # Create a random additive mask
    mask = torch.zeros(B, 1, S, S, dtype=torch.bfloat16)
    mask[:, :, :, S // 2 :] = float("-inf")  # mask out second half

    expected = torch.nn.functional.scaled_dot_product_attention(
        Q.float(), K.float(), V.float(), attn_mask=mask.float()
    ).to(torch.bfloat16)

    tt_Q = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_K = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_V = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_mask = ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    tt_out = ttnn.operations.scaled_dot_product_attention.scaled_dot_product_attention(
        tt_Q, tt_K, tt_V, attn_mask=tt_mask
    )
    result = ttnn.to_torch(tt_out)

    pcc = torch.corrcoef(torch.stack([result.float().flatten(), expected.float().flatten()]))[0, 1].item()
    assert pcc > 0.99, f"PCC={pcc} for custom mask"


def test_batch_read_cross_attention(device):
    """Verify batch reads work with cross-attention (S_q != S_kv)."""
    B, H, S_q, S_kv, D = 1, 4, 128, 64, 64
    torch.manual_seed(42)
    Q = torch.randn(B, H, S_q, D, dtype=torch.bfloat16)
    K = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16)
    V = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16)

    expected = torch.nn.functional.scaled_dot_product_attention(Q.float(), K.float(), V.float()).to(torch.bfloat16)

    tt_Q = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_K = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_V = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    tt_out = ttnn.operations.scaled_dot_product_attention.scaled_dot_product_attention(tt_Q, tt_K, tt_V)
    result = ttnn.to_torch(tt_out)

    pcc = torch.corrcoef(torch.stack([result.float().flatten(), expected.float().flatten()]))[0, 1].item()
    assert pcc > 0.99, f"PCC={pcc} for cross-attention"


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_large_seq_completion(device, dtype):
    """Verify large sequence (S=8192) completes without timeout.

    This is a scaled-down version of the translated test_sdpa_tt_large_seq__nightly
    that exercises the same code path (causal + MQA + large S) within the
    default dispatch timeout.
    """
    B, H, nkv, S, D = 1, 8, 1, 8192, 128
    torch.manual_seed(1234)

    Q = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    K = torch.randn(B, nkv, S, D, dtype=torch.bfloat16)
    V = torch.randn(B, nkv, S, D, dtype=torch.bfloat16)

    K_expanded = K.expand(B, H, S, D)
    V_expanded = V.expand(B, H, S, D)
    expected = torch.nn.functional.scaled_dot_product_attention(
        Q.float(), K_expanded.float(), V_expanded.float(), is_causal=True
    ).to(torch.bfloat16)

    compute_kernel_config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
    )

    tt_Q = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_K = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_V = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    tt_out = ttnn.operations.scaled_dot_product_attention.scaled_dot_product_attention(
        tt_Q, tt_K, tt_V, is_causal=True, compute_kernel_config=compute_kernel_config
    )
    result = ttnn.to_torch(tt_out)

    pcc = torch.corrcoef(torch.stack([result.float().flatten(), expected.float().flatten()]))[0, 1].item()
    assert pcc > 0.994, f"PCC={pcc} for dtype={dtype}, S={S}"
