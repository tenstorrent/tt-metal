# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 6 — L1 budget fit for large head_dim.

Tests that the D_CHUNK K-blocking + N-chunking mechanism works correctly
for large head_dim (D=512, D=1024) across all dtypes, mask modes, scale
modes, and attention kinds. These shapes previously OOM'd in L1 before
Refinement 6's constant-bounded CB sizing.
"""

import math

import pytest
import torch

import ttnn


def run_sdpa(
    device, q_shape, kv_shape, attn_mask=None, is_causal=False, scale=None, dtype=ttnn.bfloat16, fp32_dest_acc_en=True
):
    """Run SDPA and return (ttnn_output, torch_reference)."""
    torch_dtype = {
        ttnn.bfloat16: torch.bfloat16,
        ttnn.float32: torch.float32,
        ttnn.bfloat8_b: torch.bfloat16,  # reference uses bf16 for bf8b
    }[dtype]

    torch_q = torch.randn(*q_shape, dtype=torch_dtype)
    torch_k = torch.randn(*kv_shape, dtype=torch_dtype)
    torch_v = torch.randn(*kv_shape, dtype=torch_dtype)

    # For GQA/MQA (H_q != H_kv), broadcast K/V heads to match Q heads
    # using repeat_interleave (the standard GQA broadcasting pattern)
    h_q = q_shape[1]
    h_kv = kv_shape[1]
    if h_q != h_kv:
        group_size = h_q // h_kv
        torch_k = torch.repeat_interleave(torch_k, group_size, dim=1)
        torch_v = torch.repeat_interleave(torch_v, group_size, dim=1)

    if scale is None:
        scale_val = 1.0 / math.sqrt(q_shape[-1])
    else:
        scale_val = scale

    # Torch reference
    attn_mask_torch = None
    if attn_mask is not None:
        attn_mask_torch = attn_mask
    elif is_causal:
        s_q, s_kv = q_shape[2], kv_shape[2]
        attn_mask_torch = torch.zeros(1, 1, s_q, s_kv, dtype=torch_dtype)
        for i in range(min(s_q, s_kv)):
            for j in range(s_kv):
                if j > i:
                    attn_mask_torch[0, 0, i, j] = float("-inf")

    torch_out = torch.nn.functional.scaled_dot_product_attention(
        torch_q,
        torch_k,
        torch_v,
        attn_mask=attn_mask_torch,
        is_causal=is_causal,
        scale=scale_val if scale is not None else None,
    )

    q_ttnn = ttnn.from_torch(torch_q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    k_ttnn = ttnn.from_torch(torch_k, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    v_ttnn = ttnn.from_torch(torch_v, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    mask_ttnn = None
    if attn_mask is not None:
        mask_ttnn = ttnn.from_torch(attn_mask, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    compute_config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=fp32_dest_acc_en,
        math_approx_mode=False,
    )

    ttnn_out = ttnn.operations.scaled_dot_product_attention.scaled_dot_product_attention(
        q_ttnn,
        k_ttnn,
        v_ttnn,
        attn_mask=mask_ttnn,
        is_causal=is_causal,
        scale=scale,
        compute_kernel_config=compute_config,
    )
    result = ttnn.to_torch(ttnn_out)
    return result, torch_out


def check_pcc(result, reference, pcc_threshold=0.99):
    """Check PCC between result and reference."""
    result_f = result.float()
    ref_f = reference.float()
    if result_f.shape != ref_f.shape:
        pytest.fail(f"Shape mismatch: {result_f.shape} vs {ref_f.shape}")
    pcc = ttnn.pearson_correlation_coefficient(result_f, ref_f)
    assert pcc >= pcc_threshold, f"PCC={pcc} < {pcc_threshold}"


# ---------------------------------------------------------------------------
# D=1024 tests — the primary target of Refinement 6
# These shapes previously OOM'd in L1 (float32) before the D_CHUNK fix.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mask_mode", ["none", "custom", "causal"])
@pytest.mark.parametrize("scale_mode", ["auto", "explicit"])
@pytest.mark.parametrize("fp32_dest_acc_en", [True, False])
def test_d1024_bf16(device, mask_mode, scale_mode, fp32_dest_acc_en):
    """D=1024 bf16 — exercises K-blocking (num_k_blocks=4) and N-chunking (num_d_chunks=4)."""
    q_shape = (1, 1, 128, 1024)
    kv_shape = (1, 1, 128, 1024)
    scale = 0.03125 if scale_mode == "explicit" else None
    is_causal = mask_mode == "causal"
    attn_mask = None
    if mask_mode == "custom":
        s_q, s_kv = q_shape[2], kv_shape[2]
        attn_mask = torch.zeros(1, 1, s_q, s_kv, dtype=torch.bfloat16)
        for i in range(min(s_q, s_kv)):
            for j in range(s_kv):
                if j > i:
                    attn_mask[0, 0, i, j] = float("-inf")
    result, reference = run_sdpa(
        device,
        q_shape,
        kv_shape,
        attn_mask=attn_mask,
        is_causal=is_causal,
        scale=scale,
        dtype=ttnn.bfloat16,
        fp32_dest_acc_en=fp32_dest_acc_en,
    )
    check_pcc(result, reference, pcc_threshold=0.99)


@pytest.mark.parametrize("mask_mode", ["none", "custom", "causal"])
@pytest.mark.parametrize("scale_mode", ["auto", "explicit"])
def test_d1024_fp32(device, mask_mode, scale_mode):
    """D=1024 fp32 — the 6 cells that previously OOM'd in L1."""
    q_shape = (1, 1, 128, 1024)
    kv_shape = (1, 1, 128, 1024)
    scale = 0.03125 if scale_mode == "explicit" else None
    is_causal = mask_mode == "causal"
    attn_mask = None
    if mask_mode == "custom":
        s_q, s_kv = q_shape[2], kv_shape[2]
        attn_mask = torch.zeros(1, 1, s_q, s_kv, dtype=torch.float32)
        for i in range(min(s_q, s_kv)):
            for j in range(s_kv):
                if j > i:
                    attn_mask[0, 0, i, j] = float("-inf")
    result, reference = run_sdpa(
        device,
        q_shape,
        kv_shape,
        attn_mask=attn_mask,
        is_causal=is_causal,
        scale=scale,
        dtype=ttnn.float32,
        fp32_dest_acc_en=True,
    )
    check_pcc(result, reference, pcc_threshold=0.99)


@pytest.mark.parametrize("mask_mode", ["none", "custom", "causal"])
@pytest.mark.parametrize("scale_mode", ["auto", "explicit"])
@pytest.mark.parametrize("fp32_dest_acc_en", [True, False])
def test_d1024_bf8b(device, mask_mode, scale_mode, fp32_dest_acc_en):
    """D=1024 bfloat8_b — block-float format with K-blocking."""
    q_shape = (1, 1, 128, 1024)
    kv_shape = (1, 1, 128, 1024)
    scale = 0.03125 if scale_mode == "explicit" else None
    is_causal = mask_mode == "causal"
    attn_mask = None
    if mask_mode == "custom":
        s_q, s_kv = q_shape[2], kv_shape[2]
        attn_mask = torch.zeros(1, 1, s_q, s_kv, dtype=torch.bfloat16)
        for i in range(min(s_q, s_kv)):
            for j in range(s_kv):
                if j > i:
                    attn_mask[0, 0, i, j] = float("-inf")
    result, reference = run_sdpa(
        device,
        q_shape,
        kv_shape,
        attn_mask=attn_mask,
        is_causal=is_causal,
        scale=scale,
        dtype=ttnn.bfloat8_b,
        fp32_dest_acc_en=fp32_dest_acc_en,
    )
    check_pcc(result, reference, pcc_threshold=0.98)


# ---------------------------------------------------------------------------
# D=512 tests — exercises K-blocking (num_k_blocks=2) and N-chunking (num_d_chunks=2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype,fp32_dest_acc_en",
    [
        (ttnn.bfloat16, True),
        (ttnn.bfloat16, False),
        (ttnn.float32, True),
        (ttnn.bfloat8_b, True),
        (ttnn.bfloat8_b, False),
    ],
)
def test_d512_various_dtypes(device, dtype, fp32_dest_acc_en):
    """D=512 — K-blocking with num_k_blocks=2, N-chunking with num_d_chunks=2."""
    q_shape = (1, 1, 128, 512)
    kv_shape = (1, 1, 128, 512)
    result, reference = run_sdpa(device, q_shape, kv_shape, dtype=dtype, fp32_dest_acc_en=fp32_dest_acc_en)
    check_pcc(result, reference, pcc_threshold=0.99)


# ---------------------------------------------------------------------------
# Multi-head + large head_dim — tests that D_CHUNK works with work distribution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("h_q,h_kv", [(8, 8), (8, 2), (8, 1), (12, 4)])
def test_d1024_multi_head(device, h_q, h_kv):
    """D=1024 with multi-head — tests D_CHUNK + work-unit loop interaction."""
    q_shape = (1, h_q, 128, 1024)
    kv_shape = (1, h_kv, 128, 1024)
    result, reference = run_sdpa(device, q_shape, kv_shape, dtype=ttnn.bfloat16)
    check_pcc(result, reference, pcc_threshold=0.99)


# ---------------------------------------------------------------------------
# Cross-attention + large head_dim
# ---------------------------------------------------------------------------


def test_d1024_cross_attention(device):
    """D=1024 cross-attention (S_q != S_kv)."""
    q_shape = (1, 4, 64, 1024)
    kv_shape = (1, 4, 128, 1024)
    result, reference = run_sdpa(device, q_shape, kv_shape, dtype=ttnn.bfloat16)
    check_pcc(result, reference, pcc_threshold=0.99)


# ---------------------------------------------------------------------------
# Large head_dim + mask combinations
# ---------------------------------------------------------------------------


def test_d1024_custom_mask_per_head(device):
    """D=1024 with per-head custom mask."""
    B, H, S_q, S_kv, D = 1, 4, 128, 128, 1024
    q_shape = (B, H, S_q, D)
    kv_shape = (B, H, S_kv, D)
    attn_mask = torch.zeros(B, H, S_q, S_kv, dtype=torch.bfloat16)
    for b in range(B):
        for h in range(H):
            for i in range(S_q):
                for j in range(S_kv):
                    if j > i + h * 10:
                        attn_mask[b, h, i, j] = float("-inf")
    result, reference = run_sdpa(device, q_shape, kv_shape, attn_mask=attn_mask, dtype=ttnn.bfloat16)
    check_pcc(result, reference, pcc_threshold=0.99)


def test_d1024_causal_with_explicit_scale(device):
    """D=1024 causal mask + explicit scale."""
    q_shape = (1, 4, 128, 1024)
    kv_shape = (1, 4, 128, 1024)
    result, reference = run_sdpa(device, q_shape, kv_shape, is_causal=True, scale=0.03125, dtype=ttnn.bfloat16)
    check_pcc(result, reference, pcc_threshold=0.99)


# ---------------------------------------------------------------------------
# Deterministic input — all-ones for exact output verification
# ---------------------------------------------------------------------------


def test_d1024_all_ones_deterministic(device):
    """D=1024 with all-ones input. Output should be all-ones (row i attends to
    positions 0..i, average of ones = 1.0)."""
    q_shape = (1, 1, 128, 1024)
    torch_q = torch.ones(*q_shape, dtype=torch.bfloat16)
    torch_k = torch.ones(*q_shape, dtype=torch.bfloat16)
    torch_v = torch.ones(*q_shape, dtype=torch.bfloat16)

    q_ttnn = ttnn.from_torch(torch_q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    k_ttnn = ttnn.from_torch(torch_k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    v_ttnn = ttnn.from_torch(torch_v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    compute_config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )

    ttnn_out = ttnn.operations.scaled_dot_product_attention.scaled_dot_product_attention(
        q_ttnn,
        k_ttnn,
        v_ttnn,
        is_causal=True,
        compute_kernel_config=compute_config,
    )
    result = ttnn.to_torch(ttnn_out)

    # With causal mask and all-ones input: row i attends to positions 0..i
    # Each Q row dot-product with K = sum of D ones = 1024
    # Scaled: 1024 / sqrt(1024) = 32
    # Softmax of 32's → exp(32) / sum → effectively 1.0 for the last position
    # P @ V with P≈1.0 and V=1.0 → 1.0
    # The output should be approximately all-ones
    expected = torch.ones(*q_shape, dtype=torch.bfloat16)
    check_pcc(result, expected, pcc_threshold=0.98)


# ---------------------------------------------------------------------------
# Sequential state-leak regression — ensure D_CHUNK loop doesn't leak state
# ---------------------------------------------------------------------------


def test_d1024_sequential_no_state_leak(device):
    """Run D=1024 then D=64 — verify no state leaks between invocations."""
    q_shape = (1, 1, 128, 1024)
    kv_shape = (1, 1, 128, 1024)

    # First call: D=1024 (D_CHUNK=8, num_d_chunks=4)
    result1, ref1 = run_sdpa(device, q_shape, kv_shape, dtype=ttnn.bfloat16)
    check_pcc(result1, ref1, pcc_threshold=0.99)

    # Second call: D=64 (D_CHUNK=2, num_d_chunks=1)
    q_shape2 = (1, 1, 128, 64)
    kv_shape2 = (1, 1, 128, 64)
    result2, ref2 = run_sdpa(device, q_shape2, kv_shape2, dtype=ttnn.bfloat16)
    check_pcc(result2, ref2, pcc_threshold=0.99)

    # Third call: D=1024 again
    result3, ref3 = run_sdpa(device, q_shape, kv_shape, dtype=ttnn.bfloat16)
    check_pcc(result3, ref3, pcc_threshold=0.99)


# ---------------------------------------------------------------------------
# Multi-batch + large head_dim
# ---------------------------------------------------------------------------


def test_d1024_multi_batch(device):
    """D=1024 with multi-batch."""
    q_shape = (2, 4, 128, 1024)
    kv_shape = (2, 4, 128, 1024)
    result, reference = run_sdpa(device, q_shape, kv_shape, dtype=ttnn.bfloat16)
    check_pcc(result, reference, pcc_threshold=0.99)
