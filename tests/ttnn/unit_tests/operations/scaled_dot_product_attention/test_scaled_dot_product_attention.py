# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for scaled_dot_product_attention (Flash Attention).

Immutable spec — the implementer is told not to modify this file.

Tests the Flash Attention operation against torch.nn.functional.
scaled_dot_product_attention for multiple shapes, mask modes, and
scale modes. Uses PCC tolerances keyed by dtype (same thresholds as
the golden suite).
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


# ── PCC tolerances keyed by dtype (same as golden suite) ───────────────

PCC_TOLERANCES = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.995,
    ttnn.bfloat8_b: 0.99,
}


# ── Test shapes ────────────────────────────────────────────────────────
# Minimum 4: single-tile, multi-tile, non-square, multi-batch.
# Plus long-context and cross-attention shapes per requirements.

TEST_SHAPES = [
    # (B, H, S_q, D, S_kv)  — S_kv == S_q for self-attention unless noted
    # Single-tile
    pytest.param(1, 1, 32, 32, 32, id="single_tile_32x32"),
    # Multi-tile (sequence length scaling)
    pytest.param(1, 1, 128, 64, 128, id="multi_tile_128x64"),
    pytest.param(1, 1, 256, 64, 256, id="multi_tile_256x64"),
    # Non-square (D != S)
    pytest.param(1, 1, 128, 128, 128, id="non_square_128x128"),
    pytest.param(1, 1, 128, 32, 128, id="non_square_128x32"),
    # Multi-batch + multi-head
    pytest.param(2, 4, 128, 64, 128, id="multi_batch_2x4_128x64"),
    pytest.param(4, 8, 128, 64, 128, id="multi_batch_4x8_128x64"),
    # Large model
    pytest.param(1, 32, 128, 128, 128, id="large_model_32h_128x128"),
    # Long-context
    pytest.param(1, 1, 512, 64, 512, id="long_context_512"),
    pytest.param(1, 1, 1024, 64, 1024, id="long_context_1024"),
    pytest.param(1, 1, 2048, 64, 2048, id="long_context_2048"),
    # Cross-attention (S_q != S_kv)
    pytest.param(1, 4, 64, 64, 128, id="cross_attn_64_to_128"),
    pytest.param(1, 4, 128, 64, 64, id="cross_attn_128_to_64"),
    pytest.param(1, 12, 128, 64, 512, id="cross_attn_128_to_512"),
]


@pytest.mark.parametrize(
    "B, H, S_q, D, S_kv",
    TEST_SHAPES,
)
def test_scaled_dot_product_attention_no_mask(device, B, H, S_q, D, S_kv):
    """Test SDPA with no mask (mask_mode=none, scale_mode=auto)."""
    torch.manual_seed(42)
    dtype = torch.bfloat16

    q = torch.randn(B, H, S_q, D, dtype=dtype)
    k = torch.randn(B, H, S_kv, D, dtype=dtype)
    v = torch.randn(B, H, S_kv, D, dtype=dtype)

    # PyTorch reference
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    # TTNN
    q_t = ttnn.from_torch(
        q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    k_t = ttnn.from_torch(
        k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    v_t = ttnn.from_torch(
        v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    output = scaled_dot_product_attention(q_t, k_t, v_t)
    output_torch = ttnn.to_torch(output)

    pcc_threshold = PCC_TOLERANCES[ttnn.bfloat16]
    assert_with_pcc(ref, output_torch, pcc=pcc_threshold)


@pytest.mark.parametrize(
    "B, H, S_q, D, S_kv",
    [
        pytest.param(1, 1, 128, 64, 128, id="explicit_scale_128x64"),
        pytest.param(1, 4, 128, 64, 128, id="explicit_scale_multihead"),
        pytest.param(1, 1, 256, 64, 256, id="explicit_scale_256x64"),
        pytest.param(2, 4, 128, 64, 128, id="explicit_scale_multibatch"),
    ],
)
def test_scaled_dot_product_attention_explicit_scale(device, B, H, S_q, D, S_kv):
    """Test SDPA with explicit scale (scale_mode=explicit, mask_mode=none)."""
    torch.manual_seed(42)
    dtype = torch.bfloat16
    scale = 0.125

    q = torch.randn(B, H, S_q, D, dtype=dtype)
    k = torch.randn(B, H, S_kv, D, dtype=dtype)
    v = torch.randn(B, H, S_kv, D, dtype=dtype)

    # PyTorch reference
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)

    # TTNN
    q_t = ttnn.from_torch(
        q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    k_t = ttnn.from_torch(
        k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    v_t = ttnn.from_torch(
        v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    output = scaled_dot_product_attention(q_t, k_t, v_t, scale=scale)
    output_torch = ttnn.to_torch(output)

    pcc_threshold = PCC_TOLERANCES[ttnn.bfloat16]
    assert_with_pcc(ref, output_torch, pcc=pcc_threshold)


@pytest.mark.parametrize(
    "B, H, S_q, D, S_kv",
    [
        pytest.param(1, 1, 128, 64, 128, id="custom_mask_128x64"),
        pytest.param(1, 4, 128, 64, 128, id="custom_mask_multihead"),
        pytest.param(1, 1, 256, 64, 256, id="custom_mask_256x64"),
        pytest.param(1, 4, 64, 64, 128, id="custom_mask_cross_attn"),
    ],
)
def test_scaled_dot_product_attention_custom_mask(device, B, H, S_q, D, S_kv):
    """Test SDPA with a custom additive attn_mask (mask_mode=custom)."""
    torch.manual_seed(42)
    dtype = torch.bfloat16

    q = torch.randn(B, H, S_q, D, dtype=dtype)
    k = torch.randn(B, H, S_kv, D, dtype=dtype)
    v = torch.randn(B, H, S_kv, D, dtype=dtype)

    # Create a custom additive mask: 0 = attend, -inf = mask out
    # Use a triangular mask (same pattern as causal, but via the additive mask path)
    mask = torch.zeros(B, 1, S_q, S_kv, dtype=dtype)
    for i in range(S_q):
        for j in range(S_kv):
            if j > i + (S_kv - S_q):
                mask[:, :, i, j] = float("-inf")

    # PyTorch reference
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    # TTNN
    q_t = ttnn.from_torch(
        q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    k_t = ttnn.from_torch(
        k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    v_t = ttnn.from_torch(
        v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    mask_t = ttnn.from_torch(
        mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    output = scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=mask_t)
    output_torch = ttnn.to_torch(output)

    pcc_threshold = PCC_TOLERANCES[ttnn.bfloat16]
    assert_with_pcc(ref, output_torch, pcc=pcc_threshold)


@pytest.mark.parametrize(
    "B, H, S_q, D, S_kv",
    [
        pytest.param(1, 1, 128, 64, 128, id="custom_mask_per_head"),
        pytest.param(1, 4, 128, 64, 128, id="custom_mask_per_head_multihead"),
    ],
)
def test_scaled_dot_product_attention_per_head_mask(device, B, H, S_q, D, S_kv):
    """Test SDPA with a per-head (B, H, S_q, S_kv) additive attn_mask."""
    torch.manual_seed(42)
    dtype = torch.bfloat16

    q = torch.randn(B, H, S_q, D, dtype=dtype)
    k = torch.randn(B, H, S_kv, D, dtype=dtype)
    v = torch.randn(B, H, S_kv, D, dtype=dtype)

    # Per-head mask: different masking pattern per head
    mask = torch.zeros(B, H, S_q, S_kv, dtype=dtype)
    for h in range(H):
        for i in range(S_q):
            for j in range(S_kv):
                if j > i + h * 4:
                    mask[:, h, i, j] = float("-inf")

    # PyTorch reference
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    # TTNN
    q_t = ttnn.from_torch(
        q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    k_t = ttnn.from_torch(
        k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    v_t = ttnn.from_torch(
        v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    mask_t = ttnn.from_torch(
        mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    output = scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=mask_t)
    output_torch = ttnn.to_torch(output)

    pcc_threshold = PCC_TOLERANCES[ttnn.bfloat16]
    assert_with_pcc(ref, output_torch, pcc=pcc_threshold)


@pytest.mark.parametrize(
    "B, H, S_q, D, S_kv",
    [
        pytest.param(1, 1, 128, 64, 128, id="causal_explicit_scale"),
        pytest.param(1, 4, 128, 64, 128, id="causal_explicit_scale_multihead"),
        pytest.param(1, 1, 256, 64, 256, id="causal_explicit_scale_256"),
    ],
)
def test_scaled_dot_product_attention_causal_with_explicit_scale(device, B, H, S_q, D, S_kv):
    """Test SDPA with causal mask (via custom additive mask) + explicit scale.

    Phase 0 does not support is_causal=True natively. This test uses a
    custom triangular additive mask to exercise the causal pattern through
    the mask_mode=custom path, combined with an explicit scale.
    """
    torch.manual_seed(42)
    dtype = torch.bfloat16
    scale = 0.1

    q = torch.randn(B, H, S_q, D, dtype=dtype)
    k = torch.randn(B, H, S_kv, D, dtype=dtype)
    v = torch.randn(B, H, S_kv, D, dtype=dtype)

    # Causal mask via additive mask (0 on/below diagonal, -inf above)
    mask = torch.zeros(1, 1, S_q, S_kv, dtype=dtype)
    for i in range(S_q):
        for j in range(S_kv):
            if j > i:
                mask[0, 0, i, j] = float("-inf")

    # PyTorch reference (use is_causal for comparison)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)

    # TTNN (use custom mask to emulate causal in Phase 0)
    q_t = ttnn.from_torch(
        q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    k_t = ttnn.from_torch(
        k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    v_t = ttnn.from_torch(
        v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    mask_t = ttnn.from_torch(
        mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    output = scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=mask_t, scale=scale)
    output_torch = ttnn.to_torch(output)

    pcc_threshold = PCC_TOLERANCES[ttnn.bfloat16]
    assert_with_pcc(ref, output_torch, pcc=pcc_threshold)


@pytest.mark.parametrize(
    "B, H, S_q, D, S_kv",
    [
        pytest.param(1, 1, 4096, 64, 4096, id="long_context_4096"),
        pytest.param(1, 1, 2048, 64, 2048, id="long_context_2048_large"),
    ],
)
def test_scaled_dot_product_attention_long_context(device, B, H, S_q, D, S_kv):
    """Test SDPA with long-context sequences (primary use case)."""
    torch.manual_seed(42)
    dtype = torch.bfloat16

    q = torch.randn(B, H, S_q, D, dtype=dtype)
    k = torch.randn(B, H, S_kv, D, dtype=dtype)
    v = torch.randn(B, H, S_kv, D, dtype=dtype)

    # PyTorch reference
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    # TTNN
    q_t = ttnn.from_torch(
        q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    k_t = ttnn.from_torch(
        k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    v_t = ttnn.from_torch(
        v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    output = scaled_dot_product_attention(q_t, k_t, v_t)
    output_torch = ttnn.to_torch(output)

    pcc_threshold = PCC_TOLERANCES[ttnn.bfloat16]
    assert_with_pcc(ref, output_torch, pcc=pcc_threshold)
