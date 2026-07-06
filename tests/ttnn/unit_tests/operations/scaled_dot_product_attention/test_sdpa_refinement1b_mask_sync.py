# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 1b — Multi-block mask sync tests.

Tests that exercise the mask application path (mask_mode=custom) across
multiple KV blocks. These tests would have caught the double-pop bug in
cb_attn_mask that caused the full golden suite to hang.

The root cause was: BinaryFpu<cb_scores, cb_attn_mask, Add> with default
InputLifecycle::Streaming already pops all mask tiles internally, but the
kernel also did a manual cb_pop_front(cb_attn_mask, B_q*B_kv) — a double-pop
that corrupted the CB read pointer and deadlocked the reader on the next
KV block iteration.
"""

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


PCC_THRESHOLD = 0.995


def _make_causal_mask(B, S_q, S_kv, dtype=torch.bfloat16):
    """Triangular additive mask: 0 on/below diagonal, -inf above."""
    mask = torch.zeros(B, 1, S_q, S_kv, dtype=dtype)
    for i in range(S_q):
        for j in range(S_kv):
            if j > i + (S_kv - S_q):
                mask[:, :, i, j] = float("-inf")
    return mask


@pytest.mark.parametrize(
    "B, H, S_q, D, S_kv",
    [
        # Multi-KV-block mask: S_kv > 32 means multiple KV blocks with mask
        pytest.param(1, 1, 128, 64, 128, id="mask_multikv_128x64"),
        pytest.param(1, 1, 256, 64, 256, id="mask_multikv_256x64"),
        pytest.param(1, 1, 512, 64, 512, id="mask_multikv_512x64"),
        pytest.param(1, 1, 1024, 64, 1024, id="mask_multikv_1024x64"),
        pytest.param(1, 1, 2048, 64, 2048, id="mask_multikv_2048x64"),
        # Multi-head with mask (exercises multiple Q blocks + mask)
        pytest.param(1, 4, 128, 64, 128, id="mask_multihead_4_128x64"),
        pytest.param(1, 8, 256, 64, 256, id="mask_multihead_8_256x64"),
        pytest.param(1, 12, 128, 64, 128, id="mask_multihead_12_128x64"),
        # Multi-batch with mask
        pytest.param(2, 4, 128, 64, 128, id="mask_multibatch_2x4"),
        pytest.param(4, 8, 128, 64, 128, id="mask_multibatch_4x8"),
        # Cross-attention with mask (S_q != S_kv)
        pytest.param(1, 4, 64, 64, 128, id="mask_cross_64_to_128"),
        pytest.param(1, 4, 128, 64, 64, id="mask_cross_128_to_64"),
        # Large D with mask
        pytest.param(1, 1, 128, 128, 128, id="mask_large_d_128x128"),
        pytest.param(1, 1, 128, 256, 128, id="mask_large_d_128x256"),
        # Per-head mask
        pytest.param(1, 4, 128, 64, 128, id="mask_per_head_4_128x64"),
    ],
)
def test_mask_multiblock_sync(device, B, H, S_q, D, S_kv):
    """Test that mask application works correctly across multiple KV blocks.

    This is the key test that catches the double-pop bug: the BinaryFpu<Add>
    chain with Streaming lifecycle pops mask tiles internally, so a manual
    cb_pop_front after the chain is a double-pop that corrupts CB state.
    """
    torch.manual_seed(42)
    dtype = torch.bfloat16

    q = torch.randn(B, H, S_q, D, dtype=dtype)
    k = torch.randn(B, H, S_kv, D, dtype=dtype)
    v = torch.randn(B, H, S_kv, D, dtype=dtype)

    # Create custom additive mask
    mask = _make_causal_mask(B, S_q, S_kv, dtype)

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

    assert_with_pcc(ref, output_torch, pcc=PCC_THRESHOLD)


@pytest.mark.parametrize(
    "B, H, S_q, D, S_kv",
    [
        # Mask + explicit scale across multi-block
        pytest.param(1, 1, 128, 64, 128, id="mask_explicit_scale_128x64"),
        pytest.param(1, 4, 256, 64, 256, id="mask_explicit_scale_multihead"),
        pytest.param(2, 4, 128, 64, 128, id="mask_explicit_scale_multibatch"),
    ],
)
def test_mask_with_explicit_scale_multiblock(device, B, H, S_q, D, S_kv):
    """Test mask + explicit scale across multiple KV blocks."""
    torch.manual_seed(42)
    dtype = torch.bfloat16
    scale = 0.125

    q = torch.randn(B, H, S_q, D, dtype=dtype)
    k = torch.randn(B, H, S_kv, D, dtype=dtype)
    v = torch.randn(B, H, S_kv, D, dtype=dtype)
    mask = _make_causal_mask(B, S_q, S_kv, dtype)

    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)

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

    assert_with_pcc(ref, output_torch, pcc=PCC_THRESHOLD)


def test_mask_no_hang_full_suite(device):
    """Regression test: running multiple mask cells in sequence must not hang.

    The double-pop bug would corrupt CB state, causing the reader to deadlock
    on cb_reserve_back for cb_attn_mask on the second KV block iteration.
    This test runs several mask shapes sequentially to verify no state leak.
    """
    torch.manual_seed(42)
    dtype = torch.bfloat16

    shapes = [
        (1, 1, 128, 64, 128),
        (1, 1, 256, 64, 256),
        (1, 4, 128, 64, 128),
        (1, 1, 512, 64, 512),
    ]

    for B, H, S_q, D, S_kv in shapes:
        q = torch.randn(B, H, S_q, D, dtype=dtype)
        k = torch.randn(B, H, S_kv, D, dtype=dtype)
        v = torch.randn(B, H, S_kv, D, dtype=dtype)
        mask = _make_causal_mask(B, S_q, S_kv, dtype)

        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)

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

        assert_with_pcc(ref, output_torch, pcc=PCC_THRESHOLD)
