# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 4 — Causal masking tests.

Tests the is_causal=True path where the op generates a triangular
causal mask on-device. Verifies correctness across:
- Shape scaling (single-tile → long-context)
- Multi-head / multi-batch
- GQA / MQA head broadcasting
- Explicit scale + causal
- Cross-attention + causal (EXCLUSION — should raise)
- is_causal + attn_mask (mutual exclusion — should raise ValueError)
- Sequential causal→none (state-leak regression)
- Deterministic all-ones input
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


PCC_THRESHOLD = 0.995


# ── Causal self-attention across shape scaling ──────────────────────


@pytest.mark.parametrize(
    "B, H, S, D",
    [
        (1, 1, 32, 32),  # single tile
        (1, 1, 128, 64),  # multi-tile seq
        (1, 1, 256, 64),  # longer seq
        (1, 1, 512, 64),  # long context
        (1, 1, 1024, 64),  # long context
        (1, 1, 2048, 64),  # very long context
        (1, 1, 128, 128),  # large head dim
        (1, 1, 128, 32),  # small head dim
    ],
)
def test_causal_self_attention(device, B, H, S, D):
    """Causal mask (is_causal=True) across shape scaling."""
    torch.manual_seed(42)
    q = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    k = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    v = torch.randn(B, H, S, D, dtype=torch.bfloat16)

    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

    q_t = ttnn.from_torch(
        q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    k_t = ttnn.from_torch(
        k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    v_t = ttnn.from_torch(
        v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    output = scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True)
    output_torch = ttnn.to_torch(output)

    assert_with_pcc(ref, output_torch, pcc=PCC_THRESHOLD)


# ── Causal with multi-head / multi-batch ────────────────────────────


@pytest.mark.parametrize(
    "B, H, S, D",
    [
        (1, 4, 128, 64),  # multi-head
        (1, 8, 128, 64),  # more heads
        (1, 12, 128, 64),  # BERT-base heads
        (2, 4, 128, 64),  # multi-batch
        (4, 8, 128, 64),  # large batch
        (1, 32, 128, 128),  # large model
    ],
)
def test_causal_multihead_multibatch(device, B, H, S, D):
    """Causal mask with multi-head and multi-batch configurations."""
    torch.manual_seed(42)
    q = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    k = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    v = torch.randn(B, H, S, D, dtype=torch.bfloat16)

    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

    q_t = ttnn.from_torch(
        q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    k_t = ttnn.from_torch(
        k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    v_t = ttnn.from_torch(
        v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    output = scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True)
    output_torch = ttnn.to_torch(output)

    assert_with_pcc(ref, output_torch, pcc=PCC_THRESHOLD)


# ── Causal with explicit scale ──────────────────────────────────────


@pytest.mark.parametrize(
    "B, H, S, D, scale",
    [
        (1, 1, 128, 64, 0.125),
        (1, 4, 128, 64, 0.1),
        (1, 1, 256, 64, 0.125),
        (2, 4, 128, 64, 0.125),
    ],
)
def test_causal_explicit_scale(device, B, H, S, D, scale):
    """Causal mask with explicit scale factor."""
    torch.manual_seed(42)
    q = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    k = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    v = torch.randn(B, H, S, D, dtype=torch.bfloat16)

    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)

    q_t = ttnn.from_torch(
        q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    k_t = ttnn.from_torch(
        k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    v_t = ttnn.from_torch(
        v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    output = scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True, scale=scale)
    output_torch = ttnn.to_torch(output)

    assert_with_pcc(ref, output_torch, pcc=PCC_THRESHOLD)


# ── Causal with GQA / MQA ───────────────────────────────────────────


@pytest.mark.parametrize(
    "B, H_q, H_kv, S, D, desc",
    [
        (1, 8, 2, 128, 64, "gqa_4to1"),
        (1, 8, 2, 256, 64, "gqa_4to1_long"),
        (1, 32, 8, 128, 128, "gqa_llama3"),
        (1, 8, 1, 128, 64, "mqa"),
        (1, 12, 1, 128, 64, "mqa_bert"),
        (1, 16, 1, 256, 64, "mqa_long"),
        (2, 8, 2, 128, 64, "gqa_batch"),
    ],
)
def test_causal_gqa_mqa(device, B, H_q, H_kv, S, D, desc):
    """Causal mask with GQA/MQA head broadcasting."""
    torch.manual_seed(42)
    q = torch.randn(B, H_q, S, D, dtype=torch.bfloat16)
    k = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16)
    v = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16)

    # Reference: broadcast K/V to match Q heads
    repeats = H_q // H_kv
    k_ref = k.repeat_interleave(repeats, dim=1)
    v_ref = v.repeat_interleave(repeats, dim=1)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k_ref, v_ref, is_causal=True)

    q_t = ttnn.from_torch(
        q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    k_t = ttnn.from_torch(
        k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    v_t = ttnn.from_torch(
        v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    output = scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True)
    output_torch = ttnn.to_torch(output)

    assert_with_pcc(ref, output_torch, pcc=PCC_THRESHOLD)


# ── Causal + cross-attention EXCLUSION ──────────────────────────────


def test_causal_cross_attention_exclusion(device):
    """Causal mask + cross-attention (S_q != S_kv) should raise ExcludedCell."""
    torch.manual_seed(42)
    q = torch.randn(1, 4, 64, 64, dtype=torch.bfloat16)
    k = torch.randn(1, 4, 128, 64, dtype=torch.bfloat16)
    v = torch.randn(1, 4, 128, 64, dtype=torch.bfloat16)

    q_t = ttnn.from_torch(
        q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    k_t = ttnn.from_torch(
        k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    v_t = ttnn.from_torch(
        v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    with pytest.raises(NotImplementedError):
        scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True)


# ── is_causal + attn_mask mutual exclusion ─────────────────────────


def test_causal_with_attn_mask_mutual_exclusion(device):
    """is_causal=True + attn_mask should raise ValueError."""
    torch.manual_seed(42)
    q = torch.randn(1, 1, 128, 64, dtype=torch.bfloat16)
    k = torch.randn(1, 1, 128, 64, dtype=torch.bfloat16)
    v = torch.randn(1, 1, 128, 64, dtype=torch.bfloat16)
    mask = torch.zeros(1, 1, 128, 128, dtype=torch.bfloat16)

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

    with pytest.raises(ValueError):
        scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=mask_t, is_causal=True)


# ── Causal equivalence with custom triangular mask ─────────────────


@pytest.mark.parametrize(
    "B, H, S, D",
    [
        (1, 1, 128, 64),
        (1, 4, 128, 64),
        (1, 1, 256, 64),
    ],
)
def test_causal_equals_custom_triangular_mask(device, B, H, S, D):
    """is_causal=True should produce the same result as a custom triangular mask."""
    torch.manual_seed(42)
    q = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    k = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    v = torch.randn(B, H, S, D, dtype=torch.bfloat16)

    # Custom triangular mask (same pattern as causal)
    mask = torch.zeros(B, 1, S, S, dtype=torch.bfloat16)
    for i in range(S):
        for j in range(S):
            if j > i:
                mask[:, :, i, j] = float("-inf")

    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

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

    # Run both paths
    output_causal = scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True)
    output_custom = scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=mask_t)

    output_causal_torch = ttnn.to_torch(output_causal)
    output_custom_torch = ttnn.to_torch(output_custom)

    # Both should match the reference
    assert_with_pcc(ref, output_causal_torch, pcc=PCC_THRESHOLD)
    assert_with_pcc(ref, output_custom_torch, pcc=PCC_THRESHOLD)

    # And they should match each other (the on-device causal mask should
    # produce the same result as the custom triangular mask path)
    assert_with_pcc(output_causal_torch, output_custom_torch, pcc=0.999)


# ── Sequential state-leak regression ────────────────────────────────


def test_causal_then_no_mask_sequential(device):
    """Run causal, then no-mask — verify no state leak between calls."""
    torch.manual_seed(42)
    S, D = 128, 64

    # First call: causal
    q1 = torch.randn(1, 1, S, D, dtype=torch.bfloat16)
    k1 = torch.randn(1, 1, S, D, dtype=torch.bfloat16)
    v1 = torch.randn(1, 1, S, D, dtype=torch.bfloat16)
    ref1 = torch.nn.functional.scaled_dot_product_attention(q1, k1, v1, is_causal=True)

    q1_t = ttnn.from_torch(
        q1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    k1_t = ttnn.from_torch(
        k1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    v1_t = ttnn.from_torch(
        v1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out1 = scaled_dot_product_attention(q1_t, k1_t, v1_t, is_causal=True)
    assert_with_pcc(ref1, ttnn.to_torch(out1), pcc=PCC_THRESHOLD)

    # Second call: no mask (should not be affected by the causal mask from call 1)
    q2 = torch.randn(1, 1, S, D, dtype=torch.bfloat16)
    k2 = torch.randn(1, 1, S, D, dtype=torch.bfloat16)
    v2 = torch.randn(1, 1, S, D, dtype=torch.bfloat16)
    ref2 = torch.nn.functional.scaled_dot_product_attention(q2, k2, v2)

    q2_t = ttnn.from_torch(
        q2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    k2_t = ttnn.from_torch(
        k2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    v2_t = ttnn.from_torch(
        v2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out2 = scaled_dot_product_attention(q2_t, k2_t, v2_t)
    assert_with_pcc(ref2, ttnn.to_torch(out2), pcc=PCC_THRESHOLD)


# ── Deterministic all-ones input ────────────────────────────────────


def test_causal_all_ones_input(device):
    """All-ones input with causal mask: each row i attends to cols 0..i.

    With all-ones Q/K/V, the attention scores are all 1.0 (before mask).
    After causal masking, row i has (i+1) unmasked positions each with score 1.0.
    Softmax makes each unmasked position get weight 1/(i+1).
    The output for row i is then the average of V[0..i] = 1.0 (since V is all ones).
    So the output should be all ones.
    """
    S, D = 128, 64
    q = torch.ones(1, 1, S, D, dtype=torch.bfloat16)
    k = torch.ones(1, 1, S, D, dtype=torch.bfloat16)
    v = torch.ones(1, 1, S, D, dtype=torch.bfloat16)

    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

    q_t = ttnn.from_torch(
        q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    k_t = ttnn.from_torch(
        k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    v_t = ttnn.from_torch(
        v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    output = scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True)
    output_torch = ttnn.to_torch(output)

    # With all-ones input and causal mask, output should be all ones
    # (row 0 attends only to position 0 → output = V[0] = 1.0;
    #  row i attends to positions 0..i → average = 1.0)
    assert_with_pcc(ref, output_torch, pcc=PCC_THRESHOLD)

    # The reference itself should be all ones
    assert torch.allclose(
        ref.float(), torch.ones_like(ref.float()), rtol=0.01, atol=0.01
    ), f"Expected all ones, got max={ref.float().max()}, min={ref.float().min()}"


# ── Causal with non-power-of-2 heads ────────────────────────────────


@pytest.mark.parametrize(
    "B, H, S, D",
    [
        (1, 3, 96, 64),
        (1, 7, 128, 64),
    ],
)
def test_causal_non_power_of_2_heads(device, B, H, S, D):
    """Causal mask with non-power-of-2 head counts."""
    torch.manual_seed(42)
    q = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    k = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    v = torch.randn(B, H, S, D, dtype=torch.bfloat16)

    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

    q_t = ttnn.from_torch(
        q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    k_t = ttnn.from_torch(
        k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    v_t = ttnn.from_torch(
        v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    output = scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True)
    output_torch = ttnn.to_torch(output)

    assert_with_pcc(ref, output_torch, pcc=PCC_THRESHOLD)


# ── Causal with long context ────────────────────────────────────────


@pytest.mark.parametrize(
    "S",
    [4096, 2048],
)
def test_causal_long_context(device, S):
    """Causal mask with long-context sequences."""
    torch.manual_seed(42)
    D = 64
    q = torch.randn(1, 1, S, D, dtype=torch.bfloat16)
    k = torch.randn(1, 1, S, D, dtype=torch.bfloat16)
    v = torch.randn(1, 1, S, D, dtype=torch.bfloat16)

    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

    q_t = ttnn.from_torch(
        q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    k_t = ttnn.from_torch(
        k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    v_t = ttnn.from_torch(
        v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    output = scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True)
    output_torch = ttnn.to_torch(output)

    assert_with_pcc(ref, output_torch, pcc=PCC_THRESHOLD)
