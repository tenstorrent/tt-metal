# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 3 — Causal masking (on-device triangular bias).

`is_causal=True` makes the op generate an upper-triangular -inf bias
ON-DEVICE (no caller tensor, no materialized S_q x S_kv mask) and add it
to each score block before the running max. Three regions per
(Q-block i, KV-block j): past blocks (j < qi) unmasked; future blocks
(j > qi) skipped entirely (no QK^T/softmax/PV — the ~half-KV-work win);
the single diagonal block (j == qi) gets the per-element triangular mask.

Causal requires S_q == S_kv (decoder self-attention), so causal +
cross-attention is an op EXCLUSION (raises NotImplementedError).
"""

import math

import pytest
import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


PCC_BY_DTYPE = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.995,
    ttnn.bfloat8_b: 0.99,
}


def _torch_causal_reference(q, k, v, scale=None):
    """SDPA with a causal (triangular) mask + GQA/MQA head broadcast."""
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    h_q, h_kv = q.shape[1], k.shape[1]
    if h_q != h_kv:
        repeats = h_q // h_kv
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
    s_q, s_kv = q.shape[-2], k.shape[-2]
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    mask = torch.triu(torch.full((s_q, s_kv), float("-inf")), diagonal=1)
    scores = scores + mask
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v.float())


# Self-attention shapes (S_q == S_kv). Cover single-tile, multi-tile (so the
# KV loop hits past + diagonal blocks), multi-head, multi-batch, GQA, MQA.
SELF_CONFIGS = [
    ((1, 1, 32, 32), (1, 1, 32, 32), "single_tile"),
    ((1, 1, 32, 64), (1, 1, 32, 64), "single_qtile_wideD"),
    ((1, 1, 128, 64), (1, 1, 128, 64), "4kv_blocks"),
    ((1, 1, 256, 64), (1, 1, 256, 64), "8kv_blocks"),
    ((1, 4, 128, 64), (1, 4, 128, 64), "multihead"),
    ((2, 4, 128, 64), (2, 4, 128, 64), "multibatch"),
    ((1, 8, 256, 64), (1, 8, 256, 64), "multihead_long"),
    ((1, 8, 128, 64), (1, 2, 128, 64), "gqa_4to1"),
    ((1, 8, 128, 64), (1, 1, 128, 64), "mqa_8to1"),
    ((1, 32, 128, 128), (1, 8, 128, 128), "gqa_llama3"),
]
SELF_IDS = [c[2] for c in SELF_CONFIGS]


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b])
@pytest.mark.parametrize("q_shape,kv_shape,label", SELF_CONFIGS, ids=SELF_IDS)
def test_causal_self_attention(q_shape, kv_shape, label, dtype, device):
    torch.manual_seed(2024)
    q = torch.randn(q_shape, dtype=torch.float32)
    k = torch.randn(kv_shape, dtype=torch.float32)
    v = torch.randn(kv_shape, dtype=torch.float32)

    expected = _torch_causal_reference(q, k, v)

    ttnn_q = ttnn.from_torch(q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_k = ttnn.from_torch(k, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_v = ttnn.from_torch(v, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    ttnn_out = scaled_dot_product_attention(ttnn_q, ttnn_k, ttnn_v, is_causal=True)
    result = ttnn.to_torch(ttnn_out)

    assert_with_pcc(expected, result.float(), PCC_BY_DTYPE[dtype])


@pytest.mark.parametrize("scale", [None, 0.125])
def test_causal_explicit_scale(scale, device):
    """Causal composes with the explicit-scale path."""
    torch.manual_seed(11)
    q = torch.randn(1, 4, 256, 64, dtype=torch.float32)
    k = torch.randn(1, 4, 256, 64, dtype=torch.float32)
    v = torch.randn(1, 4, 256, 64, dtype=torch.float32)

    expected = _torch_causal_reference(q, k, v, scale=scale)

    ttnn_q = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_k = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_v = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    ttnn_out = scaled_dot_product_attention(ttnn_q, ttnn_k, ttnn_v, is_causal=True, scale=scale)
    result = ttnn.to_torch(ttnn_out)

    assert_with_pcc(expected, result.float(), 0.995)


def test_causal_first_row_attends_only_to_first_key(device):
    """Deterministic diagonal-block check. With a single Q tile-row (qi==0),
    output row 0 may attend ONLY to key position 0 (everything above the
    diagonal is masked). With keys made distinct per position, output[...,0,:]
    must equal V[...,0,:] exactly (softmax over a single unmasked logit)."""
    B, H, S, D = 1, 1, 32, 32
    q = torch.ones(B, H, S, D, dtype=torch.float32)
    k = torch.zeros(B, H, S, D, dtype=torch.float32)
    # Distinct, separable keys so the masked logits would otherwise dominate.
    for s in range(S):
        k[0, 0, s, 0] = float(s)
    v = torch.arange(S, dtype=torch.float32).reshape(1, 1, S, 1).repeat(1, 1, 1, D)

    expected = _torch_causal_reference(q, k, v)

    ttnn_q = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_k = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_v = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    ttnn_out = scaled_dot_product_attention(ttnn_q, ttnn_k, ttnn_v, is_causal=True)
    result = ttnn.to_torch(ttnn_out).float()

    # Row 0 attends only to key 0 → output row 0 == V row 0 (== 0).
    assert torch.allclose(
        result[0, 0, 0, :], expected[0, 0, 0, :], atol=0.05
    ), f"row0: got {result[0, 0, 0, 0].item()} expected {expected[0, 0, 0, 0].item()}"
    assert_with_pcc(expected, result, 0.99)


def test_causal_cross_attention_excluded(device):
    """Causal + cross-attention (S_q != S_kv) is an EXCLUSION → NotImplementedError."""
    q = torch.randn(1, 4, 64, 64, dtype=torch.float32)
    k = torch.randn(1, 4, 128, 64, dtype=torch.float32)
    v = torch.randn(1, 4, 128, 64, dtype=torch.float32)
    ttnn_q = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_k = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_v = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    with pytest.raises(NotImplementedError):
        scaled_dot_product_attention(ttnn_q, ttnn_k, ttnn_v, is_causal=True)


def test_causal_and_mask_mutually_exclusive(device):
    """is_causal + attn_mask must raise ValueError."""
    q = torch.randn(1, 1, 128, 64, dtype=torch.float32)
    mask = torch.zeros(1, 1, 128, 128, dtype=torch.float32)
    ttnn_q = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_mask = ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    with pytest.raises(ValueError):
        scaled_dot_product_attention(ttnn_q, ttnn_q, ttnn_q, attn_mask=ttnn_mask, is_causal=True)
