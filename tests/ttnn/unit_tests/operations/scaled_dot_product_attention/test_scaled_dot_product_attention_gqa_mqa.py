# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 2 — GQA / MQA (KV head broadcast).

Exercises grouped-query (H_q > H_kv > 1) and multi-query (H_kv == 1)
attention. The reader remaps each Q head h to KV head h // (H_q / H_kv);
output / Q indexing stay on H_q. The torch reference broadcasts K/V via
repeat_interleave, which is exactly the same head→KV-head mapping.
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


def _torch_reference(q, k, v, scale=None):
    """SDPA with GQA/MQA head broadcast (repeat_interleave on K/V)."""
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    h_q, h_kv = q.shape[1], k.shape[1]
    if h_q != h_kv:
        repeats = h_q // h_kv
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v.float())


# (Q_shape, K/V_shape, label) — realistic GQA / MQA ratios.
CONFIGS = [
    ((1, 8, 128, 64), (1, 2, 128, 64), "gqa_4to1"),
    ((1, 32, 128, 128), (1, 8, 128, 128), "gqa_llama3_4to1"),
    ((2, 8, 256, 64), (2, 4, 256, 64), "gqa_2to1_multibatch"),
    ((1, 8, 256, 64), (1, 1, 256, 64), "mqa_8to1"),
    ((1, 32, 128, 128), (1, 1, 128, 128), "mqa_32to1"),
    ((2, 16, 512, 64), (2, 2, 512, 64), "gqa_8to1_long"),
]
IDS = [c[2] for c in CONFIGS]


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b])
@pytest.mark.parametrize("q_shape,kv_shape,label", CONFIGS, ids=IDS)
def test_gqa_mqa(q_shape, kv_shape, label, dtype, device):
    torch.manual_seed(2024)
    q = torch.randn(q_shape, dtype=torch.float32)
    k = torch.randn(kv_shape, dtype=torch.float32)
    v = torch.randn(kv_shape, dtype=torch.float32)

    expected = _torch_reference(q, k, v)

    ttnn_q = ttnn.from_torch(q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_k = ttnn.from_torch(k, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_v = ttnn.from_torch(v, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    ttnn_out = scaled_dot_product_attention(ttnn_q, ttnn_k, ttnn_v)
    result = ttnn.to_torch(ttnn_out)

    assert_with_pcc(expected, result.float(), PCC_BY_DTYPE[dtype])


def test_mqa_with_mask(device):
    """MQA composes with a custom additive mask (broadcast over Q heads)."""
    torch.manual_seed(7)
    B, H_q, H_kv, S, D = 1, 8, 1, 128, 64
    q = torch.randn(B, H_q, S, D, dtype=torch.float32)
    k = torch.randn(B, H_kv, S, D, dtype=torch.float32)
    v = torch.randn(B, H_kv, S, D, dtype=torch.float32)
    mask = torch.zeros(B, 1, S, S, dtype=torch.float32)
    mask[..., S // 2 :] = -1e9  # mask out the second half of keys

    scale = 1.0 / math.sqrt(D)
    k_b = k.repeat_interleave(H_q, dim=1)
    v_b = v.repeat_interleave(H_q, dim=1)
    scores = torch.matmul(q, k_b.transpose(-2, -1)) * scale + mask
    expected = torch.matmul(torch.softmax(scores, dim=-1), v_b)

    ttnn_q = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_k = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_v = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_mask = ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    ttnn_out = scaled_dot_product_attention(ttnn_q, ttnn_k, ttnn_v, attn_mask=ttnn_mask)
    result = ttnn.to_torch(ttnn_out)

    assert_with_pcc(expected, result.float(), 0.995)


def test_non_divisible_heads_rejected(device):
    """H_q not a multiple of H_kv must raise (undefined head broadcast)."""
    q = torch.randn(1, 6, 128, 64, dtype=torch.float32)
    k = torch.randn(1, 4, 128, 64, dtype=torch.float32)
    v = torch.randn(1, 4, 128, 64, dtype=torch.float32)
    ttnn_q = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_k = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_v = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    with pytest.raises(ValueError):
        scaled_dot_product_attention(ttnn_q, ttnn_k, ttnn_v)
