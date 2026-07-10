# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic debug tests for Flash-Attention SDPA. DO NOT DELETE.

Exercises the online-softmax recurrence (num_kv_blocks > 1), which the acceptance
test never reaches (all its shapes have S_kv <= 128 => single KV block). Also
covers Q-chunking (num_q_blocks_per_bh > 1) and long-context multi-core.
"""

import math

import pytest
import torch

import ttnn

from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


def _pcc(golden, computed):
    a = golden.flatten().to(torch.float32)
    b = computed.flatten().to(torch.float32)
    if torch.allclose(a, b):
        return 1.0
    if a.std() == 0 or b.std() == 0:
        return 1.0 if torch.allclose(a, b) else 0.0
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _reference(Q, K, V, scale=None):
    return torch.nn.functional.scaled_dot_product_attention(
        Q.float(), K.float(), V.float(), scale=scale
    )


# Shapes chosen so Skv_t > kv_chunk_t (=> multiple KV blocks) and/or
# Sq_t > q_chunk_t (=> multiple Q chunks per head).
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 128, 256, 64),  # S_kv=256 -> Skv_t=8 -> 2 KV blocks
        (1, 1, 256, 256, 64),  # both S=256 -> 2 Q chunks, 2 KV blocks
        (1, 1, 128, 512, 64),  # S_kv=512 -> 4 KV blocks
        (1, 2, 256, 384, 64),  # multi-head, uneven KV (384 -> Skv_t=12 -> 3 blocks)
        (2, 4, 384, 256, 64),  # multi-batch, multi Q chunk + multi KV block
        (1, 8, 256, 256, 128),  # multi-head, D=128
    ],
)
def test_multi_kv_block(device, shape):
    torch.manual_seed(0)
    B, H, S_q, S_kv, D = shape
    Q = torch.randn(B, H, S_q, D, dtype=torch.bfloat16)
    K = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16)
    V = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16)

    expected = _reference(Q, K, V)

    to_dev = lambda t: ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = ttnn.to_torch(scaled_dot_product_attention(to_dev(Q), to_dev(K), to_dev(V))).float()

    assert list(out.shape) == [B, H, S_q, D]
    pcc = _pcc(expected, out)
    assert pcc >= 0.99, f"PCC {pcc:.5f} for shape {shape}"


def test_all_ones_multi_kv(device):
    """All-ones: softmax is uniform -> output = mean(V) = 1.0 everywhere."""
    B, H, S_q, S_kv, D = 1, 1, 64, 256, 64
    Q = torch.ones(B, H, S_q, D, dtype=torch.bfloat16)
    K = torch.ones(B, H, S_kv, D, dtype=torch.bfloat16)
    V = torch.ones(B, H, S_kv, D, dtype=torch.bfloat16)

    to_dev = lambda t: ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = ttnn.to_torch(scaled_dot_product_attention(to_dev(Q), to_dev(K), to_dev(V))).float()
    # softmax of constant scores is uniform; sum_j (1/S_kv) * 1.0 = 1.0
    assert torch.allclose(out, torch.ones_like(out), rtol=0.05, atol=0.05), f"max diff {(out-1.0).abs().max()}"
