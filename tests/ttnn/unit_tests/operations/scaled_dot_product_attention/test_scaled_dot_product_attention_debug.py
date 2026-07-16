# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

# Deterministic debugging test for scaled_dot_product_attention.
# DO NOT DELETE — documents the debugging process (DEVICE_PRINT + hand-calc).

import math

import pytest
import torch

import ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


def _ref(q, k, v, scale=None):
    B, H, Sq, D = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    w = torch.softmax(scores, dim=-1)
    return torch.matmul(w, v.float())


def test_single_tile_ones(device):
    """All-ones Q,K,V single tile. scores = D*scale (uniform) -> softmax uniform
    (1/Skv each) -> output = mean of V rows = 1.0 everywhere."""
    q = torch.ones(1, 1, 32, 32, dtype=torch.bfloat16)
    k = torch.ones(1, 1, 32, 32, dtype=torch.bfloat16)
    v = torch.ones(1, 1, 32, 32, dtype=torch.bfloat16)

    tq = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    out = scaled_dot_product_attention(tq, tk, tv)
    res = ttnn.to_torch(out).float()
    expected = torch.ones(1, 1, 32, 32)
    diff = (res - expected).abs().max()
    assert diff < 0.1, f"max diff {diff}\n{res[0,0,:2,:4]}"


def test_single_tile_random(device):
    torch.manual_seed(0)
    q = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
    k = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
    v = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)

    tq = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    out = scaled_dot_product_attention(tq, tk, tv)
    res = ttnn.to_torch(out).float()
    ref = _ref(q, k, v)
    diff = (res - ref).abs().max()
    assert diff < 0.2, f"max diff {diff}\nres={res[0,0,:2,:4]}\nref={ref[0,0,:2,:4]}"


def test_multi_tile_d64(device):
    """128x64: Sq_t=4, Skv_t=4, Dt=2 — exercises multi-tile D contraction + multi
    q/kv chunk (Sq_chunk_t=Skv_chunk_t=4 => single chunk each here)."""
    torch.manual_seed(1)
    q = torch.randn(1, 1, 128, 64, dtype=torch.bfloat16)
    k = torch.randn(1, 1, 128, 64, dtype=torch.bfloat16)
    v = torch.randn(1, 1, 128, 64, dtype=torch.bfloat16)

    tq = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    out = scaled_dot_product_attention(tq, tk, tv)
    res = ttnn.to_torch(out).float()
    ref = _ref(q, k, v)
    diff = (res - ref).abs().max()
    assert diff < 0.2, f"max diff {diff}\nres={res[0,0,:2,:4]}\nref={ref[0,0,:2,:4]}"
