# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 5 — fp32 large-head-dim L1 budget (D=1024).

The 4 `Q1x1x128x1024` fp32 golden cells previously threw `program.cpp:1450`
("statically allocated circular buffers grow beyond max L1 size"): fp32 is
4 B/elem and the seven D_t-scaling, double-buffered CBs total ~1.82 MB > the
1.5 MB L1 budget at D_t=32. The program descriptor now single-buffers the
intra-compute scratch CBs (cb_o_acc / cb_pv / cb_o_tmp) and, when still tight,
cb_out — only as far as needed to fit, and only on the shapes that would OOM.
bf16 / bf8b and small-D fp32 keep their double-buffered sizing (byte-identical
to Refinement 4).

These tests exercise fp32 D=1024 directly (the golden shape and a few neighbors)
across mask modes and head-broadcast modes, and assert no L1 OOM + correctness.
"""

import math

import pytest
import torch

import ttnn


def _pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def _ref(q, k, v, attn_mask=None, is_causal=False, scale=None):
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=scale
    )


def _run(device, q, k, v, *, attn_mask=None, is_causal=False, scale=None):
    from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

    qt = ttnn.from_torch(q, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    kt = ttnn.from_torch(k, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    vt = ttnn.from_torch(v, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    mt = None
    if attn_mask is not None:
        mt = ttnn.from_torch(attn_mask, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    out = scaled_dot_product_attention(qt, kt, vt, attn_mask=mt, is_causal=is_causal, scale=scale)
    return ttnn.to_torch(out)


# The golden shape (B,H,S_q,S_kv,D) and a couple of fp32 large-D neighbors that
# also push the D_t-scaling CBs past the double-buffered L1 budget.
@pytest.mark.parametrize(
    "B,H,S,D",
    [
        (1, 1, 128, 1024),  # the 4 golden Q1x1x128x1024 fp32 cells
        (1, 2, 64, 1024),  # multi-head, large D
        (2, 1, 96, 1024),  # multi-batch, large D
    ],
)
def test_fp32_large_head_dim_no_mask(device, B, H, S, D):
    torch.manual_seed(0)
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)
    out = _run(device, q, k, v)
    ref = _ref(q, k, v)
    assert out.shape == (B, H, S, D)
    assert not torch.isnan(out).any()
    assert _pcc(out, ref) >= 0.999


def test_fp32_large_head_dim_custom_mask(device):
    B, H, S, D = 1, 1, 128, 1024
    torch.manual_seed(1)
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)
    # additive lower-triangular bias (well-formed custom mask)
    mask = torch.triu(torch.full((S, S), float("-inf")), diagonal=1).reshape(1, 1, S, S)
    out = _run(device, q, k, v, attn_mask=mask)
    ref = _ref(q, k, v, attn_mask=mask)
    assert not torch.isnan(out).any()
    assert _pcc(out, ref) >= 0.999


def test_fp32_large_head_dim_causal(device):
    B, H, S, D = 1, 1, 128, 1024
    torch.manual_seed(2)
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)
    out = _run(device, q, k, v, is_causal=True)
    ref = _ref(q, k, v, is_causal=True)
    assert not torch.isnan(out).any()
    assert _pcc(out, ref) >= 0.999


def test_fp32_large_head_dim_explicit_scale(device):
    B, H, S, D = 1, 1, 128, 1024
    torch.manual_seed(3)
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)
    scale = 0.5 / math.sqrt(D)
    out = _run(device, q, k, v, scale=scale)
    ref = _ref(q, k, v, scale=scale)
    assert not torch.isnan(out).any()
    assert _pcc(out, ref) >= 0.999


def test_fp32_large_head_dim_gqa(device):
    # GQA: H_q=4, H_kv=1 (head broadcast composes with the L1-budget sizing).
    B, Hq, Hkv, S, D = 1, 4, 1, 96, 1024
    torch.manual_seed(4)
    q = torch.randn(B, Hq, S, D)
    k = torch.randn(B, Hkv, S, D)
    v = torch.randn(B, Hkv, S, D)
    from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

    qt = ttnn.from_torch(q, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    kt = ttnn.from_torch(k, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    vt = ttnn.from_torch(v, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(scaled_dot_product_attention(qt, kt, vt))
    # reference broadcasts the single KV head across all Q heads
    ref = _ref(q, k.repeat_interleave(Hq // Hkv, dim=1), v.repeat_interleave(Hq // Hkv, dim=1))
    assert not torch.isnan(out).any()
    assert _pcc(out, ref) >= 0.999


def test_bf16_large_head_dim_unchanged(device):
    # Sanity: bf16 D=1024 already fit double-buffered; confirm still passes
    # (the L1-budget single-buffering must not be triggered for bf16).
    B, H, S, D = 1, 1, 128, 1024
    torch.manual_seed(5)
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)
    from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

    qt = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    kt = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    vt = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(scaled_dot_product_attention(qt, kt, vt))
    ref = _ref(q, k, v)
    assert not torch.isnan(out).any()
    assert _pcc(out, ref) >= 0.995
