# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 4 — L1 budget fit for large head_dim.

Phase 0 sized the Dt-scaling CBs (cb_q/cb_qs/cb_k/cb_v and the fp32
cb_pv/cb_o_run/cb_o_new + cb_out) at chunk*Dt, which blows the ~1.5 MB per-core
L1 budget as head_dim grows — the op OOM'd at program launch for D >= 256
(``Statically allocated circular buffers ... grow to N B which is beyond max
L1 size``). This test exercises the head-dim-scaling golden shapes directly:
they must now build (no OOM) AND stay numerically correct.

The fix is host-side (single-buffered compute->compute CBs + a projected-
footprint chunk/buffer selection); nothing about the math changes, so PCC must
match the prior small-head_dim baseline.
"""

import math

import pytest
import torch
import ttnn

from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


def _pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    if a.std() == 0 or b.std() == 0:
        return float(torch.allclose(a, b))
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _ref(Q, K, V, *, attn_mask=None, scale=None):
    Qf, Kf, Vf = Q.float(), K.float(), V.float()
    H_q, H_kv = Qf.shape[1], Kf.shape[1]
    if H_q != H_kv:
        r = H_q // H_kv
        Kf, Vf = Kf.repeat_interleave(r, 1), Vf.repeat_interleave(r, 1)
    m = attn_mask.float() if attn_mask is not None else None
    return torch.nn.functional.scaled_dot_product_attention(Qf, Kf, Vf, attn_mask=m, scale=scale)


# The exact head-dim-scaling golden shapes that OOM'd at Phase 0, plus D=128
# (which OOM'd for fp32) and a couple multi-head/batch large-D configs.
LARGE_HEAD_DIM_SHAPES = [
    (1, 1, 128, 128),
    (1, 1, 128, 256),
    (1, 1, 128, 512),
    (1, 1, 128, 1024),
    (2, 4, 256, 128),
    (1, 32, 128, 128),
]


@pytest.mark.parametrize("shape", LARGE_HEAD_DIM_SHAPES, ids=lambda s: f"B{s[0]}_H{s[1]}_S{s[2]}_D{s[3]}")
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b], ids=["bf16", "fp32", "bf8b"])
def test_large_head_dim_fits_and_correct(device, shape, dtype):
    """Large head_dim builds (no L1 OOM) and matches the torch reference."""
    B, H, S, D = shape
    torch.manual_seed(0)
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)
    ref = _ref(q, k, v)

    tq = ttnn.from_torch(q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(k, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(v, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    out = scaled_dot_product_attention(tq, tk, tv)
    res = ttnn.to_torch(out).float()

    pcc = _pcc(res, ref)
    # bf8b is lower precision; bf16/fp32 track the small-head_dim baseline closely.
    target = 0.99 if dtype == ttnn.bfloat8_b else 0.999
    assert pcc >= target, f"shape={shape} dtype={dtype} PCC={pcc:.6f} < {target}"


@pytest.mark.parametrize("D", [256, 512, 1024], ids=lambda d: f"D{d}")
@pytest.mark.parametrize("mask_mode", ["none", "custom", "causal"])
def test_large_head_dim_mask_modes(device, D, mask_mode):
    """All three mask modes fit and stay correct at large head_dim (bf16)."""
    B, H, S = 1, 1, 128
    torch.manual_seed(0)
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)

    attn_mask = None
    is_causal = False
    ref_mask = None
    if mask_mode == "custom":
        attn_mask = torch.randn(B, 1, S, S) * 0.1
        ref_mask = attn_mask
    elif mask_mode == "causal":
        is_causal = True
        tri = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
        ref_mask = torch.zeros(B, 1, S, S)
        ref_mask.masked_fill_(tri, float("-inf"))

    ref = _ref(q, k, v, attn_mask=ref_mask)

    tq = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tm = (
        ttnn.from_torch(attn_mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        if attn_mask is not None
        else None
    )

    out = scaled_dot_product_attention(tq, tk, tv, attn_mask=tm, is_causal=is_causal)
    res = ttnn.to_torch(out).float()

    pcc = _pcc(res, ref)
    assert pcc >= 0.995, f"D={D} mask_mode={mask_mode} PCC={pcc:.6f}"


@pytest.mark.parametrize("scale_mode", ["auto", "explicit"])
def test_large_head_dim_scale_modes(device, scale_mode):
    """auto (1/sqrt(D)) and explicit scale both fit + correct at D=1024."""
    B, H, S, D = 1, 1, 128, 1024
    torch.manual_seed(0)
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)
    scale = 0.03 if scale_mode == "explicit" else None
    ref = _ref(q, k, v, scale=scale)

    tq = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    out = scaled_dot_product_attention(tq, tk, tv, scale=scale)
    res = ttnn.to_torch(out).float()
    pcc = _pcc(res, ref)
    assert pcc >= 0.999, f"scale_mode={scale_mode} PCC={pcc:.6f}"
