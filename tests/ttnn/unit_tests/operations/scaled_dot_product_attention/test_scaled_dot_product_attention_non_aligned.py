# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 4 — Non-tile-aligned sequence / head dim.

SDPA is TILE-only, but TILE tensors whose logical S_q / S_kv / D are not
multiples of 32 carry a partial last tile. from_torch's tilization pads that
tile with NON-ZERO garbage, which (a) pollutes the QKᵀ contraction over D and
(b) lets padded key columns enter the online-softmax. This suite exercises
the two non-aligned alignment values (w_non_aligned: D%32; h_non_aligned:
S_q%32, D aligned) across bf16 + fp32 and every mask mode, plus the
bfloat8_b × non-aligned EXCLUSION.

The layout axis itself has no gap (TILE only), so this is the alignment
analogue of the /memory-layouts "non-aligned rule": last-tile zero-pad / mask
at the data-access boundary, math always on full tiles.
"""

from __future__ import annotations

import math

import pytest
import torch
import ttnn

from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


# Per-dtype PCC gate (mirrors the golden TOLERANCES). SDPA chains
# matmul -> softmax -> matmul so error compounds.
_PCC = {ttnn.bfloat16: 0.995, ttnn.float32: 0.999}
_TORCH_DTYPE = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}


def _ref(Q, K, V, *, attn_mask=None, is_causal=False, scale=None):
    Qf, Kf, Vf = Q.float(), K.float(), V.float()
    H_q, H_kv = Qf.shape[1], Kf.shape[1]
    if H_q != H_kv:
        r = H_q // H_kv
        Kf = Kf.repeat_interleave(r, dim=1)
        Vf = Vf.repeat_interleave(r, dim=1)
    D = Qf.shape[-1]
    s = scale if scale is not None else 1.0 / math.sqrt(D)
    scores = torch.matmul(Qf, Kf.transpose(-2, -1)) * s
    if is_causal:
        S_q, S_kv = Qf.shape[-2], Kf.shape[-2]
        scores = scores + torch.triu(torch.full((S_q, S_kv), float("-inf")), diagonal=1)
    elif attn_mask is not None:
        scores = scores + attn_mask.float()
    w = torch.softmax(scores, dim=-1)
    return torch.matmul(w, Vf)


def _pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    if torch.allclose(a, b):
        return 1.0
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


# (Q_shape, K_shape, alignment) — self + cross + GQA/MQA, single + multi tile,
# multi-head + multi-batch. Mirrors feature_spec.py's non-aligned INPUTS.
_SHAPES = [
    pytest.param((1, 1, 32, 50), (1, 1, 32, 50), id="w_D50_Saligned"),
    pytest.param((1, 1, 47, 64), (1, 1, 47, 64), id="h_S47_Daligned"),
    pytest.param((1, 1, 50, 50), (1, 1, 50, 50), id="both_S50_D50"),
    pytest.param((1, 8, 64, 47), (1, 8, 64, 47), id="w_D47_multihead"),
    pytest.param((1, 12, 33, 50), (1, 12, 33, 50), id="both_S33_D50_multihead"),
    pytest.param((2, 4, 100, 64), (2, 4, 100, 64), id="h_S100_multibatch"),
    pytest.param((1, 4, 47, 64), (1, 4, 47, 64), id="h_S47_multihead"),
    pytest.param((1, 8, 47, 64), (1, 2, 47, 64), id="h_S47_gqa"),
    pytest.param((1, 8, 47, 64), (1, 1, 47, 64), id="h_S47_mqa"),
    pytest.param((1, 4, 100, 50), (1, 4, 47, 50), id="w_cross_Sq100_Skv47_D50"),
]


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("scale_mode", ["auto", "explicit"])
@pytest.mark.parametrize("q_shape,k_shape", _SHAPES)
def test_non_aligned_none(device, q_shape, k_shape, scale_mode, dtype):
    """No mask, every non-aligned shape × dtype × scale mode."""
    torch_dtype = _TORCH_DTYPE[dtype]
    torch.manual_seed(0)
    Q = torch.randn(q_shape, dtype=torch_dtype)
    K = torch.randn(k_shape, dtype=torch_dtype)
    V = torch.randn(k_shape, dtype=torch_dtype)
    scale = 0.125 if scale_mode == "explicit" else None
    exp = _ref(Q, K, V, scale=scale)

    tQ = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tK = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tV = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(scaled_dot_product_attention(tQ, tK, tV, scale=scale))

    assert out.shape == torch.Size(q_shape), f"shape {out.shape} != {q_shape}"
    assert not torch.isnan(out.float()).any(), "NaN in output (padding leaked)"
    pcc = _pcc(out, exp)
    assert pcc >= _PCC[dtype], f"PCC {pcc:.5f} < {_PCC[dtype]}"


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("q_shape,k_shape", _SHAPES)
def test_non_aligned_custom_mask(device, q_shape, k_shape, dtype):
    """Additive triangular mask + non-aligned: the kv-pad -inf mask must
    compose with the caller mask on the last KV tile."""
    torch_dtype = _TORCH_DTYPE[dtype]
    torch.manual_seed(0)
    Q = torch.randn(q_shape, dtype=torch_dtype)
    K = torch.randn(k_shape, dtype=torch_dtype)
    V = torch.randn(k_shape, dtype=torch_dtype)
    B, _H, S_q, _D = q_shape
    S_kv = k_shape[-2]
    mask = torch.zeros(B, 1, S_q, S_kv, dtype=torch_dtype)
    mask.masked_fill_(torch.triu(torch.ones(S_q, S_kv, dtype=torch.bool), diagonal=1), float("-inf"))
    exp = _ref(Q, K, V, attn_mask=mask)

    tQ = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tK = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tV = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tM = ttnn.from_torch(mask, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(scaled_dot_product_attention(tQ, tK, tV, attn_mask=tM))

    assert not torch.isnan(out.float()).any(), "NaN in output"
    pcc = _pcc(out, exp)
    assert pcc >= _PCC[dtype], f"PCC {pcc:.5f} < {_PCC[dtype]}"


# Causal requires S_q == S_kv (self-attn); use the square non-aligned shapes.
_CAUSAL_SHAPES = [
    pytest.param((1, 1, 47, 64), id="h_S47"),
    pytest.param((1, 1, 50, 50), id="both_S50_D50"),
    pytest.param((1, 12, 33, 50), id="both_S33_D50_multihead"),
    pytest.param((2, 4, 100, 64), id="h_S100_multibatch"),
]


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("shape", _CAUSAL_SHAPES)
def test_non_aligned_causal(device, shape, dtype):
    """On-device causal mask + non-aligned: the triangular bias already covers
    padded columns for valid rows, but the kv-pad mask composes on the
    diagonal block of the last query block."""
    torch_dtype = _TORCH_DTYPE[dtype]
    torch.manual_seed(0)
    Q = torch.randn(shape, dtype=torch_dtype)
    K = torch.randn(shape, dtype=torch_dtype)
    V = torch.randn(shape, dtype=torch_dtype)
    exp = _ref(Q, K, V, is_causal=True)

    tQ = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tK = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tV = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(scaled_dot_product_attention(tQ, tK, tV, is_causal=True))

    assert not torch.isnan(out.float()).any(), "NaN in output"
    pcc = _pcc(out, exp)
    assert pcc >= _PCC[dtype], f"PCC {pcc:.5f} < {_PCC[dtype]}"


@pytest.mark.parametrize(
    "q_shape,k_shape",
    [
        pytest.param((1, 1, 32, 50), (1, 1, 32, 50), id="w_D50"),
        pytest.param((1, 1, 47, 64), (1, 1, 47, 64), id="h_S47"),
    ],
)
def test_bfloat8b_non_aligned_excluded(device, q_shape, k_shape):
    """bfloat8_b on a partial last tile is an EXCLUSION: the block-float
    shared face exponent is computed (by from_torch) over the garbage padding
    and cannot be repaired post-tilize. validate() must refuse it."""
    torch.manual_seed(0)
    Q = torch.randn(q_shape, dtype=torch.bfloat16)
    K = torch.randn(k_shape, dtype=torch.bfloat16)
    V = torch.randn(k_shape, dtype=torch.bfloat16)
    tQ = ttnn.from_torch(Q, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    tK = ttnn.from_torch(K, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    tV = ttnn.from_torch(V, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    with pytest.raises(NotImplementedError):
        scaled_dot_product_attention(tQ, tK, tV)
