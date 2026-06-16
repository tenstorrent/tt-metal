# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 2 tests — attention variants: causal masking + GQA / MQA.

Exercises the two attention-semantic additions:
  - native causal masking (mask_mode="causal"): triangular −inf bias is
    generated on-device from is_causal (no caller tensor). Compared against
    the additive-mask reference (the same triangular pattern via attn_mask).
  - GQA / MQA (kv_heads_mode in {gqa, mqa}): K/V have fewer heads than Q;
    the reader head-broadcasts via h_kv = h // group.

Also covers the validate() contract: causal+cross EXCLUSION (NotImplementedError)
and the is_causal + attn_mask mutual-exclusion ValueError.
"""

import math

import pytest
import torch
import ttnn

from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

PCC_TOLERANCE = 0.995  # bfloat16


def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def reference_sdpa(Q, K, V, *, attn_mask=None, is_causal=False, scale=None):
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
        tri = torch.triu(torch.ones(S_q, S_kv, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(tri, float("-inf"))
    elif attn_mask is not None:
        scores = scores + attn_mask.float()
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, Vf).to(Q.dtype)


def _to_dev(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


# ---------------------------------------------------------------------------
# Causal masking (self-attention, S_q == S_kv)
# ---------------------------------------------------------------------------

_CAUSAL_SHAPES = [
    pytest.param((1, 1, 32, 32), id="single_tile"),
    pytest.param((1, 1, 128, 64), id="multi_tile"),
    pytest.param((1, 1, 256, 64), id="longer"),
    pytest.param((1, 4, 128, 64), id="multi_head"),
    pytest.param((2, 4, 128, 64), id="multi_batch"),
    pytest.param((1, 8, 256, 64), id="multi_head_long"),
    pytest.param((1, 1, 128, 128), id="wide_d"),
]

_SCALE_MODES = [pytest.param("auto", id="scale_auto"), pytest.param("explicit", id="scale_explicit")]
EXPLICIT_SCALE = 0.125


@pytest.mark.parametrize("shape", _CAUSAL_SHAPES)
@pytest.mark.parametrize("scale_mode", _SCALE_MODES)
def test_causal_self_attention(device, shape, scale_mode):
    B, H, S, D = shape
    torch.manual_seed(42)
    Q = torch.randn((B, H, S, D), dtype=torch.bfloat16)
    K = torch.randn((B, H, S, D), dtype=torch.bfloat16)
    V = torch.randn((B, H, S, D), dtype=torch.bfloat16)

    scale = EXPLICIT_SCALE if scale_mode == "explicit" else None
    expected = reference_sdpa(Q, K, V, is_causal=True, scale=scale)

    out = scaled_dot_product_attention(
        _to_dev(Q, device), _to_dev(K, device), _to_dev(V, device), is_causal=True, scale=scale
    )
    assert list(out.shape) == [B, H, S, D]
    out_t = ttnn.to_torch(out)
    corr = pcc(out_t, expected)
    assert corr >= PCC_TOLERANCE, f"causal PCC too low: {corr:.6f}"


def test_causal_matches_additive_triangular(device):
    """Native causal must match the equivalent additive triangular mask."""
    B, H, S, D = 1, 2, 128, 64
    torch.manual_seed(7)
    Q = torch.randn((B, H, S, D), dtype=torch.bfloat16)
    K = torch.randn((B, H, S, D), dtype=torch.bfloat16)
    V = torch.randn((B, H, S, D), dtype=torch.bfloat16)

    mask = torch.zeros(B, 1, S, S, dtype=torch.bfloat16)
    mask.masked_fill_(torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1), float("-inf"))

    causal_out = ttnn.to_torch(
        scaled_dot_product_attention(_to_dev(Q, device), _to_dev(K, device), _to_dev(V, device), is_causal=True)
    )
    custom_out = ttnn.to_torch(
        scaled_dot_product_attention(
            _to_dev(Q, device), _to_dev(K, device), _to_dev(V, device), attn_mask=_to_dev(mask, device)
        )
    )
    corr = pcc(causal_out, custom_out)
    assert corr >= 0.999, f"causal vs additive-triangular mismatch: {corr:.6f}"


# ---------------------------------------------------------------------------
# GQA / MQA (validate-only — kernel already head-broadcasts)
# ---------------------------------------------------------------------------

_GQA_CONFIGS = [
    pytest.param((1, 8, 128, 64), (1, 2, 128, 64), id="gqa_4to1"),
    pytest.param((1, 32, 128, 128), (1, 8, 128, 128), id="gqa_llama3"),
    pytest.param((1, 12, 128, 64), (1, 4, 128, 64), id="gqa_3to1"),
    pytest.param((1, 8, 256, 64), (1, 1, 256, 64), id="mqa_basic"),
    pytest.param((1, 32, 128, 128), (1, 1, 128, 128), id="mqa_large"),
    pytest.param((2, 8, 128, 64), (2, 2, 128, 64), id="gqa_batch"),
]


@pytest.mark.parametrize("q_shape,k_shape", _GQA_CONFIGS)
def test_gqa_mqa_self_attention(device, q_shape, k_shape):
    torch.manual_seed(123)
    Q = torch.randn(q_shape, dtype=torch.bfloat16)
    K = torch.randn(k_shape, dtype=torch.bfloat16)
    V = torch.randn(k_shape, dtype=torch.bfloat16)

    expected = reference_sdpa(Q, K, V)
    out = scaled_dot_product_attention(_to_dev(Q, device), _to_dev(K, device), _to_dev(V, device))
    assert list(out.shape) == list(q_shape)
    corr = pcc(ttnn.to_torch(out), expected)
    assert corr >= PCC_TOLERANCE, f"GQA/MQA PCC too low: {corr:.6f}"


def test_gqa_with_custom_mask(device):
    """GQA + additive mask together (both touch the reader head/mask index)."""
    q_shape, k_shape = (1, 8, 128, 64), (1, 2, 128, 64)
    B, H, S, D = q_shape
    torch.manual_seed(5)
    Q = torch.randn(q_shape, dtype=torch.bfloat16)
    K = torch.randn(k_shape, dtype=torch.bfloat16)
    V = torch.randn(k_shape, dtype=torch.bfloat16)
    mask = torch.zeros(B, 1, S, S, dtype=torch.bfloat16)
    mask.masked_fill_(torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1), float("-inf"))

    expected = reference_sdpa(Q, K, V, attn_mask=mask)
    out = scaled_dot_product_attention(
        _to_dev(Q, device), _to_dev(K, device), _to_dev(V, device), attn_mask=_to_dev(mask, device)
    )
    corr = pcc(ttnn.to_torch(out), expected)
    assert corr >= PCC_TOLERANCE, f"GQA+mask PCC too low: {corr:.6f}"


# ---------------------------------------------------------------------------
# validate() contract
# ---------------------------------------------------------------------------


def test_causal_cross_rejected(device):
    """causal + cross-attention (S_q != S_kv) is an EXCLUSION -> NotImplementedError."""
    Q = torch.randn((1, 4, 64, 64), dtype=torch.bfloat16)
    K = torch.randn((1, 4, 128, 64), dtype=torch.bfloat16)
    V = torch.randn((1, 4, 128, 64), dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError):
        scaled_dot_product_attention(_to_dev(Q, device), _to_dev(K, device), _to_dev(V, device), is_causal=True)


def test_causal_and_mask_mutually_exclusive(device):
    Q = torch.randn((1, 1, 128, 64), dtype=torch.bfloat16)
    K = torch.randn((1, 1, 128, 64), dtype=torch.bfloat16)
    V = torch.randn((1, 1, 128, 64), dtype=torch.bfloat16)
    mask = torch.zeros(1, 1, 128, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        scaled_dot_product_attention(
            _to_dev(Q, device),
            _to_dev(K, device),
            _to_dev(V, device),
            is_causal=True,
            attn_mask=_to_dev(mask, device),
        )
