# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
#
# SPDX-License-Identifier: Apache-2.0

"""Refinement 2 — GQA / MQA (kv_heads_mode) direct tests.

Exercises grouped-query (1 < H_kv < H_q, H_q % H_kv == 0) and multi-query
(H_kv == 1) attention. The reader already remaps each Q head to its grouped
KV head; this suite confirms the runtime gate now accepts gqa/mqa and the
numerics are correct across:
  - realistic LLM head ratios (8:2, 32:8, 32:1, 12:4, …),
  - self- and cross-attention,
  - none / causal masks (broadcast and per-head),
  - auto / explicit scale,
  - supported dtypes {bf16, fp32, bf8b}.

It also confirms the structural gate still rejects H_q % H_kv != 0 and that
MHA (H_q == H_kv) is unaffected.
"""

import math

import pytest
import torch

import ttnn

from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


# ---------------------------------------------------------------------------
# Reference (fp32) with explicit GQA/MQA head broadcast
# ---------------------------------------------------------------------------


def _ref_sdpa(Q, K, V, *, attention_mask=None, scale=None):
    Qf, Kf, Vf = Q.float(), K.float(), V.float()
    H_q, H_kv = Qf.shape[1], Kf.shape[1]
    if H_q != H_kv:
        r = H_q // H_kv
        Kf = Kf.repeat_interleave(r, dim=1)
        Vf = Vf.repeat_interleave(r, dim=1)
    s = scale if scale is not None else 1.0 / math.sqrt(Qf.shape[-1])
    scores = torch.matmul(Qf, Kf.transpose(-2, -1)) * s
    if attention_mask is not None:
        scores = scores + attention_mask.float()
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, Vf).to(Q.dtype)


_TORCH_DTYPE = {
    ttnn.bfloat16: torch.bfloat16,
    ttnn.float32: torch.float32,
    ttnn.bfloat8_b: torch.bfloat16,
}

# (pcc, rms) per dtype — mirror golden TOLERANCES.
_TOL = {
    ttnn.bfloat16: (0.995, 0.05),
    ttnn.float32: (0.999, 0.02),
    ttnn.bfloat8_b: (0.99, 0.12),
}


def _to_dev(t, device, dtype):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _check(out_ttnn, expected, dtype):
    out = ttnn.to_torch(out_ttnn).float()
    exp = expected.float()
    pcc_min, rms_max = _TOL[dtype]
    # PCC
    a = out.flatten()
    b = exp.flatten()
    a = a - a.mean()
    b = b - b.mean()
    pcc = (a @ b) / (a.norm() * b.norm() + 1e-12)
    # relative RMS
    rms = (out - exp).pow(2).mean().sqrt() / (exp.std() + 1e-12)
    assert pcc >= pcc_min, f"PCC {pcc:.5f} < {pcc_min} (rms={rms:.4f})"
    assert rms <= rms_max, f"rel-RMS {rms:.4f} > {rms_max} (pcc={pcc:.5f})"
    return float(pcc), float(rms)


# ---------------------------------------------------------------------------
# Core GQA / MQA forward — realistic LLM ratios, self-attention
# ---------------------------------------------------------------------------

# (Q_shape, K_shape, V_shape, label)
_GQA_MQA_CONFIGS = [
    ((1, 8, 128, 64), (1, 2, 128, 64), (1, 2, 128, 64), "gqa_4to1"),
    ((1, 12, 128, 64), (1, 4, 128, 64), (1, 4, 128, 64), "gqa_3to1"),
    ((1, 16, 256, 64), (1, 4, 256, 64), (1, 4, 256, 64), "gqa_4to1_long"),
    ((1, 32, 128, 128), (1, 8, 128, 128), (1, 8, 128, 128), "gqa_llama3"),
    ((2, 8, 128, 64), (2, 2, 128, 64), (2, 2, 128, 64), "gqa_batched"),
    ((1, 8, 128, 64), (1, 1, 128, 64), (1, 1, 128, 64), "mqa_8to1"),
    ((1, 12, 128, 64), (1, 1, 128, 64), (1, 1, 128, 64), "mqa_12to1"),
    ((1, 32, 128, 128), (1, 1, 128, 128), (1, 1, 128, 128), "mqa_large"),
    ((2, 8, 128, 64), (2, 1, 128, 64), (2, 1, 128, 64), "mqa_batched"),
]
_GQA_MQA_IDS = [c[3] for c in _GQA_MQA_CONFIGS]


@pytest.mark.parametrize("q_shape,k_shape,v_shape,label", _GQA_MQA_CONFIGS, ids=_GQA_MQA_IDS)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b], ids=["bf16", "fp32", "bf8b"])
def test_gqa_mqa_forward(q_shape, k_shape, v_shape, label, dtype, device):
    torch.manual_seed(0)
    td = _TORCH_DTYPE[dtype]
    Q = torch.randn(q_shape, dtype=td)
    K = torch.randn(k_shape, dtype=td)
    V = torch.randn(v_shape, dtype=td)

    expected = _ref_sdpa(Q, K, V)
    out = scaled_dot_product_attention(_to_dev(Q, device, dtype), _to_dev(K, device, dtype), _to_dev(V, device, dtype))
    assert list(out.shape) == list(q_shape)
    assert out.dtype == dtype
    _check(out, expected, dtype)


# ---------------------------------------------------------------------------
# GQA / MQA with mask + explicit scale (bf16)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q_shape,k_shape,v_shape,label", _GQA_MQA_CONFIGS, ids=_GQA_MQA_IDS)
def test_gqa_mqa_causal_mask(q_shape, k_shape, v_shape, label, device):
    """Broadcast (B,1,S,S) causal mask + explicit scale."""
    torch.manual_seed(1)
    B, H, S_q, D = q_shape
    S_kv = k_shape[-2]
    Q = torch.randn(q_shape, dtype=torch.bfloat16)
    K = torch.randn(k_shape, dtype=torch.bfloat16)
    V = torch.randn(v_shape, dtype=torch.bfloat16)
    mask = torch.zeros(B, 1, S_q, S_kv, dtype=torch.bfloat16)
    mask.masked_fill_(torch.triu(torch.ones(S_q, S_kv, dtype=torch.bool), diagonal=1), float("-inf"))

    expected = _ref_sdpa(Q, K, V, attention_mask=mask, scale=0.125)
    out = scaled_dot_product_attention(
        _to_dev(Q, device, ttnn.bfloat16),
        _to_dev(K, device, ttnn.bfloat16),
        _to_dev(V, device, ttnn.bfloat16),
        attention_mask=_to_dev(mask, device, ttnn.bfloat16),
        scale=0.125,
    )
    _check(out, expected, ttnn.bfloat16)


@pytest.mark.parametrize("q_shape,k_shape,v_shape,label", _GQA_MQA_CONFIGS, ids=_GQA_MQA_IDS)
def test_gqa_mqa_per_head_mask(q_shape, k_shape, v_shape, label, device):
    """Per-head (B,H,S,S) mask — mask_H == H_q (Q heads), not H_kv. Confirms
    the reader's mask head index keys off the Q head."""
    torch.manual_seed(2)
    B, H, S_q, D = q_shape
    S_kv = k_shape[-2]
    Q = torch.randn(q_shape, dtype=torch.bfloat16)
    K = torch.randn(k_shape, dtype=torch.bfloat16)
    V = torch.randn(v_shape, dtype=torch.bfloat16)
    mask = torch.zeros(B, H, S_q, S_kv, dtype=torch.bfloat16)
    mask.masked_fill_(torch.rand(B, H, S_q, S_kv) < 0.3, float("-inf"))

    expected = _ref_sdpa(Q, K, V, attention_mask=mask)
    out = scaled_dot_product_attention(
        _to_dev(Q, device, ttnn.bfloat16),
        _to_dev(K, device, ttnn.bfloat16),
        _to_dev(V, device, ttnn.bfloat16),
        attention_mask=_to_dev(mask, device, ttnn.bfloat16),
    )
    _check(out, expected, ttnn.bfloat16)


# ---------------------------------------------------------------------------
# GQA / MQA cross-attention (S_q != S_kv)
# ---------------------------------------------------------------------------

_CROSS_CONFIGS = [
    ((1, 8, 64, 64), (1, 2, 128, 64), (1, 2, 128, 64), "gqa_cross_sq_lt_skv"),
    ((1, 32, 128, 128), (1, 8, 512, 128), (1, 8, 512, 128), "gqa_cross_sq_ll_skv"),
    ((1, 8, 64, 64), (1, 1, 128, 64), (1, 1, 128, 64), "mqa_cross"),
]
_CROSS_IDS = [c[3] for c in _CROSS_CONFIGS]


@pytest.mark.parametrize("q_shape,k_shape,v_shape,label", _CROSS_CONFIGS, ids=_CROSS_IDS)
def test_gqa_mqa_cross_attention(q_shape, k_shape, v_shape, label, device):
    torch.manual_seed(3)
    Q = torch.randn(q_shape, dtype=torch.bfloat16)
    K = torch.randn(k_shape, dtype=torch.bfloat16)
    V = torch.randn(v_shape, dtype=torch.bfloat16)

    expected = _ref_sdpa(Q, K, V)
    out = scaled_dot_product_attention(
        _to_dev(Q, device, ttnn.bfloat16),
        _to_dev(K, device, ttnn.bfloat16),
        _to_dev(V, device, ttnn.bfloat16),
    )
    assert list(out.shape) == list(q_shape)
    _check(out, expected, ttnn.bfloat16)


# ---------------------------------------------------------------------------
# Structural gate still enforced
# ---------------------------------------------------------------------------


def test_non_divisible_heads_rejected(device):
    """H_q not a multiple of H_kv must raise (structural, not a support refusal)."""
    Q = torch.randn(1, 6, 128, 64, dtype=torch.bfloat16)
    K = torch.randn(1, 4, 128, 64, dtype=torch.bfloat16)  # 6 % 4 != 0
    V = torch.randn(1, 4, 128, 64, dtype=torch.bfloat16)
    with pytest.raises((ValueError, RuntimeError)):
        scaled_dot_product_attention(
            _to_dev(Q, device, ttnn.bfloat16),
            _to_dev(K, device, ttnn.bfloat16),
            _to_dev(V, device, ttnn.bfloat16),
        )


def test_mha_still_works(device):
    """Regression guard: MHA (H_q == H_kv) unaffected by the gate change."""
    torch.manual_seed(4)
    shape = (1, 4, 128, 64)
    Q = torch.randn(shape, dtype=torch.bfloat16)
    K = torch.randn(shape, dtype=torch.bfloat16)
    V = torch.randn(shape, dtype=torch.bfloat16)
    expected = _ref_sdpa(Q, K, V)
    out = scaled_dot_product_attention(
        _to_dev(Q, device, ttnn.bfloat16),
        _to_dev(K, device, ttnn.bfloat16),
        _to_dev(V, device, ttnn.bfloat16),
    )
    _check(out, expected, ttnn.bfloat16)
