# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 1 — non-tile-aligned shapes (w_non_aligned + h_non_aligned).

DO NOT DELETE — documents the R1 debugging/verification surface.

Three structural legs, all TILE layout (SDPA is TILE-only):
  * w_non_aligned  (D % 32 != 0): rides input tile zero-padding through the
    Q·Kᵀ contraction (over Dt) and the P·V free dim; output D-pad columns are
    written as whole tiles and sliced off by the logical shape.
  * h_non_aligned  (S_q % 32 != 0): the last Q-chunk's padding rows produce
    finite (discarded) output — whole-tile write + logical slice.
  * S_kv % 32 != 0: the last KV tile's padding columns are masked to −∞ before
    the softmax row-max/exp/row-sum so they fall out of the denominator. Keyed
    on S_kv (K.shape[-2]) independent of the Q-derived alignment tag.

Reference matches torch F.sdpa (fp32) via the existing unit-test helper.
"""

import math

import pytest
import torch

import ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

# Golden bf16 + fp32-DEST tolerances (helpers.TOLERANCES[(bfloat16, True)]).
# PCC alone is scale-invariant and would mask a softmax-denominator error from
# unmasked KV padding — the normalized-RMS gate is what actually catches it.
PCC = 0.995
REL_RMS = 0.05


def _check(ref, out):
    ref = ref.float()
    out = out.float()
    # PCC
    rf, of = ref.flatten(), out.flatten()
    rf = rf - rf.mean()
    of = of - of.mean()
    pcc = (rf @ of / (rf.norm() * of.norm() + 1e-12)).item()
    # normalized RMS (RMS(ref-out) / std(ref)) — the golden `rms` metric.
    rel_rms = (torch.sqrt(((ref - out) ** 2).mean()) / (ref.std() + 1e-12)).item()
    assert pcc >= PCC, f"PCC {pcc:.5f} < {PCC}  (rel_rms={rel_rms:.4f})"
    assert rel_rms <= REL_RMS, f"rel_rms {rel_rms:.4f} > {REL_RMS}  (pcc={pcc:.5f})"


def attention_reference(q, k, v, attn_mask=None, scale=None):
    B, H, Sq, D = q.shape
    Hkv = k.shape[1]
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    q, k, v = q.float(), k.float(), v.float()
    if Hkv != H:
        rep = H // Hkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if attn_mask is not None:
        scores = scores + attn_mask.float()
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v)


def _to_device(t, device, dtype):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def run_sdpa(device, q_shape, k_shape, v_shape, *, mask_mode="none", scale_mode="auto"):
    torch.manual_seed(42)
    dtype = ttnn.bfloat16
    q = torch.randn(q_shape)
    k = torch.randn(k_shape)
    v = torch.randn(v_shape)

    scale = None if scale_mode == "auto" else 0.25

    attn_mask = None
    tt_mask = None
    if mask_mode == "custom":
        B, _, Sq, _ = q_shape
        Skv = k_shape[-2]
        attn_mask = torch.randn(B, 1, Sq, Skv) * 2.0
        tt_mask = _to_device(attn_mask, device, dtype)

    ref = attention_reference(q, k, v, attn_mask=attn_mask, scale=scale)

    tq = _to_device(q, device, dtype)
    tk = _to_device(k, device, dtype)
    tv = _to_device(v, device, dtype)

    out = scaled_dot_product_attention(tq, tk, tv, attn_mask=tt_mask, scale=scale)
    assert list(out.shape) == list(q_shape), f"shape {list(out.shape)} != {list(q_shape)}"
    _check(ref, ttnn.to_torch(out))


# Pure w_non_aligned: D%32!=0, S_q AND S_kv tile-aligned. Should pass on the
# zero-padding alone (no softmax mask needed).
W_ONLY = {
    "w_D50_S32": ((1, 1, 32, 50), (1, 1, 32, 50), (1, 1, 32, 50)),
    "w_D47_S64_mh": ((1, 8, 64, 47), (1, 8, 64, 47), (1, 8, 64, 47)),
}

# S_kv % 32 != 0 — requires the last-KV-tile −∞ mask.
SKV_PARTIAL = {
    "h_S47_D64": ((1, 1, 47, 64), (1, 1, 47, 64), (1, 1, 47, 64)),
    "h_S100_D64_batch": ((2, 4, 100, 64), (2, 4, 100, 64), (2, 4, 100, 64)),
    "h_S47_D64_mh": ((1, 4, 47, 64), (1, 4, 47, 64), (1, 4, 47, 64)),
    "both_S50_D50": ((1, 1, 50, 50), (1, 1, 50, 50), (1, 1, 50, 50)),
    "both_S33_D50_mh": ((1, 12, 33, 50), (1, 12, 33, 50), (1, 12, 33, 50)),
    "gqa_S47": ((1, 8, 47, 64), (1, 2, 47, 64), (1, 2, 47, 64)),
    "mqa_S47": ((1, 8, 47, 64), (1, 1, 47, 64), (1, 1, 47, 64)),
    "cross_both": ((1, 4, 100, 50), (1, 4, 47, 50), (1, 4, 47, 50)),
    # Q tile-aligned, K.S_kv non-aligned — isolates the S_kv mask (not a golden
    # cell; the golden taggers only see Q, but the kernel keys on K.S_kv).
    "cross_skv_only": ((1, 1, 64, 64), (1, 1, 47, 64), (1, 1, 47, 64)),
}


@pytest.mark.parametrize("key", list(W_ONLY.keys()))
def test_w_non_aligned(device, key):
    q, k, v = W_ONLY[key]
    run_sdpa(device, q, k, v)


@pytest.mark.parametrize("key", list(SKV_PARTIAL.keys()))
def test_skv_partial_none(device, key):
    q, k, v = SKV_PARTIAL[key]
    run_sdpa(device, q, k, v, mask_mode="none")


@pytest.mark.parametrize("key", list(SKV_PARTIAL.keys()))
def test_skv_partial_custom(device, key):
    q, k, v = SKV_PARTIAL[key]
    run_sdpa(device, q, k, v, mask_mode="custom")
