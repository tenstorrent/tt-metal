# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0
#
# Refinement 3 — Non-tile-aligned shapes (alignment w / h).
#
# SDPA stays TILE-layout; only the alignment axis expands. Padding is handled
# in-kernel (no ttnn.to_layout / tilize wrapper):
#   - w_non_aligned (D not %32): from_torch zero-pads the padded D-columns, so
#     they contribute 0 to Q·Kᵀ (contraction) and the padded D-columns of the PV
#     output are zero and dropped on read-back (logical-shape slice).
#   - h_non_aligned (S_q not %32): padded query rows are zero → per-row math is
#     independent; the writer's padded rows are dropped on read-back.
#   - S_kv not %32 (the structural edge): the padded KEY columns of the last KV
#     tile MUST become −inf BEFORE the softmax row-max / row-sum, otherwise the
#     denominator over-counts the padded keys (the failure this refinement fixes).
#     The reader generates an additive −inf key-padding mask added to the last
#     KV block's scores.
#
# This is the authoritative alignment-correctness test. The golden suite also
# covers these via the alignment axis; this file pins specific edge shapes and
# asserts directly against the PyTorch reference.

import math

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


# Per-dtype (pcc, rms-rel) thresholds — mirror eval/golden_tests/.../helpers.py
# TOLERANCES (SDPA chains matmul→softmax→matmul so error compounds).
_TOL = {
    ttnn.bfloat16: (0.995, 0.05),
    ttnn.float32: (0.999, 0.02),
    ttnn.bfloat8_b: (0.99, 0.12),
}
_TORCH_DTYPE = {
    ttnn.bfloat16: torch.bfloat16,
    ttnn.float32: torch.float32,
    ttnn.bfloat8_b: torch.bfloat16,  # no native torch bf8b; reference in bf16
}
_EXPLICIT_SCALE = 0.125


def _ref(Q, K, V, *, mask=None, scale=None):
    Qf, Kf, Vf = Q.float(), K.float(), V.float()
    H_q, H_kv = Qf.shape[1], Kf.shape[1]
    if H_q != H_kv:
        r = H_q // H_kv
        Kf, Vf = Kf.repeat_interleave(r, 1), Vf.repeat_interleave(r, 1)
    s = scale if scale is not None else 1.0 / math.sqrt(Qf.shape[-1])
    scores = (Qf @ Kf.transpose(-2, -1)) * s
    if mask is not None:
        scores = scores + mask.float()
    return (torch.softmax(scores, dim=-1) @ Vf).to(Q.dtype)


def _rel_rms(out, exp):
    out, exp = out.float(), exp.float()
    denom = exp.std().item()
    if denom == 0.0:
        denom = 1.0
    return ((out - exp).pow(2).mean().sqrt().item()) / denom


def _make_causal(B, S_q, S_kv, torch_dtype):
    mask = torch.zeros(B, 1, S_q, S_kv, dtype=torch_dtype)
    mask.masked_fill_(torch.triu(torch.ones(S_q, S_kv, dtype=torch.bool), diagonal=1), float("-inf"))
    return mask


def _run(shapes, *, dtype, mask_mode, scale_mode, device):
    from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

    q_shape, k_shape, v_shape = shapes
    torch_dtype = _TORCH_DTYPE[dtype]
    torch.manual_seed(0)
    Q = torch.randn(q_shape, dtype=torch_dtype)
    K = torch.randn(k_shape, dtype=torch_dtype)
    V = torch.randn(v_shape, dtype=torch_dtype)

    B, _H, S_q, _D = q_shape
    S_kv = k_shape[-2]
    torch_mask = _make_causal(B, S_q, S_kv, torch_dtype) if mask_mode == "causal" else None
    scale = _EXPLICIT_SCALE if scale_mode == "explicit" else None

    expected = _ref(Q, K, V, mask=torch_mask, scale=scale)

    def to_tt(t):
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device) if t is not None else None

    out = scaled_dot_product_attention(to_tt(Q), to_tt(K), to_tt(V), attention_mask=to_tt(torch_mask), scale=scale)
    res = ttnn.to_torch(out)

    assert tuple(res.shape) == tuple(q_shape), f"shape {tuple(res.shape)} != {tuple(q_shape)}"
    pcc_t, rms_t = _TOL[dtype]
    rms = _rel_rms(res, expected)
    assert_with_pcc(expected.float(), res.float(), pcc_t)
    assert rms <= rms_t, f"rel-RMS {rms:.4f} > target {rms_t} (dtype={dtype}, shape={q_shape})"


# (id, Q, K, V) — curated alignment edge shapes. Each names which non-aligned
# axis it exercises (w=D, h=S_q, and whether S_kv is also non-aligned).
_SHAPES = [
    # w_non_aligned (D not %32), S_kv aligned — isolates the D-padding edge.
    ("w_D50_Skv32", (1, 1, 32, 50), (1, 1, 32, 50), (1, 1, 32, 50)),
    ("w_D47_Skv64", (1, 8, 64, 47), (1, 8, 64, 47), (1, 8, 64, 47)),
    # w_non_aligned (D not %32) + S_kv non-aligned — D edge AND key-pad edge.
    ("w_D50_Skv50", (1, 1, 50, 50), (1, 1, 50, 50), (1, 1, 50, 50)),
    ("w_D50_Skv33_mh", (1, 12, 33, 50), (1, 12, 33, 50), (1, 12, 33, 50)),
    # h_non_aligned (S_q not %32) + S_kv non-aligned (the structural edge).
    ("h_Sq47_Skv47", (1, 1, 47, 64), (1, 1, 47, 64), (1, 1, 47, 64)),
    ("h_Sq47_mh", (1, 4, 47, 64), (1, 4, 47, 64), (1, 4, 47, 64)),
    ("h_Sq100_mb", (2, 4, 100, 64), (2, 4, 100, 64), (2, 4, 100, 64)),
    # non-aligned + GQA / MQA (head remap × key-pad mask).
    ("h_Sq47_gqa", (1, 8, 47, 64), (1, 2, 47, 64), (1, 2, 47, 64)),
    ("h_Sq47_mqa", (1, 8, 47, 64), (1, 1, 47, 64), (1, 1, 47, 64)),
    # both non-aligned + cross-attention (S_q != S_kv, both non-aligned).
    ("both_cross", (1, 4, 100, 50), (1, 4, 47, 50), (1, 4, 47, 50)),
]


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b], ids=["bf16", "fp32", "bf8b"])
@pytest.mark.parametrize("mask_mode", ["none", "causal"])
@pytest.mark.parametrize("scale_mode", ["auto", "explicit"])
@pytest.mark.parametrize("name,q,k,v", _SHAPES, ids=[s[0] for s in _SHAPES])
def test_alignment_matrix(device, name, q, k, v, dtype, mask_mode, scale_mode):
    """Full alignment × dtype × mask × scale matrix for the non-aligned edges."""
    _run((q, k, v), dtype=dtype, mask_mode=mask_mode, scale_mode=scale_mode, device=device)


@pytest.mark.parametrize("S_kv", [47, 33, 50, 65, 100], ids=lambda s: f"Skv{s}")
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
def test_skv_keypad_denominator(device, S_kv, dtype):
    """Directly target the key-padding edge: S_q and D aligned, only S_kv varies
    non-aligned. Without the −inf key-pad mask the softmax denominator over-counts
    the padded keys (rms ~0.19); with it, the output matches the reference. Compares
    a no-mask case (where the over-count is purely the padded keys, most diagnostic).
    """
    _run(
        ((1, 2, 64, 64), (1, 2, S_kv, 64), (1, 2, S_kv, 64)),
        dtype=dtype,
        mask_mode="none",
        scale_mode="auto",
        device=device,
    )
