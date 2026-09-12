# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ESPnet relative-position attention, and whether it fits into flash attention.

This was flagged up front as one of the two hardest risks: both the LLM and the
flow encoder use `rel_selfattn`, not RoPE and not vanilla attention, so the
tt-metal LLM demos under `models/demos/wormhole/` are a structural reference
rather than a drop-in.

The reference computes

    matrix_ac = (q + pos_bias_u) @ k^T
    matrix_bd = rel_shift( (q + pos_bias_v) @ p^T )     p = linear_pos(pos_emb)
    scores    = (matrix_ac + matrix_bd) / sqrt(d_k)
    out       = softmax(scores) @ v

The claim these tests exist to settle: **matrix_bd is an additive bias on the
attention scores**, so the whole thing is ordinary scaled-dot-product attention
with `q' = q + pos_bias_u` and `attn_mask = matrix_bd / sqrt(d_k)`. TTNN's SDPA
already accepts `attn_mask` (sdpa.hpp:20, :170), so if the claim holds, flash
attention needs no new C++ at all.

The scaling is the part that is easy to get wrong and impossible to notice: SDPA
applies `1/sqrt(d_k)` to `q @ k^T` and then adds the mask, whereas the reference
divides the ALREADY-SUMMED `ac + bd`. So the bias handed to SDPA must be
pre-divided. Passing raw `matrix_bd` yields plausible-looking attention that is
quietly wrong by a factor of 8 on this model.

Everything is checked against a golden captured from a real
RelPositionMultiHeadedAttention layer inside the flow encoder, so the reference
math here is not a transcription anyone has to trust.
"""
from __future__ import annotations

import math
import os

import pytest
import torch

from models.demos.cosyvoice.tt.common import as_torch, load_golden, pcc

GOLDEN = "flow.rel_pos_attention"


def _have_golden() -> bool:
    from models.demos.cosyvoice.tt.common import GOLDEN_DIR

    return os.path.exists(os.path.join(GOLDEN_DIR, f"{GOLDEN}.npz"))


needs_golden = pytest.mark.skipif(not _have_golden(), reason="run scripts/gen_golden.py in the CosyVoice venv first")


def _load():
    g = load_golden(GOLDEN)
    w = load_golden(f"{GOLDEN}_weights")
    return g, w


def rel_shift(x: torch.Tensor) -> torch.Tensor:
    """cosyvoice.transformer.attention.RelPositionMultiHeadedAttention.rel_shift.

    A skew of the score matrix: pad a zero column, reinterpret the last two axes
    transposed, drop the first row, then keep the left half. In tile layout this
    is a strided gather, not an elementwise op -- which is why a native rel-pos SDPA
    was scoped as high-risk.
    """
    b, h, t1, n = x.shape
    zero_pad = torch.zeros((b, h, t1, 1), dtype=x.dtype)
    x_padded = torch.cat([zero_pad, x], dim=-1)
    x_padded = x_padded.view(b, h, n + 1, t1)
    return x_padded[:, :, 1:].view_as(x)[:, :, :, : n // 2 + 1]


def reference_scores(q, k, p, pos_bias_u, pos_bias_v, d_k):
    """The reference's own formulation: sum first, then scale."""
    q_u = (q + pos_bias_u).transpose(1, 2)  # [B, H, T, d_k]
    q_v = (q + pos_bias_v).transpose(1, 2)
    ac = torch.matmul(q_u, k.transpose(-2, -1))
    bd = torch.matmul(q_v, p.transpose(-2, -1))
    if ac.shape != bd.shape:
        bd = rel_shift(bd)
    return (ac + bd) / math.sqrt(d_k)


# --------------------------------------------------------------------------
# the claim
# --------------------------------------------------------------------------
@needs_golden
def test_rel_pos_term_is_an_additive_sdpa_bias():
    """(ac + bd)/sqrt(d) == ac/sqrt(d) + bd/sqrt(d), i.e. bd/sqrt(d) is exactly
    the additive mask SDPA already accepts. This is the whole of sec.3.3's route
    to flash attention, stated as an identity and checked as one."""
    g, w = _load()
    d_k = int(w["call0.d_k"])
    n_head = int(w["call0.n_head"])

    query = as_torch(g["call0.in_query"])
    pos_emb = as_torch(g["call0.in_pos_emb"])
    b, t, _ = query.shape

    def proj(name):
        return torch.nn.functional.linear(query, as_torch(w[f"call0.w_{name}"]), as_torch(w[f"call0.b_{name}"]))

    q = proj("query").view(b, t, n_head, d_k)
    k = proj("key").view(b, t, n_head, d_k).transpose(1, 2)
    p = torch.nn.functional.linear(pos_emb, as_torch(w["call0.w_pos"]))
    p = p.view(pos_emb.shape[0], -1, n_head, d_k).transpose(1, 2)
    u, v_ = as_torch(w["call0.pos_bias_u"]), as_torch(w["call0.pos_bias_v"])

    want = reference_scores(q, k, p, u, v_, d_k)

    # The SDPA formulation: scale q@k^T, then ADD a pre-scaled bias.
    q_u = (q + u).transpose(1, 2)
    q_v = (q + v_).transpose(1, 2)
    ac = torch.matmul(q_u, k.transpose(-2, -1))
    bd = torch.matmul(q_v, p.transpose(-2, -1))
    if ac.shape != bd.shape:
        bd = rel_shift(bd)
    got = ac / math.sqrt(d_k) + bd / math.sqrt(d_k)

    print(f"\n  scores {tuple(want.shape)}  PCC {pcc(got, want):.12f}  max|d| {(got - want).abs().max():.3e}")
    assert torch.allclose(got, want, atol=1e-4), (got - want).abs().max()


@needs_golden
def test_forgetting_to_scale_the_bias_is_detectably_wrong():
    """Guards the trap the module docstring names: handing SDPA the RAW matrix_bd
    instead of bd/sqrt(d_k) produces attention that still looks reasonable. If
    this ever starts passing, the scaling has been silently dropped."""
    g, w = _load()
    d_k = int(w["call0.d_k"])
    n_head = int(w["call0.n_head"])

    query = as_torch(g["call0.in_query"])
    pos_emb = as_torch(g["call0.in_pos_emb"])
    b, t, _ = query.shape

    def proj(name):
        return torch.nn.functional.linear(query, as_torch(w[f"call0.w_{name}"]), as_torch(w[f"call0.b_{name}"]))

    q = proj("query").view(b, t, n_head, d_k)
    k = proj("key").view(b, t, n_head, d_k).transpose(1, 2)
    p = torch.nn.functional.linear(pos_emb, as_torch(w["call0.w_pos"]))
    p = p.view(pos_emb.shape[0], -1, n_head, d_k).transpose(1, 2)
    u, v_ = as_torch(w["call0.pos_bias_u"]), as_torch(w["call0.pos_bias_v"])

    want = reference_scores(q, k, p, u, v_, d_k)
    q_u, q_v = (q + u).transpose(1, 2), (q + v_).transpose(1, 2)
    ac = torch.matmul(q_u, k.transpose(-2, -1))
    bd = torch.matmul(q_v, p.transpose(-2, -1))
    if ac.shape != bd.shape:
        bd = rel_shift(bd)
    unscaled = ac / math.sqrt(d_k) + bd  # the bug

    assert not torch.allclose(unscaled, want, atol=1e-2), "unscaled bias should NOT match"
    print(f"\n  unscaled-bias error: max|d| {(unscaled - want).abs().max():.3f} (sqrt(d_k) = {math.sqrt(d_k):.1f})")


# --------------------------------------------------------------------------
# end-to-end against the captured layer
# --------------------------------------------------------------------------
@needs_golden
def test_reference_math_reproduces_the_captured_layer():
    """Our transcription of the whole attention layer must reproduce what the real
    RelPositionMultiHeadedAttention emitted -- otherwise every claim above is
    checked against our own misunderstanding."""
    g, w = _load()
    d_k = int(w["call0.d_k"])
    n_head = int(w["call0.n_head"])

    query = as_torch(g["call0.in_query"])
    key = as_torch(g["call0.in_key"])
    value = as_torch(g["call0.in_value"])
    pos_emb = as_torch(g["call0.in_pos_emb"])
    mask = as_torch(g["call0.in_mask"])
    want = as_torch(g["call0.out_out"])
    b, t, _ = query.shape

    def proj(x, name):
        return torch.nn.functional.linear(x, as_torch(w[f"call0.w_{name}"]), as_torch(w[f"call0.b_{name}"]))

    q = proj(query, "query").view(b, t, n_head, d_k)
    k = proj(key, "key").view(b, -1, n_head, d_k).transpose(1, 2)
    v = proj(value, "value").view(b, -1, n_head, d_k).transpose(1, 2)
    p = torch.nn.functional.linear(pos_emb, as_torch(w["call0.w_pos"]))
    p = p.view(pos_emb.shape[0], -1, n_head, d_k).transpose(1, 2)

    scores = reference_scores(q, k, p, as_torch(w["call0.pos_bias_u"]), as_torch(w["call0.pos_bias_v"]), d_k)

    if mask.numel():
        m = mask.unsqueeze(1).eq(0) if mask.dim() == 3 else mask.eq(0)
        scores = scores.masked_fill(m, -float("inf"))
        attn = torch.softmax(scores, dim=-1).masked_fill(m, 0.0)
    else:
        attn = torch.softmax(scores, dim=-1)

    out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, -1, n_head * d_k)
    got = torch.nn.functional.linear(out, as_torch(w["call0.w_out"]), as_torch(w["call0.b_out"]))

    p_ = pcc(got, want)
    print(
        f"\n  layer output {tuple(got.shape)} vs {tuple(want.shape)}  PCC {p_:.10f}  max|d| {(got - want).abs().max():.3e}"
    )
    assert got.shape == want.shape, (got.shape, want.shape)
    assert p_ >= 0.9999, p_


# --------------------------------------------------------------------------
# device tier
# --------------------------------------------------------------------------
needs_l1_small = pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)


@pytest.mark.parametrize("key_len", [7, 64, 128, 209, 256])
def test_rel_shift_fast_path_only_holds_at_t1_1(key_len):
    """One query row makes the skew a plain slice -- and only one query row.

    `TtRelPosAttention.rel_shift` short-circuits to `x[..., : n // 2 + 1]` when
    `t1 == 1`, which turns seven device ops into one on a path that runs per layer
    per generated token. The justification is that at `t1 == 1` the pad-and-drop is
    its own inverse: prepending a zero to a length-`n` row and dropping the first
    element of the `(n + 1, 1)` reinterpretation gives back the same `n` elements in
    the same order, leaving only the trailing slice.

    The second half of this test is the important half. The identity is **false** for
    `t1 >= 2`, where the skew genuinely permutes, so asserting only the `t1 == 1` case
    would leave a guard that could be widened later without anything failing.
    """
    n = 2 * key_len - 1
    torch.manual_seed(0)

    x1 = torch.randn(1, 4, 1, n)
    assert torch.equal(rel_shift(x1), x1[:, :, :, : n // 2 + 1])

    for t1 in (2, 5):
        xt = torch.randn(1, 4, t1, n)
        skewed = rel_shift(xt)
        sliced = xt[:, :, :, : n // 2 + 1]
        assert skewed.shape == sliced.shape
        assert not torch.equal(skewed, sliced), f"skew is a no-op at t1={t1}; the fast path would be unguarded"


def _have_flow_weights():
    from models.demos.cosyvoice.tt.weights import default_weights_path

    return os.path.exists(default_weights_path().replace("hift_", "flow_"))


needs_flow_weights = pytest.mark.skipif(
    not _have_flow_weights(),
    reason="run scripts/export_weights.py --module flow in the CosyVoice venv first",
)


@needs_golden
@needs_flow_weights
@needs_l1_small
def test_device_rel_pos_attention_matches_golden(device):
    """The attention layer on device, against the captured reference output.

    This is the piece called a structural risk up front, and the one both
    the flow encoder and the LLM depend on -- so it gets checked against real
    weights and real activations rather than random ones.
    """
    import ttnn
    from models.demos.cosyvoice.tt.flow.encoder import TtRelPosAttention
    from models.demos.cosyvoice.tt.weights import WeightBag, default_weights_path

    g = load_golden(GOLDEN)
    query = as_torch(g["call0.in_query"])
    pos_emb = as_torch(g["call0.in_pos_emb"])
    want = as_torch(g["call0.out_out"])

    bag = WeightBag.load(default_weights_path().replace("hift_", "flow_"))
    meta = bag.meta
    attn = TtRelPosAttention(device, bag.sub("encoder.encoders.0.self_attn"), meta["n_head"], meta["d_k"])

    x = ttnn.from_torch(query, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    pe = ttnn.from_torch(pos_emb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    got = ttnn.to_torch(attn(x, pe)).float()

    p = pcc(got, want)
    print(f"\n  rel-pos attention on device: PCC {p:.10f}  max|d| {(got - want).abs().max():.3e}")
    assert got.shape == want.shape, (got.shape, want.shape)
    assert p >= 0.99, p
