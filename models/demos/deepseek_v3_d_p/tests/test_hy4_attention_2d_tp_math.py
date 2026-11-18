# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the 2D tensor-parallel attention layout of the Hy4 SC1 unit (#65885, ``tt/mla/mla_2d.py``).

* Emulating every chip of a ``hp x tp`` mesh with the SAME host layout the device loads
  (``ttMLA2DTP.host_weights_2d`` + ``chip_shard``, i.e. what ``ShardTensor2dMesh`` puts on each chip) and the module's
  collectives in torch reproduces the fp32 HF reference sublayer (``reference/hy4_attention.py``), sink and gate
  included, for several mesh shapes;
* the gate's hidden-axis reduce must precede the sigmoid (summing per-row sigmoids is visibly wrong);
* at Hy4's real dims on the 4x8 SC1 mesh every projection is sharded once over all 32 chips (no replica), with the
  per-chip shapes of the module docstring, and the head ownership of q_b / the gate / o_proj agree;
* the sparse KV push's row-major chip order equals the TP-deduped cache's stripe order only with SP on mesh axis 0
  (the TP=8 PCC drop of #65921, ``ttMLA.push_matches_stripe_order``);
* the config switch keeps ttMLA for every model that does not opt in.
"""

from __future__ import annotations

import types

import pytest
import torch

from models.demos.deepseek_v3_d_p.reference.hy4_attention import (
    Hy4AttnDims,
    apply_rope_half_split,
    hy4_attention,
    mla_latents,
    rope_cos_sin,
    sink_softmax,
)
from models.demos.deepseek_v3_d_p.tt.mla.mla import ttMLA
from models.demos.deepseek_v3_d_p.tt.mla.mla_2d import MLA_2D_TP_FLAG, mla_class, ttMLA2DTP, uses_2d_tp

F32 = torch.float32

# The smallest dims that tile like Hy4 on a 2 x 4 mesh (32-wide tiles, 2 heads per chip).
SMALL = Hy4AttnDims(
    num_heads=16,
    q_lora_rank=256,
    kv_lora_rank=64,
    qk_nope_head_dim=32,
    qk_rope_head_dim=16,
    v_head_dim=32,
    rms_norm_eps=1e-5,
    rope_theta=1e4,
)
SMALL_HIDDEN = 256


def _config(d: Hy4AttnDims, hidden: int, **extra):
    return types.SimpleNamespace(
        num_attention_heads=d.num_heads,
        hidden_size=hidden,
        q_lora_rank=d.q_lora_rank,
        kv_lora_rank=d.kv_lora_rank,
        qk_nope_head_dim=d.qk_nope_head_dim,
        qk_rope_head_dim=d.qk_rope_head_dim,
        v_head_dim=d.v_head_dim,
        rms_norm_eps=d.rms_norm_eps,
        rope_theta=d.rope_theta,
        mla_use_output_gate=True,
        mla_learnable_sink=True,
        rope_scaling=None,
        **extra,
    )


def _hf_weights(d: Hy4AttnDims, hidden: int, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g) / s[-1] ** 0.5  # noqa: E731
    H = d.num_heads
    return {
        "q_a_proj.weight": r(d.q_lora_rank, hidden),
        "q_a_layernorm.weight": 1 + 0.1 * torch.randn(d.q_lora_rank, generator=g),
        "q_b_proj.weight": r(H * d.qk_head_dim, d.q_lora_rank),
        "kv_a_proj_with_mqa.weight": r(d.kv_lora_rank + d.qk_rope_head_dim, hidden),
        "kv_a_layernorm.weight": 1 + 0.1 * torch.randn(d.kv_lora_rank, generator=g),
        "kv_b_proj.weight": r(H * (d.qk_nope_head_dim + d.v_head_dim), d.kv_lora_rank),
        "o_proj.weight": r(hidden, H * d.v_head_dim),
        "linear_gate.weight": r(H * d.v_head_dim, hidden),
        "learnable_sink_param": torch.randn(H, generator=g) * 2,
    }


def _ttmla_names(w: dict) -> dict:
    """HF names -> ttMLA's canonical names (``hy4_checkpoint.hy4_mla_weights`` minus the RoPE frame reorder, which is a
    row permutation of the same tensors; the emulation ropes half-split like the reference)."""
    out = {k: v for k, v in w.items() if k not in ("linear_gate.weight", "learnable_sink_param")}
    out["g_proj.weight"] = w["linear_gate.weight"]
    out["attn_sink.weight"] = w["learnable_sink_param"]
    return out


def _emulate_2d(hn: torch.Tensor, w_hf: dict, topk: torch.Tensor, d: Hy4AttnDims, hp: int, tp: int, gate_order="fused"):
    """The 2D TP sublayer, chip by chip, from the device's own host layout. ``gate_order='per_row_sigmoid'`` applies
    the sigmoid before the hidden-axis reduce (the wrong order) for the negative test."""
    config = _config(d, hn.shape[1])
    host = ttMLA2DTP.host_weights_2d(_ttmla_names(w_hf), config, hp, tp)
    W = lambda name, r, c: ttMLA2DTP.chip_shard(*host[name], hp, tp, r, c).to(F32)  # noqa: E731
    S, H, N, R, V, L = hn.shape[0], d.num_heads, d.qk_nope_head_dim, d.qk_rope_head_dim, d.v_head_dim, d.kv_lora_rank
    hpc = H // (hp * tp)
    chips = [(r, c) for r in range(hp) for c in range(tp)]
    x = {(r, c): hn.chunk(hp, dim=1)[r] for r, c in chips}  # [S, hidden / hp], replicated over the head axis

    # q_a: K over hidden, N over heads -> reduce over the hidden axis, gather both axes -> q_a_layernorm on the full row.
    qa_col = [sum(x[r, c] @ W("q_a_proj", r, c) for r in range(hp)) for c in range(tp)]
    qa = torch.cat(qa_col, dim=1)
    q_ln = host["q_a_layernorm"][0].reshape(-1).to(F32)
    qn = qa * torch.rsqrt(qa.square().mean(-1, keepdim=True) + d.rms_norm_eps) * q_ln
    # q_b: K (q_lora) over hidden, heads over heads -> reduce-scatter over the hidden axis: 2 heads per chip.
    q_chip = {}
    for c in range(tp):
        col = sum(qn.chunk(hp, dim=1)[r] @ W("q_b_proj", r, c) for r in range(hp))
        for r in range(hp):
            q_chip[r, c] = col.chunk(hp, dim=1)[r].reshape(S, hpc, N + R)
    # kv_a: K over both axes, reduced over all chips.
    kv = sum(x[r, c].chunk(tp, dim=1)[c] @ W("kv_a_proj_with_mqa", r, c)[0, 0] for r, c in chips)
    kv_ln = host["kv_a_layernorm"][0].reshape(-1).to(F32)
    latent = kv[:, :L] * torch.rsqrt(kv[:, :L].square().mean(-1, keepdim=True) + d.rms_norm_eps) * kv_ln
    cos, sin = rope_cos_sin(torch.arange(S), R, d.rope_theta)
    k_pe = apply_rope_half_split(kv[:, L:], cos, sin)
    # Per-chip heads: wkv_b1, attention (the sink in sparse_sdpa's convention: the op multiplies it by the scale),
    # wkv_b2, the gate (hidden-axis reduce, then sigmoid), then o_proj (gather over hidden, K over heads, reduce).
    # The sink is replicated (no layout to test); the cached one is sparse_sdpa's bf16 sink / scale, whose rounding the
    # #65921 convention test covers, so the emulation uses the model's fp32 logits.
    sink_cached = host["attn_sink"][0].reshape(-1).to(F32) * d.softmax_scale
    sink_op = w_hf["learnable_sink_param"].to(F32)
    torch.testing.assert_close(sink_cached, sink_op, rtol=1e-2, atol=1e-2)
    topk = topk.long()
    pos = torch.arange(S).unsqueeze(1)
    valid = (topk >= 0) & (topk <= pos)
    safe = topk.clamp(min=0)
    gated = {}
    for r, c in chips:
        heads = ttMLA2DTP.heads_of_chip(H, hp, tp, r, c)
        q = q_chip[r, c]
        q_lat = torch.einsum("nhd,hdc->nhc", q[..., :N], W("wkv_b1", r, c)[0])
        q_pe = apply_rope_half_split(q[..., N:], cos, sin)
        sc = torch.einsum("nhc,nkc->nhk", q_lat, latent[safe]) + torch.einsum("nhr,nkr->nhk", q_pe, k_pe[safe])
        sc = (sc * d.softmax_scale).masked_fill(~valid.unsqueeze(1), float("-inf"))
        probs = sink_softmax(sc, sink_op[heads])
        o = torch.einsum("nhc,hcv->nhv", torch.einsum("nhk,nkc->nhc", probs, latent[safe]), W("wkv_b2", r, c)[0])
        if gate_order == "fused":
            g_col = sum(x[rr, c] @ W("g_proj", rr, c) for rr in range(hp))
            g = torch.sigmoid(g_col.chunk(hp, dim=1)[r])
        else:
            g_col = sum(torch.sigmoid(x[rr, c] @ W("g_proj", rr, c)) for rr in range(hp))
            g = g_col.chunk(hp, dim=1)[r]
        gated[r, c] = o.reshape(S, hpc * V) * g
    out_rows = []
    for r in range(hp):
        out_rows.append(
            sum(torch.cat([gated[rr, c] for rr in range(hp)], dim=1) @ W("o_proj", r, c) for c in range(tp))
        )
    return torch.cat(out_rows, dim=1), q_chip


def _causal_topk(S, K, seed=1):
    g = torch.Generator().manual_seed(seed)
    out = torch.full((S, K), -1, dtype=torch.long)
    for i in range(S):
        keys = torch.randperm(i + 1, generator=g)[:K]
        out[i, : keys.numel()] = keys
    return out


@pytest.mark.parametrize("hp,tp", [(2, 4), (4, 2), (2, 2), (1, 8)])
def test_2d_tp_emulation_matches_hf_reference(hp, tp):
    S = 48
    hn = torch.randn(S, SMALL_HIDDEN, generator=torch.Generator().manual_seed(3))
    w = _hf_weights(SMALL, SMALL_HIDDEN)
    topk = _causal_topk(S, 24)
    ref, _ = hy4_attention(hn, w, topk, SMALL, block=16)
    out, q_chip = _emulate_2d(hn, w, topk, SMALL, hp, tp)
    torch.testing.assert_close(out, ref, rtol=2e-4, atol=2e-4)
    # The q heads each chip ends up with are the ones heads_of_chip names (q_b's reduce-scatter order).
    H, qk = SMALL.num_heads, SMALL.qk_head_dim
    q_resid, _ = mla_latents(hn, w, SMALL, *rope_cos_sin(torch.arange(S), SMALL.qk_rope_head_dim, SMALL.rope_theta))
    q_full = (q_resid @ w["q_b_proj.weight"].T).view(S, H, qk)
    for (r, c), q in q_chip.items():
        torch.testing.assert_close(q, q_full[:, ttMLA2DTP.heads_of_chip(H, hp, tp, r, c)], rtol=2e-4, atol=2e-4)


def test_gate_reduces_over_hidden_before_the_sigmoid():
    S, hp, tp = 32, 4, 2
    hn = torch.randn(S, SMALL_HIDDEN, generator=torch.Generator().manual_seed(4))
    w = _hf_weights(SMALL, SMALL_HIDDEN, seed=5)
    topk = _causal_topk(S, 16)
    ref, _ = hy4_attention(hn, w, topk, SMALL, block=16)
    good, _ = _emulate_2d(hn, w, topk, SMALL, hp, tp)
    bad, _ = _emulate_2d(hn, w, topk, SMALL, hp, tp, gate_order="per_row_sigmoid")
    torch.testing.assert_close(good, ref, rtol=2e-4, atol=2e-4)
    assert (bad - ref).abs().max() > 0.1 * ref.abs().max()


HY4 = Hy4AttnDims()  # num_heads 64, q_lora 2048, kv_lora 512, qk 192 + 64, v 256
HY4_HIDDEN = 6144

# Per-chip shard shapes on the SC1 4x8 mesh (hidden over 4, heads over 8), as the module docstring lists them.
HY4_CHIP_SHAPES = {
    "q_a_proj": (1536, 256),
    "q_b_proj": (512, 8 * 256),
    "kv_a_proj_with_mqa": (1, 1, 192, 576),
    "wkv_b1": (1, 2, 192, 512),
    "wkv_b2": (1, 2, 512, 256),
    "g_proj": (1536, 8 * 256),
    "o_proj": (8 * 256, 1536),
}


def test_hy4_weights_are_sharded_once_over_the_sc1_mesh():
    hp, tp = 4, 8
    config = _config(HY4, HY4_HIDDEN)
    ttMLA2DTP.check_dims(config, hp, tp)
    host = ttMLA2DTP.host_weights_2d(None, config, hp, tp)
    total_attention = 0
    for name, shape in HY4_CHIP_SHAPES.items():
        t, hidden_dim, head_dim = host[name]
        assert hidden_dim is not None and head_dim is not None, f"{name} must be split over both mesh axes"
        assert tuple(ttMLA2DTP.chip_shard(t, hidden_dim, head_dim, hp, tp, 0, 0).shape) == shape, name
        # Sharded once: the 32 chip shards are disjoint and cover the tensor exactly.
        assert torch.Size(shape).numel() * hp * tp == t.numel(), f"{name} is replicated"
        total_attention += t.numel()
    # The only replicated attention tensors are the two norm vectors and the 64 sinks.
    replicated = {k for k, (_, a, b) in host.items() if a is None and b is None}
    assert replicated == {"q_a_layernorm", "kv_a_layernorm", "attn_sink"}
    # 266M attention parameters per layer (#65884), split over 32 chips.
    assert abs(total_attention / 1e6 - 266) < 1


def test_head_ownership_agrees_across_q_b_gate_and_o_proj():
    """q_b and the gate reduce-scatter their column's 8 heads over the hidden axis, o_proj all-gathers them back in
    hidden-axis order; wkv_b1 / wkv_b2 are loaded per heads_of_chip. All four must name the same heads."""
    H, hp, tp, V = 64, 4, 8, 4
    owned = [ttMLA2DTP.heads_of_chip(H, hp, tp, r, c) for c in range(tp) for r in range(hp)]
    assert sorted(h for hs in owned for h in hs) == list(range(H))
    w_uv = torch.arange(H).view(H, 1, 1).expand(H, 2, V).float()  # head h's rows hold h
    host = {"wkv_b2": (w_uv.reshape(tp, H // tp, 2, V), 1, 0)}
    for c in range(tp):
        col_heads = torch.arange(H).view(tp, H // tp)[c]  # the N range of this column in q_b / g_proj / o_proj's K
        for r in range(hp):
            mine = ttMLA2DTP.heads_of_chip(H, hp, tp, r, c)
            assert col_heads.chunk(hp)[r].tolist() == mine  # reduce-scatter over hidden, chunk r
            loaded = ttMLA2DTP.chip_shard(*host["wkv_b2"], hp, tp, r, c)[0, :, 0, 0]
            assert loaded.long().tolist() == mine


@pytest.mark.parametrize("mesh,sp_axis", [((8, 4), 0), ((8, 4), 1), ((4, 8), 0), ((4, 4), 1)])
def test_sparse_kv_push_chip_order_is_the_stripe_order_only_with_sp_on_axis_0(mesh, sp_axis):
    """The push ops place cache block b on chip b % chips in row-major mesh order; the TP-deduped cache puts stripe
    s * tp + t on the chip at SP position s, TP position t (``striped_cache_to_host``)."""
    rows, cols = mesh
    sp, tp = (rows, cols) if sp_axis == 0 else (cols, rows)
    agree = True
    for s in range(sp):
        for t in range(tp):
            r, c = (s, t) if sp_axis == 0 else (t, s)
            agree &= r * cols + c == s * tp + t
    assert agree == ttMLA.push_matches_stripe_order(sp_axis) == (sp_axis == 0)


def test_config_switch_keeps_ttmla_unless_opted_in():
    plain = types.SimpleNamespace()
    assert not uses_2d_tp(plain) and mla_class(plain) is ttMLA
    for model in ("glm", "deepseek", "kimi"):
        assert mla_class(types.SimpleNamespace(model=model)) is ttMLA
    opted = types.SimpleNamespace(**{MLA_2D_TP_FLAG: True})
    assert mla_class(opted) is ttMLA2DTP and issubclass(ttMLA2DTP, ttMLA)


def test_dims_that_do_not_tile_are_refused(expect_error):
    with expect_error(ValueError, "does not tile"):
        ttMLA2DTP.check_dims(_config(HY4, HY4_HIDDEN), 3, 8)  # 64 heads do not split over 24 chips
