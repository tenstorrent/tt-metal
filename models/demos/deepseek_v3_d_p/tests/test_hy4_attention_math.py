# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for Hy4-preview's attention deltas (#65921): the learnable per-head sink and the output gate.

* the absorbed sparse reference (``reference/hy4_attention.py``) equals a dense, non-absorbed HF-style
  eager attention with the sink as an extra softmax column and the gate on the head-major output;
* the sink in ``sparse_sdpa``'s convention (``ttMLA.sink_for_sparse_sdpa``: divided by the softmax scale,
  bf16; the op multiplies by scale and adds exp to the denominator only) reproduces the model's sink;
* the LSE fold (the alternative to the op's sink, not used) is the same function;
* ttMLA's output gate (``g_proj`` = ``linear_gate^T`` N-sharded over TP, multiplied after
  ``nlp_concat_heads``) is HF's ``sigmoid(hn @ linear_gate^T).view(n, H, v)`` for any TP head split;
* cache stems, the softmax scale and (when the #65919 golden reference is importable) the reference itself.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from models.demos.deepseek_v3_d_p.reference.hy4_attention import (
    Hy4AttnDims,
    apply_rope_half_split,
    causal_topk_all,
    hy4_attention,
    mla_latents,
    rope_cos_sin,
    sink_softmax,
)

F32 = torch.float32
TINY = Hy4AttnDims(
    num_heads=8,
    q_lora_rank=32,
    kv_lora_rank=32,
    qk_nope_head_dim=16,
    qk_rope_head_dim=8,
    v_head_dim=16,
    rms_norm_eps=1e-5,
    rope_theta=1e4,
)
HIDDEN = 64


def _weights(d: Hy4AttnDims, hidden: int = HIDDEN, seed: int = 0, sink_scale: float = 2.0) -> dict:
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g) * 0.3  # noqa: E731
    H = d.num_heads
    return {
        "q_a_proj.weight": r(d.q_lora_rank, hidden),
        "q_a_layernorm.weight": 1 + 0.1 * r(d.q_lora_rank),
        "q_b_proj.weight": r(H * d.qk_head_dim, d.q_lora_rank),
        "kv_a_proj_with_mqa.weight": r(d.kv_lora_rank + d.qk_rope_head_dim, hidden),
        "kv_a_layernorm.weight": 1 + 0.1 * r(d.kv_lora_rank),
        "kv_b_proj.weight": r(H * (d.qk_nope_head_dim + d.v_head_dim), d.kv_lora_rank),
        "o_proj.weight": r(hidden, H * d.v_head_dim),
        "linear_gate.weight": r(H * d.v_head_dim, hidden),
        "learnable_sink_param": torch.randn(H, generator=g) * sink_scale,
    }


def _random_topk(S: int, K: int, seed: int = 1) -> torch.Tensor:
    """Per query, K distinct causal keys (fewer near the start, padded with -1), in random order."""
    g = torch.Generator().manual_seed(seed)
    out = torch.full((S, K), -1, dtype=torch.long)
    for i in range(S):
        keys = torch.randperm(i + 1, generator=g)[:K]
        out[i, : keys.numel()] = keys
    return out


def _dense_eager(hn, w, topk, d: Hy4AttnDims, use_sink=True, use_gate=True):
    """HF-style eager attention: materialized per-head K/V, a dense [H, S, S] mask from the top-k, the sink
    as an extra softmax column, then the gate and o_proj. Independent of the absorbed formulation."""
    S, H, N, R, V, L = hn.shape[0], d.num_heads, d.qk_nope_head_dim, d.qk_rope_head_dim, d.v_head_dim, d.kv_lora_rank
    cos, sin = rope_cos_sin(torch.arange(S), R, d.rope_theta)
    q_resid, kvpe = mla_latents(hn, w, d, cos, sin)
    q = (q_resid @ w["q_b_proj.weight"].T).view(S, H, N + R)
    q = torch.cat([q[..., :N], apply_rope_half_split(q[..., N:], cos, sin)], dim=-1).transpose(0, 1)  # [H,S,N+R]
    kv = (kvpe[:, :L] @ w["kv_b_proj.weight"].T).view(S, H, N + V)
    k = torch.cat([kv[..., :N], kvpe[:, L:].unsqueeze(1).expand(S, H, R)], dim=-1).transpose(0, 1)
    v = kv[..., N:].transpose(0, 1)  # [H, S, V]
    allowed = torch.zeros(S, S, dtype=torch.bool)
    rows = torch.arange(S).unsqueeze(1).expand_as(topk)
    ok = (topk >= 0) & (topk <= rows)
    allowed[rows[ok], topk[ok]] = True
    scores = (q @ k.transpose(1, 2)) * d.softmax_scale
    scores = scores.masked_fill(~allowed, float("-inf"))
    if use_sink:
        sink_col = w["learnable_sink_param"].view(H, 1, 1).expand(H, S, 1)
        probs = torch.softmax(torch.cat([scores, sink_col], dim=-1), dim=-1)[..., :-1]
    else:
        probs = torch.softmax(scores, dim=-1)
    o = (probs @ v).transpose(0, 1)  # [S, H, V]
    if use_gate:
        o = o * torch.sigmoid(hn @ w["linear_gate.weight"].T).view(S, H, V)
    return o.reshape(S, H * V) @ w["o_proj.weight"].T


@pytest.mark.parametrize("use_sink,use_gate", [(True, True), (False, True), (True, False)])
def test_absorbed_sparse_reference_matches_dense_eager(use_sink, use_gate):
    S, K = 48, 12
    w = _weights(TINY)
    hn = torch.randn(S, HIDDEN, generator=torch.Generator().manual_seed(3))
    topk = _random_topk(S, K)
    got, _ = hy4_attention(hn, w, topk, TINY, use_sink=use_sink, use_gate=use_gate, block=16)
    want = _dense_eager(hn, w, topk, TINY, use_sink=use_sink, use_gate=use_gate)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


def test_sink_and_gate_change_the_output():
    """Neither delta is a no-op on these weights, so the equivalence above is not vacuous."""
    S = 32
    w = _weights(TINY)
    hn = torch.randn(S, HIDDEN, generator=torch.Generator().manual_seed(4))
    topk = causal_topk_all(S, 64)
    full, _ = hy4_attention(hn, w, topk, TINY)
    for kw in ({"use_sink": False}, {"use_gate": False}):
        other, _ = hy4_attention(hn, w, topk, TINY, **kw)
        assert (full - other).abs().max() > 1e-2, kw


def _kernel_softmax_with_sink(scores_unscaled, sink_in, scale):
    """What sparse_sdpa computes per head: softmax over scale*QK with exp(scale * sink_in) added to the
    denominator only (sparse_sdpa.hpp: 'the sink is multiplied by scale and contributes only to the softmax
    denominator'). ``sink_in`` is the [1, 1, 1, H] tensor handed to the op."""
    s = scores_unscaled * scale  # [n, H, K]
    sink = (sink_in.float().view(1, -1, 1)) * scale  # [1, H, 1]
    m = torch.maximum(s.amax(-1, keepdim=True), sink)
    e = torch.exp(s - m)
    return e / (e.sum(-1, keepdim=True) + torch.exp(sink - m))


@pytest.mark.parametrize("sink_scale", [0.5, 3.0, 8.0])
def test_sink_in_sparse_sdpa_convention(sink_scale):
    from models.demos.deepseek_v3_d_p.tt.mla.mla import ttMLA

    H, n, K = 64, 16, 40
    scale = Hy4AttnDims().softmax_scale
    g = torch.Generator().manual_seed(5)
    sink = torch.randn(H, generator=g) * sink_scale
    qk = torch.randn(n, H, K, generator=g) * 20.0  # unscaled q.k, ~ +-1.5 logits after the scale
    sink_in = ttMLA.sink_for_sparse_sdpa(sink, scale)
    assert sink_in.shape == (1, 1, 1, H) and sink_in.dtype == torch.bfloat16
    got = _kernel_softmax_with_sink(qk, sink_in, scale)
    want = sink_softmax(qk * scale, sink)
    # bf16 rounding of sink/scale: relative 2^-9 on the sink logit.
    torch.testing.assert_close(got, want, rtol=0, atol=2e-3 * max(1.0, sink_scale))
    # Without the pre-division the op would add exp(scale * sink): visibly different.
    wrong = _kernel_softmax_with_sink(qk, sink.to(torch.bfloat16).view(1, 1, 1, H), scale)
    assert (wrong - want).abs().max() > 1e-2


def test_lse_fold_is_the_same_function():
    """The alternative to the op's sink: run attention without it and rescale each head's output by
    exp(lse) / (exp(lse) + exp(sink)) = sigmoid(lse - sink). Exact -- the sink only grows the denominator."""
    H, n, K, V = 8, 10, 24, 16
    g = torch.Generator().manual_seed(6)
    scores, vals, sink = torch.randn(n, H, K, generator=g) * 3, torch.randn(n, K, V, generator=g), torch.randn(H)
    with_sink = torch.einsum("nhk,nkv->nhv", sink_softmax(scores, sink), vals)
    no_sink = torch.einsum("nhk,nkv->nhv", torch.softmax(scores, -1), vals)
    lse = torch.logsumexp(scores, dim=-1, keepdim=True)
    folded = no_sink * torch.sigmoid(lse - sink.view(1, H, 1))
    torch.testing.assert_close(folded, with_sink, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("tp", [1, 4, 8])
def test_ttmla_output_gate_layout_is_hf_gate(tp):
    """ttMLA's gate: g_proj = linear_gate^T, N-sharded over TP in contiguous blocks (mapper_tp1, the same
    contiguous head ranges as q_b_proj / wkv_b2), sigmoid, then multiplied into nlp_concat_heads' head-major
    [n, (H/tp) * v] on each chip. Reassembled over TP it must be HF's per-head elementwise gate."""
    H, V, n = 16, 8, 5
    g = torch.Generator().manual_seed(7)
    hn = torch.randn(n, HIDDEN, generator=g)
    linear_gate = torch.randn(H * V, HIDDEN, generator=g)
    o = torch.randn(n, H, V, generator=g)  # attention output after wkv_b2, per head
    want = (o * torch.sigmoid(hn @ linear_gate.T).view(n, H, V)).reshape(n, H * V)

    g_proj = linear_gate.T  # ttMLA._convert_and_cache_weights: state_dict["g_proj.weight"].transpose(-2, -1)
    hl = H // tp
    chips = []
    for t in range(tp):
        g_local = torch.sigmoid(hn @ g_proj[:, t * hl * V : (t + 1) * hl * V])  # all-gathered hn @ N-shard
        v_out = o[:, t * hl : (t + 1) * hl].reshape(n, hl * V)  # nlp_concat_heads of this chip's heads
        chips.append(v_out * g_local)
    torch.testing.assert_close(torch.cat(chips, dim=-1), want)


def test_cache_stems_and_softmax_scale():
    from models.demos.deepseek_v3_d_p.reference.hy4_config import hy4_hf_config
    from models.demos.deepseek_v3_d_p.tt.mla.mla import ttMLA

    assert "attn_sink" not in ttMLA.weight_names()
    assert "attn_sink" not in ttMLA.weight_names(has_output_gate=True)
    names = ttMLA.weight_names(has_output_gate=True, has_sink=True)
    assert names[-2:] == ["g_proj", "attn_sink"]
    cfg = hy4_hf_config()
    assert cfg.mla_learnable_sink and cfg.mla_use_output_gate
    assert ttMLA.softmax_scale(cfg) == pytest.approx(256**-0.5)  # plain RoPE: no YaRN mscale
    assert ttMLA.softmax_scale(cfg) == pytest.approx(Hy4AttnDims.from_config(cfg).softmax_scale)


def test_matches_the_golden_reference_attention():
    """When #65919's streaming reference is importable, its attention pass (with the iHC pre-mix made an
    identity) gives the same sublayer output as reference/hy4_attention.py."""
    model = pytest.importorskip("models.demos.deepseek_v3_d_p.reference.hy4.model")
    config = pytest.importorskip("models.demos.deepseek_v3_d_p.reference.hy4.config")
    tiny = config.tiny_random_config()
    d = Hy4AttnDims(
        num_heads=tiny.num_attention_heads,
        q_lora_rank=tiny.q_lora_rank,
        kv_lora_rank=tiny.kv_lora_rank,
        qk_nope_head_dim=tiny.qk_nope_head_dim,
        qk_rope_head_dim=tiny.qk_rope_head_dim,
        v_head_dim=tiny.v_head_dim,
        rms_norm_eps=tiny.rms_norm_eps,
        rope_theta=tiny.rope_theta,
    )
    if abs(tiny.softmax_scale - d.softmax_scale) > 1e-9:
        pytest.skip("tiny config uses a non-default softmax scale")
    hidden, S = tiny.hidden_size, 24
    w_hf = _weights(d, hidden=hidden, seed=11)
    hn = torch.randn(S, hidden, generator=torch.Generator().manual_seed(12))
    topk = _random_topk(S, 6, seed=13)
    want, _ = hy4_attention(hn, w_hf, topk, d, block=8)

    # Drive the golden reference's _attention on hn: its iHC pre-mix returns stream 0 with post = 1, and its
    # input_layernorm (recognised by identity) returns its input, so the residual add is the sublayer output.
    ref = model.Hy4PrefillReference.__new__(model.Hy4PrefillReference)
    ref.cfg, ref.attn_block, ref.proj_block = tiny, 8, 64
    hc = tiny.hc_mult
    input_norm = torch.ones(hidden)
    orig_hc, orig_norm = model.hyper_connection, model.rms_norm
    try:
        model.hyper_connection = lambda x4, *a, **k: (torch.ones(x4.shape[0], hc), x4[:, 0])
        model.rms_norm = lambda x, weight, eps: x.to(F32) if weight is input_norm else orig_norm(x, weight, eps)
        w = {
            **{n: w_hf[f"{n}.weight"] for n in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj")},
            "linear_gate": w_hf["linear_gate.weight"],
            "q_a_layernorm": w_hf["q_a_layernorm.weight"],
            "kv_a_layernorm": w_hf["kv_a_layernorm.weight"],
            "sinks": w_hf["learnable_sink_param"],
            "input_layernorm": input_norm,
            "attn_hc.fn": None,
            "attn_hc.base": None,
            "attn_hc.scale": None,
        }
        x4 = torch.zeros(S, hc, hidden)
        x4[:, 0] = hn
        cos, sin = model.rope_cos_sin(torch.arange(S), d.qk_rope_head_dim, d.rope_theta)
        shared = next(i for i in range(tiny.num_hidden_layers) if not tiny.is_full_indexer(i))
        before = x4.clone()
        ref._attention(shared, w, x4, cos, sin, topk.to(torch.int32))
    finally:
        model.hyper_connection, model.rms_norm = orig_hc, orig_norm
    got = (x4 - before)[:, 0]  # post == 1: the residual add is exactly the sublayer output
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------------------------- low-rank norm eps
HF_MODELING_CANDIDATES = (
    os.environ.get("HY4_HF_MODELING_PY", ""),
    "/mnt/tt-data/Hy4-Preview/modeling_hy_v4.py",
)


def test_low_rank_norm_eps_is_hf_hyv4rmsnorm_default():
    """HF hy_v4: ``q_a_layernorm = HYV4RMSNorm(q_lora_rank)`` and ``kv_a_layernorm = HYV4RMSNorm(kv_lora_rank)`` take
    HYV4RMSNorm's default eps 1e-6; only the decoder norms pass ``rms_norm_eps`` (1e-5). Device and reference both use
    1e-6 for those two; GLM-5.3 keeps rms_norm_eps."""
    from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import glm_5_3_hf_config
    from models.demos.deepseek_v3_d_p.reference.hy4_config import Hy4Config, hy4_hf_config
    from models.demos.deepseek_v3_d_p.tt.mla.mla import ttMLA

    cfg = hy4_hf_config()
    assert cfg.rms_norm_eps == 1e-5 and Hy4Config.LOW_RANK_NORM_EPS == 1e-6
    assert ttMLA.low_rank_norm_eps(cfg) == 1e-6
    assert Hy4AttnDims.from_config(cfg).low_rank_norm_eps == 1e-6 and Hy4AttnDims().low_rank_norm_eps == 1e-6
    glm = glm_5_3_hf_config()
    assert ttMLA.low_rank_norm_eps(glm) == glm.rms_norm_eps
    assert Hy4AttnDims.from_config(glm).low_rank_norm_eps == glm.rms_norm_eps


def test_low_rank_norm_eps_matches_the_hf_source():
    """When HF's modeling_hy_v4.py is on disk: the two latent norms are built without an eps and HYV4RMSNorm defaults
    to the value Hy4Config.LOW_RANK_NORM_EPS holds."""
    import re

    from models.demos.deepseek_v3_d_p.reference.hy4_config import Hy4Config

    path = next((Path(p) for p in HF_MODELING_CANDIDATES if p and Path(p).is_file()), None)
    if path is None:
        pytest.skip("HF modeling_hy_v4.py not found (set HY4_HF_MODELING_PY)")
    src = path.read_text()
    default = re.search(
        r"class HYV4RMSNorm\(nn\.Module\):\s*def __init__\(self, hidden_size, eps: float = ([0-9.e-]+)\)", src
    )
    assert default and float(default.group(1)) == Hy4Config.LOW_RANK_NORM_EPS
    assert re.search(r"self\.q_a_layernorm = HYV4RMSNorm\(self\.q_lora_rank\)", src)
    assert re.search(r"self\.kv_a_layernorm = HYV4RMSNorm\(config\.kv_lora_rank\)", src)


def test_reference_normalizes_the_latents_with_the_low_rank_eps(monkeypatch):
    """reference/hy4_attention.py uses low_rank_norm_eps for both latent norms (rms_norm_eps is the decoder norms')."""
    import models.demos.deepseek_v3_d_p.reference.hy4_attention as ref

    seen = []
    orig = ref.rms_norm
    monkeypatch.setattr(ref, "rms_norm", lambda x, w, eps: seen.append(eps) or orig(x, w, eps))
    d = Hy4AttnDims(
        num_heads=4, q_lora_rank=64, kv_lora_rank=32, qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16
    )
    w = _weights(d, hidden=64, seed=3)
    ref.hy4_attention(torch.randn(8, 64), w, _random_topk(8, 4, seed=4), d, block=8)
    assert seen and set(seen) == {d.low_rank_norm_eps} and d.low_rank_norm_eps != d.rms_norm_eps
