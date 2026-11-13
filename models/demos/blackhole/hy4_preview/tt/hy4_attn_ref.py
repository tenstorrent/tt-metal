# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU (torch) model of the device dataflow of ``Hy4Attention`` (absorbed MLA, column head groups, sink, gate), used by the S0 tests and as the dump-diff
reference of the device stages (every intermediate named like the device code's). Pure torch, fp32.

``attention_naive`` is the non-absorbed formulation straight from the checkpoint tensors (the oracle's formulas); ``attention_dataflow`` follows the device:
per column c: q_b slice -> q_nope @ w_uk -> [q_abs | rope(q_pe)] (576) against latent rows [lat | rope(k_pe)] -> sink softmax over the selected rows ->
o_latent @ w_uv_t -> x gate -> o_proj slice (partial) -> sum over columns.
"""
import torch
import torch.nn.functional as F

from models.demos.blackhole.hy4_preview.tt.hy4_attn_weights import build_attn_host_weights, column_shard
from models.demos.blackhole.hy4_preview.tt.hy4_rope import apply_rope_host, rope_tables


def rms(x, w, eps):
    x = x.float()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * w.float()


def sparse_softmax_sink(scores, allowed, sink_logit):
    """scores [H, S, T] (already scaled), allowed [S, T] bool, sink_logit [H] (logit domain: one extra softmax column without a value) -> probs [H, S, T]."""
    sc = scores.masked_fill(~allowed.unsqueeze(0), float("-inf"))
    full = torch.cat([sc, sink_logit.view(-1, 1, 1).expand(-1, sc.shape[1], 1)], -1)
    return full.softmax(-1)[..., :-1]


def attention_naive(cfg, sd, h, allowed, rope_interleave=True, gate_act="sigmoid"):
    """h [S, D] (post input_layernorm) -> (attention output [S, D], pre-gate o [S, H*vd]). allowed [S,S] bool (causal & selected)."""
    S = h.shape[0]
    H, nope, rope, vd, kvl = (
        cfg.num_attention_heads,
        cfg.qk_nope_head_dim,
        cfg.qk_rope_head_dim,
        cfg.v_head_dim,
        cfg.kv_lora_rank,
    )
    g = lambda k: sd[f"self_attn.{k}"].float()
    qa = rms(F.linear(h, g("q_a_proj.weight")), g("q_a_layernorm.weight"), cfg.rms_norm_eps)
    q = F.linear(qa, g("q_b_proj.weight")).view(S, H, nope + rope)
    kv = F.linear(h, g("kv_a_proj_with_mqa.weight"))
    lat = rms(kv[:, :kvl], g("kv_a_layernorm.weight"), cfg.rms_norm_eps)
    kvb = F.linear(lat, g("kv_b_proj.weight")).view(S, H, nope + vd)
    k_nope, v = kvb[..., :nope], kvb[..., nope:]
    cos, sin = rope_tables(torch.arange(S), rope, cfg.rope_theta, rope_interleave)
    q_pe = apply_rope_host(q[..., nope:].permute(1, 0, 2), cos, sin, rope_interleave).permute(1, 0, 2)  # [S,H,64]
    k_pe = apply_rope_host(kv[:, kvl:], cos, sin, rope_interleave)
    qf = torch.cat([q[..., :nope], q_pe], -1)
    kf = torch.cat([k_nope, k_pe.unsqueeze(1).expand(S, H, rope)], -1)
    sc = torch.einsum("shd,thd->hst", qf, kf) * (nope + rope) ** -0.5
    p = sparse_softmax_sink(
        sc, allowed, g("learnable_sink_param") if "self_attn.learnable_sink_param" in sd else torch.zeros(H)
    )
    o = torch.einsum("hst,thd->shd", p, v).reshape(S, H * vd)
    gate = F.linear(h, g("linear_gate.weight"))
    gate = torch.sigmoid(gate) if gate_act == "sigmoid" else gate
    return F.linear(o * gate, g("o_proj.weight")), o


def attention_dataflow(cfg, sd, h, allowed, rope_interleave=True, tp=8, positions=None):
    """The device dataflow in torch (see module doc). Returns (out [S,D], dict of per-column intermediates for dump-diff)."""
    S = h.shape[0]
    hw = build_attn_host_weights(cfg, sd, tp)
    H, nope, rope, vd, kvl = (
        cfg.num_attention_heads,
        cfg.qk_nope_head_dim,
        cfg.qk_rope_head_dim,
        cfg.v_head_dim,
        cfg.kv_lora_rank,
    )
    hc = H // tp
    pos = torch.arange(S) if positions is None else positions
    cos, sin = rope_tables(pos, rope, cfg.rope_theta, rope_interleave)
    P = {k: v.tensor for k, v in hw.items()}
    qa = torch.cat([h @ column_shard(hw["q_a"], c, tp)[0, 0] for c in range(tp)], -1)  # N-shards -> all_gather
    qa = rms(qa, sd["self_attn.q_a_layernorm.weight"], cfg.rms_norm_eps)
    kv = h @ P["kv_a"][0, 0]
    lat = rms(kv[:, :kvl], sd["self_attn.kv_a_layernorm.weight"], cfg.rms_norm_eps)
    k_pe = apply_rope_host(kv[:, kvl:], cos, sin, rope_interleave)
    rows = torch.cat([lat, k_pe], -1)  # the cache rows [S, 576]
    out = torch.zeros(S, cfg.hidden_size)
    inter = {"qa": qa, "rows": rows}
    for c in range(tp):
        qc = (qa @ column_shard(hw["q_b"], c, tp)[0, 0]).view(S, hc, nope + rope)
        wuk = column_shard(hw["w_uk"], c, tp)[0]  # [hc,192,512]
        q_abs = torch.einsum("shd,hdk->shk", qc[..., :nope], wuk)  # [S,hc,512]
        q_pe = apply_rope_host(qc[..., nope:].permute(1, 0, 2), cos, sin, rope_interleave).permute(1, 0, 2)
        qfull = torch.cat([q_abs, q_pe], -1)  # [S,hc,576]
        sc = torch.einsum("shk,tk->hst", qfull, rows) * (nope + rope) ** -0.5
        sink = column_shard(hw["sink_over_scale"], c, tp).reshape(-1) * (nope + rope) ** -0.5
        p = sparse_softmax_sink(sc, allowed, sink)
        o_lat = torch.einsum("hst,tk->shk", p, rows[:, :kvl])  # [S,hc,512]
        o = torch.einsum("shk,hkd->shd", o_lat, column_shard(hw["w_uv_t"], c, tp)[0]).reshape(S, hc * vd)
        gate = torch.sigmoid(h @ column_shard(hw["gate"], c, tp)[0, 0])
        out = out + (o * gate) @ column_shard(hw["o_proj"], c, tp)[0, 0]
        inter[f"o_c{c}"] = o
    return out, inter


def column_intermediates(cfg, hw, qa, rows, allowed, c, p0, rope_interleave=True, tp=8):
    """CPU values of column c's stages for the chunk at positions p0..p0+n: qa [n,2048] (post q_a_layernorm), rows [p0+n, 576] cache rows, allowed [n, p0+n] bool
    -> dict(qfull [1,8,n,576], o_lat [1,8,n,512], o [n,2048]) in the device tap layouts."""
    n = qa.shape[0]
    hc = cfg.num_attention_heads // tp
    nope, rope = cfg.qk_nope_head_dim, cfg.qk_rope_head_dim
    cos, sin = rope_tables(torch.arange(p0, p0 + n), rope, cfg.rope_theta, rope_interleave)
    qc = (qa @ column_shard(hw["q_b"], c, tp)[0, 0]).view(n, hc, nope + rope)
    q_abs = torch.einsum("shd,hdk->shk", qc[..., :nope], column_shard(hw["w_uk"], c, tp)[0])
    q_pe = apply_rope_host(qc[..., nope:].permute(1, 0, 2), cos, sin, rope_interleave).permute(1, 0, 2)
    qfull = torch.cat([q_abs, q_pe], -1)
    sc = torch.einsum("shk,tk->hst", qfull, rows) * (nope + rope) ** -0.5
    sink = column_shard(hw["sink_over_scale"], c, tp).reshape(-1) * (nope + rope) ** -0.5
    p = sparse_softmax_sink(sc, allowed, sink)
    o_lat = torch.einsum("hst,tk->shk", p, rows[:, : cfg.kv_lora_rank])
    o = torch.einsum("shk,hkd->shd", o_lat, column_shard(hw["w_uv_t"], c, tp)[0]).reshape(n, -1)
    return {"qfull": qfull.permute(1, 0, 2).unsqueeze(0), "o_lat": o_lat.permute(1, 0, 2).unsqueeze(0), "o": o}
