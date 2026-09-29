# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Standalone torch CPU reference for Hy4 Preview (text decoder) chunked prefill.

Independent of HF transformers (HF is only the oracle in check_hf). Loads the safetensors checkpoint directly and
runs chunked causal prefill against an explicit full-length per-layer state: the MLA latent cache and the indexer key
cache. Attention is sparse (each query attends to the <= 2048 keys its indexer picked), computed in the absorbed
latent form (576 = 512 latent + 64 RoPE per key), so a 5120-token chunk at a 51k prefix touches 2048 keys per query,
not the prefix. Every block runs through ``run_block``, so the block graph below is the code.

Residual stream: iHC with 4 streams. The block input/output is the streams flattened, [S, 4 * H] (H = 6144; stream j
is columns [j * H, (j + 1) * H), HF's ``flatten(2)`` order). Per sublayer, in fp32:
    mixes = (flat [S, 4H] @ fn[8, 4H]^T) * rsqrt(mean(flat^2) + 1e-5)
    pre   = sigmoid(mixes[:, :4] * scale[0] + base[:4]) + 1e-6          weights of the 4 streams in the sublayer input
    post  = 2 * sigmoid(mixes[:, 4:] * scale[1] + base[4:]) + 1e-6      stream_j += post_j * y

Block (78 layers; layer 0 dense FFN, 1-77 MoE; "full" layers 0, 1, 5, 9, ... run their own indexer, "shared" layers
reuse the latest full layer's top-k of the same chunk):
    in           [S, 4H]
    attn_hc      = cat(pre, post) [S, 8] fp32 from in                      hc_attn_layer
    attn_x       = sum_j pre_j * stream_j(in)                              [S, H]
    attn_norm    = w * rms(attn_x)                                         input_layernorm (eps 1e-5)
    q_resid      = q_a_layernorm(q_a_proj(attn_norm))                      [S, 2048]
  full:
    topk         = indexer(attn_norm, q_resid)        [stateful]           [S, 2048] int64 key positions, ascending,
                                                                           -1 padded (row p keeps all p + 1 keys
                                                                           while p + 1 <= 2048)
  shared:
    topk         = the latest full layer's topk for this chunk (ctx.extra["shared_topk"] overrides)
    attn_out     = attention(attn_norm, q_resid, topk)  [stateful]         MLA over the latent cache, sink, gate, o_proj
    h_mid        = stream_j(in) + post_j * attn_out for each j             [S, 4H]
    ffn_hc       = cat(pre, post) from h_mid                               hc_mlp_layer
    ffn_x        = sum_j pre_j * stream_j(h_mid)
    ffn_norm     = w * rms(ffn_x)                                          post_attention_layernorm
  dense (layer 0):
    mlp_out      = down(silu(gate(x)) * up(x))                             intermediate 18432
  MoE:
    router       = dense routing weights [S, 256] fp32                     sigmoid, top-8 on sigmoid + e_score_
                                                                           correction_bias (1 group), weights = the
                                                                           unbiased sigmoids renormalized, x 2.827
    experts_out  = sum_e router[:, e] * expert_e(ffn_norm)                 silu(min(g, 10)) * clamp(u, -10, 10)
    shared_out   = down(silu(gate(x)) * up(x))                             1 shared expert, intermediate 2048, no clamp
    mlp_out      = experts_out + shared_out
    out          = stream_j(h_mid) + post_j * mlp_out for each j           ffn_residual

Attention (64 heads, q_lora 2048, kv_lora 512, qk 192 NoPE + 64 RoPE, v 256, scale 256^-0.5 = 1/16):
    q = q_b_proj(q_resid) [S, 64, 256] -> q_nope 192 | q_rope 64 (RoPE)
    kv_a_proj_with_mqa(x) [S, 576] -> latent 512 (kv_a_layernorm) | k_rope 64 (RoPE, one head)
    kv_latent state row = [kv_a_layernorm(latent) | RoPE(k_rope)] (576)
    absorbed: q_lat = q_nope @ W_uk (per head 192 -> 512); score = [q_lat | q_rope] . kv_latent_j * 1/16 over the
              selected keys; softmax with the head's learnable sink as one more logit (its mass is dropped);
              out_lat = p @ latent_j (512); o_h = W_uv out_lat (512 -> 256); o_h *= sigmoid(linear_gate(x))_h;
              attn_out = o_proj(o)
    RoPE: interleaved (GPT-J pairs (x[2i], x[2i+1]) rotated by frequency i), theta 1e7, 64 dims, absolute positions,
    in the MLA and in the indexer. This is how Hy4 is served (SGLang configs/hy_v4.py: rope_interleave and
    indexer_rope_interleave True); transformers 5.17's port uses rotate-half, which the vendored oracle fixes
    (hf_hy_v4/__init__.py). Rows stay in checkpoint order: k_rope in kv_latent is the interleaved rotation of
    kv_a_proj's last 64 outputs.
    q_a_layernorm / kv_a_layernorm use eps 1e-6 (HF builds them with HYV4RMSNorm's default; SGLang uses 1e-5).
Indexer (32 heads x 128): q = wq_b(q_resid) [S, 32, 128], k = LayerNorm(wk(x)) [S, 128] (eps 1e-5, with bias), RoPE
    on the last 64 of the 128 dims; w = weights_proj(x) * 32^-0.5 * 128^-0.5;
    score[s, t] = sum_h w[s, h] * relu(q[s, h] . k[t]) over t <= s; topk 2048.
State per layer: kv_latent [max_seq, 576]; index_key [max_seq, 128] (full layers; [0, 128] on shared layers).
Model: embedding -> 4 identical streams -> layers -> hc_head (4-way sigmoid collapse, fp32) -> RMSNorm -> fp32
lm_head (untied).

Precision: weights and activations in ``dtype`` (fp32 for the gates); iHC, the router and the LM head always fp32, as
in HF. Row blocks of attention and the indexer are aligned to absolute positions and expert token groups are padded
to 32 rows, so chunked prefill reproduces one-shot prefill (MKL's GEMM kernels depend on M).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from models.demos.common.bringup.reference.interface import Ctx, Step, noop, run_block
from models.demos.hy4_preview_d_p.reference.weights import WeightLoader


@dataclass
class Hy4Config:
    hidden_size: int = 6144
    intermediate_size: int = 18432
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 78
    num_attention_heads: int = 64
    q_lora_rank: int = 2048
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 2048
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    routed_scaling_factor: float = 2.827
    norm_topk_prob: bool = True
    n_group: int = 1
    topk_group: int = 1
    swiglu_limit: float = 10.0
    hc_mult: int = 4
    hc_magnitude: float = 2.0
    hc_eps: float = 1e-6
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1e7
    vocab_size: int = 120832
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False
    mlp_layer_types: list = field(default_factory=list)
    indexer_types: list = field(default_factory=list)

    @classmethod
    def from_json(cls, path: str) -> "Hy4Config":
        with open(path) as f:
            raw = json.load(f)
        cfg = cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})
        rp = raw.get("rope_parameters") or {}
        assert rp.get("rope_type", "default") == "default", rp
        cfg.rope_theta = float(rp.get("rope_theta", cfg.rope_theta))
        assert cfg.hidden_act == "silu" and cfg.n_group == 1 and cfg.topk_group == 1 and cfg.n_shared_experts == 1
        assert not raw.get("attention_bias", False) and not cfg.tie_word_embeddings
        assert raw.get("gating_type", "elementwise") == "elementwise" and raw.get("learnable_sink", True)
        assert raw.get("enable_ihc", True) and raw.get("q_lora_rank") is not None
        return cfg

    def is_moe(self, i: int) -> bool:
        return self.mlp_layer_types[i] == "sparse"

    def is_full(self, i: int) -> bool:
        return self.indexer_types[i] == "full"

    def topk_source(self, i: int) -> int:
        """The layer whose indexer layer i uses: itself on full layers, the latest full layer before it otherwise."""
        return max(j for j in range(i + 1) if self.is_full(j))

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def scale(self) -> float:
        return self.qk_head_dim**-0.5


# --------------------------------------------------------------------------------------
# Functional building blocks (the per-op goldens)
# --------------------------------------------------------------------------------------


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """HF HYV4RMSNorm: w * (x * rsqrt(mean(x^2) + eps)) with the normalization in fp32, cast back before * w."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * y.to(x.dtype)


def rope_cos_sin(positions: torch.Tensor, theta: float, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Interleaved tables [S, dim]: entries 2i and 2i+1 hold cos / sin of position * theta^(-2i / dim) (the
    frequencies of HF HYV4RotaryEmbedding, default rope, fp32)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
    freqs = (positions.float()[:, None] * inv_freq[None, :]).repeat_interleave(2, dim=-1)
    return freqs.cos(), freqs.sin()


def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    """(x0, x1, x2, x3, ...) -> (-x1, x0, -x3, x2, ...)."""
    return torch.stack((-x[..., 1::2], x[..., 0::2]), dim=-1).flatten(-2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [S, ..., R] with interleaved cos/sin [S, R]: GPT-J RoPE on the whole last dim."""
    shape = (cos.shape[0],) + (1,) * (x.dim() - 2) + (cos.shape[1],)
    c, s = cos.view(shape), sin.view(shape)
    return x * c + rotate_every_two(x) * s


def hc_gates(streams: torch.Tensor, fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor, cfg: Hy4Config):
    """iHC pre/post gates [S, 8] fp32 (pre 4 | post 4) from the streams [S, 4H] (HF HYV4HyperConnection)."""
    flat = streams.float()
    mixes = F.linear(flat, fn.float()) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + cfg.rms_norm_eps)
    m = cfg.hc_mult
    pre = torch.sigmoid(mixes[:, :m] * scale[0] + base[:m]) + cfg.hc_eps
    post = cfg.hc_magnitude * torch.sigmoid(mixes[:, m:] * scale[1] + base[m:]) + cfg.hc_eps
    return torch.cat([pre, post], dim=-1)


def hc_pre(streams: torch.Tensor, gates: torch.Tensor, cfg: Hy4Config) -> torch.Tensor:
    """Sublayer input [S, H] = sum_j pre_j * stream_j (fp32 sum, cast to the stream dtype)."""
    pre = gates[:, : cfg.hc_mult]
    st = streams.view(streams.shape[0], cfg.hc_mult, -1)
    return torch.sum(pre.unsqueeze(-1) * st, dim=1).to(streams.dtype)


def hc_post(streams: torch.Tensor, gates: torch.Tensor, y: torch.Tensor, cfg: Hy4Config) -> torch.Tensor:
    """streams [S, 4H] + post_j * y [S, H] on each stream j, in fp32, cast to the stream dtype."""
    post = gates[:, cfg.hc_mult :]
    st = streams.view(streams.shape[0], cfg.hc_mult, -1)
    out = post.float().unsqueeze(-1) * y.float().unsqueeze(-2) + st.float()
    return out.to(streams.dtype).flatten(1)


def hc_head(streams: torch.Tensor, fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor, cfg: Hy4Config):
    """Final 4-stream collapse (HF HYV4HyperHead), fp32."""
    flat = streams.float()
    mixes = F.linear(flat, fn.float()) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + cfg.rms_norm_eps)
    pre = torch.sigmoid(mixes * scale.float() + base.float()) + cfg.hc_eps
    st = streams.view(streams.shape[0], cfg.hc_mult, -1)
    return (pre.unsqueeze(-1) * st).sum(dim=1).to(streams.dtype)


def swiglu_mlp(x, w_gate, w_up, w_down):
    return F.linear(F.silu(F.linear(x, w_gate)) * F.linear(x, w_up), w_down)


LATENT_NORM_EPS = 1e-6  # q_a_layernorm, kv_a_layernorm (HF HYV4RMSNorm default; the other norms use rms_norm_eps)
ROW_BLOCK = 64  # attention / indexer query rows per block, aligned to absolute positions


def _row_blocks(start: int, length: int, block: int = ROW_BLOCK):
    """Local [a, b) row ranges whose absolute positions are aligned to ``block``."""
    a = 0
    while a < length:
        b = min(length, (start + a) // block * block + block - start)
        yield a, b
        a = b


def indexer_topk(q: torch.Tensor, k_all: torch.Tensor, w: torch.Tensor, start: int, topk: int) -> torch.Tensor:
    """Top-k key positions per query for queries at [start, start + S), ascending, -1 padded to ``topk``.

    q [S, Hi, D] (post-RoPE), k_all [>= start + S, D] (post-RoPE index keys, prefix + chunk), w [S, Hi] (the scaled
    head weights). score[s, t] = sum_h w[s, h] * relu(q[s, h] . k[t]) for t <= start + s (HF HYV4Indexer, fp32).
    A row that sees at most ``topk`` keys keeps all of them (HF's top-k then selects every unmasked key)."""
    s_len = q.shape[0]
    out = torch.full((s_len, topk), -1, dtype=torch.int64)
    ar = torch.arange(topk)
    for a, b in _row_blocks(start, s_len):
        pos = torch.arange(start + a, start + b)
        end = start + b
        if end <= topk:
            out[a:b] = torch.where(ar[None, :] <= pos[:, None], ar[None, :], -1)
            continue
        sc = torch.matmul(q[a:b].float(), k_all[:end].float().t())  # [b, Hi, end]
        sc = torch.matmul(w[a:b].float().unsqueeze(-2), F.relu(sc)).squeeze(-2)  # [b, end]
        sc = sc.masked_fill(torch.arange(end)[None, :] > pos[:, None], float("-inf"))
        idx = sc.topk(topk, dim=-1).indices.sort(dim=-1).values
        small = pos + 1 <= topk  # these rows keep every key they see
        if small.any():
            idx[small] = torch.where(ar[None, :] <= pos[small, None], ar[None, :], -1)
        out[a:b] = idx
    return out


def sparse_mla(
    q: torch.Tensor, kv_all: torch.Tensor, idx: torch.Tensor, sink: torch.Tensor, scale: float, start: int, lat: int
) -> torch.Tensor:
    """Absorbed sparse MLA with a per-head sink. q [S, Hq, 576], kv_all [T, 576], idx [S, K] (-1 = no key),
    sink [Hq]. Softmax over the selected keys plus the sink logit (its mass dropped). Returns [S, Hq, lat]."""
    s_len, hq, _ = q.shape
    out = q.new_empty(s_len, hq, lat)
    for a, b in _row_blocks(start, s_len):
        ii = idx[a:b]
        valid = ii >= 0
        g = kv_all[ii.clamp(min=0)]  # [b, K, 576]
        sc = torch.bmm(q[a:b], g.transpose(1, 2)) * scale  # [b, Hq, K]
        sc = sc.masked_fill(~valid[:, None, :], float("-inf"))
        sk = sink.to(sc.dtype).view(1, hq, 1).expand(b - a, hq, 1)
        sc = torch.cat([sc, sk], dim=-1)
        sc = sc - sc.amax(dim=-1, keepdim=True)
        p = F.softmax(sc, dim=-1, dtype=sc.dtype)[..., :-1]
        out[a:b] = torch.bmm(p.to(g.dtype), g[..., :lat])
    return out


def route(x: torch.Tensor, gate_w: torch.Tensor, bias: torch.Tensor, cfg: Hy4Config):
    """HF HYV4TopkRouter (sigmoid, noaux_tc, 1 group): (top-k weights [S, K] fp32, top-k expert ids [S, K])."""
    logits = F.linear(x.float(), gate_w.float())
    scores = logits.sigmoid()
    choice = scores + bias.float()[None, :]
    ti = torch.topk(choice, k=cfg.num_experts_per_tok, dim=-1, sorted=False)[1]
    tw = scores.gather(1, ti)
    if cfg.norm_topk_prob:
        tw = tw / (tw.sum(dim=-1, keepdim=True) + 1e-20)
    return tw * cfg.routed_scaling_factor, ti


def dense_routing(tw: torch.Tensor, ti: torch.Tensor, num_experts: int) -> torch.Tensor:
    """[S, E] fp32 routing matrix: the top-k weight at each selected expert, 0 elsewhere (sigmoid weights are > 0)."""
    return torch.zeros(tw.shape[0], num_experts, dtype=tw.dtype).scatter_(1, ti, tw)


EXPERT_ROW_BLOCK = 32  # per-expert GEMM rows are padded to a multiple of this (see experts_forward)


def experts_forward(x: torch.Tensor, routing: torch.Tensor, expert_weights, limit: float) -> torch.Tensor:
    """sum over selected experts of routing[t, e] * expert_e(x[t]), accumulated in fp32 in expert-id order (the HF
    order), returned in x.dtype. ``expert_weights(e) -> (gate_up [2I, H], down [H, I])``; the expert is
    down(silu(min(gate, limit)) * clamp(up, -limit, limit)).

    Tokens are grouped per expert and each group is zero-padded to a multiple of EXPERT_ROW_BLOCK rows: MKL sgemm
    picks M-dependent kernels for small or odd M, so without padding a token's expert output would depend on how many
    other tokens chose that expert, and chunked prefill would stop matching one-shot."""
    out = torch.zeros(x.shape, dtype=torch.float32)
    tok_all, exp_all = (routing != 0).nonzero(as_tuple=True)
    order = torch.argsort(exp_all, stable=True)
    tok_all, exp_all = tok_all[order], exp_all[order]
    experts, counts = torch.unique_consecutive(exp_all, return_counts=True)
    off = 0
    for e, n in zip(experts.tolist(), counts.tolist()):
        tok = tok_all[off : off + n]
        off += n
        xe = x[tok]
        pad = -n % EXPERT_ROW_BLOCK
        if pad:
            xe = torch.cat([xe, xe.new_zeros(pad, xe.shape[1])])
        w_gu, w_d = expert_weights(e)
        gate, up = F.linear(xe, w_gu.to(x.dtype)).chunk(2, dim=-1)
        h = F.silu(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit)
        y = F.linear(h, w_d.to(x.dtype))[:n]
        out.index_add_(0, tok, (y * routing[tok, e, None].to(y.dtype)).float())
    return out.to(x.dtype)


# --------------------------------------------------------------------------------------
# Weights
# --------------------------------------------------------------------------------------


@dataclass
class LayerWeights:
    attn_hc: tuple  # (fn [8, 4H], base [8], scale [2]) fp32
    ffn_hc: tuple
    input_norm: torch.Tensor
    post_attn_norm: torch.Tensor
    q_a: torch.Tensor  # [2048, H]
    q_a_norm: torch.Tensor  # [2048]
    q_b: torch.Tensor  # [Hq * 256, 2048]
    kv_a: torch.Tensor  # [576, H]
    kv_a_norm: torch.Tensor  # [512]
    w_uk: torch.Tensor  # [Hq, 192, 512] (kv_b rows of k_nope per head)
    w_uv: torch.Tensor  # [Hq, 256, 512] (kv_b rows of v per head)
    gate: torch.Tensor  # [Hq * 256, H] (linear_gate)
    wo: torch.Tensor  # [H, Hq * 256]
    sink: torch.Tensor  # [Hq] fp32
    # indexer (full layers)
    idx_wq_b: torch.Tensor | None = None  # [32 * 128, 2048]
    idx_wk: torch.Tensor | None = None  # [128, H]
    idx_k_norm: tuple | None = None  # (w, b) fp32
    idx_weights_proj: torch.Tensor | None = None  # [32, H] fp32
    # dense layer
    w_gate: torch.Tensor | None = None
    w_up: torch.Tensor | None = None
    w_down: torch.Tensor | None = None
    # MoE layer
    router_w: torch.Tensor | None = None  # [E, H] fp32
    router_bias: torch.Tensor | None = None  # [E] fp32
    experts_gate_up: object = None  # [E, 2I, H] tensor, or an ExpertSlab (lazy)
    experts_down: object = None  # [E, H, I]
    shared: tuple | None = None  # (gate, up, down)


def load_layer(loader: WeightLoader, cfg: Hy4Config, i: int, dtype=torch.float32, eager_experts=True) -> LayerWeights:
    p = f"model.layers.{i}."
    g = lambda n: loader.get(p + n).to(dtype)  # noqa: E731
    f32 = lambda n: loader.get(p + n).float()  # noqa: E731
    hq, dn, dv, lat = cfg.num_attention_heads, cfg.qk_nope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank
    kv_b = g("self_attn.kv_b_proj.weight").view(hq, dn + dv, lat)
    w = LayerWeights(
        attn_hc=tuple(f32(f"hc_attn_layer.hc_pre.hc_{n}") for n in ("fn", "base", "scale")),
        ffn_hc=tuple(f32(f"hc_mlp_layer.hc_pre.hc_{n}") for n in ("fn", "base", "scale")),
        input_norm=g("input_layernorm.weight"),
        post_attn_norm=g("post_attention_layernorm.weight"),
        q_a=g("self_attn.q_a_proj.weight"),
        q_a_norm=g("self_attn.q_a_layernorm.weight"),
        q_b=g("self_attn.q_b_proj.weight"),
        kv_a=g("self_attn.kv_a_proj_with_mqa.weight"),
        kv_a_norm=g("self_attn.kv_a_layernorm.weight"),
        w_uk=kv_b[:, :dn].contiguous(),
        w_uv=kv_b[:, dn:].contiguous(),
        gate=g("self_attn.linear_gate.weight"),
        wo=g("self_attn.o_proj.weight"),
        sink=f32("self_attn.learnable_sink_param"),
    )
    if cfg.is_full(i):
        w.idx_wq_b = g("self_attn.indexer.wq_b.weight")
        w.idx_wk = g("self_attn.indexer.wk.weight")
        w.idx_k_norm = (f32("self_attn.indexer.k_norm.weight"), f32("self_attn.indexer.k_norm.bias"))
        w.idx_weights_proj = f32("self_attn.indexer.weights_proj.weight")
    if not cfg.is_moe(i):
        w.w_gate, w.w_up, w.w_down = (g(f"mlp.{n}.weight") for n in ("gate_proj", "up_proj", "down_proj"))
        return w
    w.router_w = f32("mlp.gate.weight")
    w.router_bias = f32("mlp.gate.e_score_correction_bias")
    w.shared = tuple(g(f"mlp.shared_experts.{n}.weight") for n in ("gate_proj", "up_proj", "down_proj"))
    if eager_experts:
        w.experts_gate_up = g("mlp.experts.gate_up_proj")
        w.experts_down = g("mlp.experts.down_proj")
    else:
        from models.demos.hy4_preview_d_p.reference.weights import ExpertSlab

        w.experts_gate_up = ExpertSlab(loader, p + "mlp.experts.gate_up_proj", dtype)
        w.experts_down = ExpertSlab(loader, p + "mlp.experts.down_proj", dtype)
    return w


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------


class State:
    """Per layer: kv_latent [max_seq, 576] (normed latent | RoPE'd k_rope), index_key [max_seq, 128] (full layers)."""

    def __init__(self, cfg: Hy4Config, layers: list[int], max_seq: int, dtype=torch.float32):
        w = cfg.kv_lora_rank + cfg.qk_rope_head_dim
        self.kv = {i: torch.zeros(max_seq, w, dtype=dtype) for i in layers}
        self.ik = {i: torch.zeros(max_seq, cfg.index_head_dim, dtype=dtype) for i in layers if cfg.is_full(i)}
        self.max_seq = max_seq


def _attn_steps(full: bool) -> list[Step]:
    topk = (
        Step("indexer", ("attn_norm", "q_resid"), "topk", "op", stateful=True)
        if full
        else Step("topk_shared", ("attn_norm",), "topk", "op")
    )
    return [
        Step("attn_hc", ("in",), "attn_hc", "op"),
        Step("attn_hc_pre", ("in", "attn_hc"), "attn_x", "op"),
        Step("attn_norm", ("attn_x",), "attn_norm", "norm"),
        Step("q_a", ("attn_norm",), "q_resid", "op"),
        topk,
        Step("attention", ("attn_norm", "q_resid", "topk"), "attn_out", "attention", stateful=True),
        Step("attn_residual", ("in", "attn_hc", "attn_out"), "h_mid", "residual"),
        Step("ffn_hc", ("h_mid",), "ffn_hc", "op"),
        Step("ffn_hc_pre", ("h_mid", "ffn_hc"), "ffn_x", "op"),
        Step("ffn_norm", ("ffn_x",), "ffn_norm", "norm"),
    ]


_DENSE_TAIL = [Step("mlp", ("ffn_norm",), "mlp_out", "mlp")]
_MOE_TAIL = [
    Step("router", ("ffn_norm",), "router", "router"),
    Step("experts", ("ffn_norm", "router"), "experts_out", "moe"),
    Step("shared_expert", ("ffn_norm",), "shared_out", "mlp"),
    Step("moe_combine", ("experts_out", "shared_out"), "mlp_out", "residual"),
]
_RESIDUAL = [Step("ffn_residual", ("h_mid", "ffn_hc", "mlp_out"), "out", "residual")]

# Up to this many MoE layers the routed experts are held in memory (in ``dtype``) at load; beyond it they are read
# per expert from the checkpoint when they run (``ExpertSlab``). Both give identical weights.
EAGER_EXPERT_LAYERS = 8


class Hy4Reference:
    """Implements models/demos/common/bringup/reference/interface.py. Keeps the requested layers resident."""

    def __init__(
        self,
        model_path: str,
        layers: list[int] | None = None,
        dtype=torch.float32,
        lm_head: bool = True,
        eager_experts: bool | None = None,
    ):
        self.cfg = Hy4Config.from_json(os.path.join(model_path, "config.json"))
        self.loader = WeightLoader(model_path)
        self.dtype = dtype
        self.layer_ids = list(range(self.cfg.num_hidden_layers)) if layers is None else list(layers)
        n_moe = sum(self.cfg.is_moe(i) for i in self.layer_ids)
        self.eager_experts = n_moe <= EAGER_EXPERT_LAYERS if eager_experts is None else eager_experts
        self.embed = self.loader.get("model.embed_tokens.weight").to(dtype)
        self.final_norm_w = self.loader.get("model.norm.weight").to(dtype)
        self.hc_head_w = tuple(self.loader.get(f"model.hc_head.hc_head_{n}").float() for n in ("fn", "base", "scale"))
        self.lm_head = self.loader.get("lm_head.weight").float() if lm_head else None
        self.w = {i: load_layer(self.loader, self.cfg, i, dtype, self.eager_experts) for i in self.layer_ids}
        self._rope_cache = {}
        self._topk = {}  # (full layer, start, length) -> topk of that layer's latest chunk

    # ---- interface
    def new_state(self, max_seq: int) -> State:
        return State(self.cfg, self.layer_ids, max_seq, self.dtype)

    def state_tensors(self, state: State, layer: int, length: int) -> dict:
        ik = state.ik[layer][:length] if layer in state.ik else state.kv[layer].new_zeros(0, self.cfg.index_head_dim)
        return {"kv_latent": state.kv[layer][:length].clone(), "index_key": ik.clone()}

    def load_state(self, state: State, layer: int, tensors: dict, length: int) -> None:
        state.kv[layer][:length] = tensors["kv_latent"][:length].to(state.kv[layer].dtype)
        if layer in state.ik:
            state.ik[layer][:length] = tensors["index_key"][:length].to(state.ik[layer].dtype)

    def block_graph(self, layer: int) -> list[Step]:
        tail = _MOE_TAIL if self.cfg.is_moe(layer) else _DENSE_TAIL
        return _attn_steps(self.cfg.is_full(layer)) + tail + _RESIDUAL

    def chunk_context(self, layer: int, start: int, length: int, state) -> Ctx:
        key = (start, length)
        if key not in self._rope_cache:
            cos, sin = rope_cos_sin(torch.arange(start, start + length), self.cfg.rope_theta, self.cfg.qk_rope_head_dim)
            self._rope_cache = {key: (cos.to(self.dtype), sin.to(self.dtype))}
        cos, sin = self._rope_cache[key]
        return Ctx(layer, start, length, state, extra={"cos": cos, "sin": sin})

    def expert_weights(self, layer: int):
        w = self.w[layer]
        return lambda e: (w.experts_gate_up[e], w.experts_down[e])

    def component(self, layer: int, name: str):
        cfg, w, eps = self.cfg, self.w[layer], self.cfg.rms_norm_eps
        norm = lambda wt: lambda ctx, x: rms_norm(x, wt, eps)  # noqa: E731
        table = {
            "attn_hc": lambda ctx, s: hc_gates(s, *w.attn_hc, cfg),
            "attn_hc_pre": lambda ctx, s, gt: hc_pre(s, gt, cfg),
            "attn_norm": norm(w.input_norm),
            "q_a": lambda ctx, x: rms_norm(F.linear(x, w.q_a), w.q_a_norm, LATENT_NORM_EPS),
            "indexer": lambda ctx, x, qr: self._indexer(layer, ctx, x, qr),
            "topk_shared": lambda ctx, x: self._shared_topk(layer, ctx),
            "attention": lambda ctx, x, qr, tk: self._attention(layer, ctx, x, qr, tk),
            "attn_residual": lambda ctx, s, gt, y: hc_post(s, gt, y, cfg),
            "ffn_hc": lambda ctx, s: hc_gates(s, *w.ffn_hc, cfg),
            "ffn_hc_pre": lambda ctx, s, gt: hc_pre(s, gt, cfg),
            "ffn_norm": norm(w.post_attn_norm),
            "ffn_residual": lambda ctx, s, gt, y: hc_post(s, gt, y, cfg),
        }
        if cfg.is_moe(layer):
            table.update(
                router=lambda ctx, x: dense_routing(*route(x, w.router_w, w.router_bias, cfg), cfg.n_routed_experts),
                experts=lambda ctx, x, r: experts_forward(x, r, self.expert_weights(layer), cfg.swiglu_limit),
                shared_expert=lambda ctx, x: swiglu_mlp(x, *w.shared),
                moe_combine=lambda ctx, a, b: a + b,
            )
        else:
            table["mlp"] = lambda ctx, x: swiglu_mlp(x, w.w_gate, w.w_up, w.w_down)
        return table[name]

    # ---- attention pieces
    def _indexer(self, layer: int, ctx: Ctx, x: torch.Tensor, q_resid: torch.Tensor) -> torch.Tensor:
        cfg, w = self.cfg, self.w[layer]
        s, hi, d, r = x.shape[0], cfg.index_n_heads, cfg.index_head_dim, cfg.qk_rope_head_dim
        cos, sin = ctx.extra["cos"], ctx.extra["sin"]
        q = F.linear(q_resid, w.idx_wq_b).view(s, hi, d)
        q = torch.cat([q[..., : d - r], apply_rope(q[..., d - r :], cos, sin)], dim=-1)
        k = F.layer_norm(F.linear(x, w.idx_wk).float(), (d,), *w.idx_k_norm, eps=cfg.rms_norm_eps).to(x.dtype)
        k = torch.cat([k[..., : d - r], apply_rope(k[..., d - r :], cos, sin)], dim=-1)
        wts = F.linear(x.float(), w.idx_weights_proj) * (hi**-0.5) * (d**-0.5)
        start, end = ctx.start, ctx.start + s
        ik = ctx.state.ik[layer]
        ik[start:end] = k
        topk = indexer_topk(q, ik[:end], wts, start, cfg.index_topk)
        self._topk = {kk: v for kk, v in self._topk.items() if kk[1:] == (start, s)}
        self._topk[(layer, start, s)] = topk
        return topk

    def _shared_topk(self, layer: int, ctx: Ctx) -> torch.Tensor:
        if "shared_topk" in ctx.extra:
            return ctx.extra["shared_topk"]
        src = self.cfg.topk_source(layer)
        key = (src, ctx.start, ctx.length)
        if key not in self._topk:
            raise KeyError(
                f"layer {layer} reuses layer {src}'s top-k, but layer {src} has not run on chunk "
                f"[{ctx.start}, {ctx.start + ctx.length}); run it first or set ctx.extra['shared_topk'] "
                f"(the golden's L{src}.topk)"
            )
        return self._topk[key]

    def _attention(self, layer: int, ctx: Ctx, x: torch.Tensor, q_resid: torch.Tensor, topk: torch.Tensor):
        cfg, w = self.cfg, self.w[layer]
        s, hq = x.shape[0], cfg.num_attention_heads
        dn, r, dv, lat = cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank
        cos, sin = ctx.extra["cos"], ctx.extra["sin"]
        q = F.linear(q_resid, w.q_b).view(s, hq, dn + r)
        q_lat = torch.einsum("shd,hdc->shc", q[..., :dn], w.w_uk)  # [S, Hq, 512]
        q = torch.cat([q_lat, apply_rope(q[..., dn:], cos, sin)], dim=-1)  # [S, Hq, 576]
        ckv = F.linear(x, w.kv_a)
        kv = torch.cat([rms_norm(ckv[:, :lat], w.kv_a_norm, LATENT_NORM_EPS), apply_rope(ckv[:, lat:], cos, sin)], -1)
        start, end = ctx.start, ctx.start + s
        st = ctx.state.kv[layer]
        st[start:end] = kv
        o_lat = sparse_mla(q, st[:end], topk, w.sink, cfg.scale, start, lat)  # [S, Hq, 512]
        o = torch.einsum("shc,hvc->shv", o_lat, w.w_uv)  # [S, Hq, 256]
        o = o * torch.sigmoid(F.linear(x, w.gate).view(s, hq, dv))
        return F.linear(o.reshape(s, hq * dv), w.wo)

    # ---- model
    @torch.no_grad()
    def forward_chunk(self, tokens, start, state, rec=noop, logits_last_n=0):
        """tokens [S] int at absolute positions [start, start+S). Returns (final_norm [S, H], logits [n, V] | None)."""
        tokens = tokens.long()
        h = F.embedding(tokens, self.embed)
        rec("embed", h)
        h = h.repeat(1, self.cfg.hc_mult)  # [S, 4H]: 4 copies of the embedding
        for i in self.layer_ids:
            ctx = self.chunk_context(i, start, tokens.shape[0], state)
            h = run_block(self.block_graph(i), lambda name, i=i: self.component(i, name), ctx, h, rec, prefix=f"L{i}.")
        hh = hc_head(h, *self.hc_head_w, self.cfg)
        rec("hc_head", hh)
        out = rms_norm(hh, self.final_norm_w, self.cfg.rms_norm_eps)
        rec("final_norm", out)
        logits = None
        if logits_last_n and self.lm_head is not None:
            logits = self.logits(out[-logits_last_n:])
            rec("logits", logits)
        return out, logits

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden.float(), self.lm_head)
