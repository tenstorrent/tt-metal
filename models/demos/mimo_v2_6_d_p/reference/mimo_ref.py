# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Standalone torch CPU reference for MiMo-V2.6-Flash-RL (text decoder) chunked prefill.

Independent of HF transformers (HF is only the oracle in check_hf). Loads the safetensors checkpoint directly
(dequantizing fp8 / mxfp4 as weights.py documents) and runs chunked causal prefill with an explicit full-length
per-layer KV state. Every block runs through ``run_block``, so the block graph below is the code.

Block (48 layers; layer 0 dense, layers 1-47 MoE; hybrid_layer_pattern picks full or sliding attention):
    in
    attn_norm   = w * rms(in)                                    input_layernorm (plain w, eps 1e-6)
    attn_out    = attention(attn_norm)          [stateful]       fused qkv, partial RoPE, (sink) SDPA, o_proj
    h_mid       = in + attn_out
    ffn_norm    = w * rms(h_mid)                                 post_attention_layernorm
  dense (layer 0):
    mlp_out     = down(silu(gate(x)) * up(x))                    intermediate 16384
    out         = h_mid + mlp_out
  MoE:
    router      = dense routing weights [S, 256] fp32 from ffn_norm    sigmoid scores, top-8 on scores + e_score_
                                                                 correction_bias (noaux_tc, 1 group), weights =
                                                                 unbiased scores of the chosen 8, renormalized
    experts_out = sum_e router[:, e] * expert_e(ffn_norm)        256 experts, intermediate 2048, silu; no shared
    out         = h_mid + experts_out

Attention (head_dim QK 192, V 128, scale 192^-0.5, RoPE rotate-half on the first 64 dims of q and k):
    full    (layers 0, 5, 11, ...): 64 q heads, 4 KV heads, theta 1e7, no sink
    sliding (the rest): window 128 (key j visible to query i iff i - 128 < j <= i), 64 q heads, 8 KV heads,
            theta 1e4, per-head sink logit appended to the softmax (probability mass dropped, not attended)
    V is multiplied by attention_value_scale (0.707) before it is cached.
State: key [Hkv, max_seq, 192] (post-RoPE), value [Hkv, max_seq, 128] (x 0.707), per layer.
Embedding: plain lookup. Final norm, then an untied lm_head.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.demos.common.bringup.reference.interface import Ctx, Step, noop, run_block
from models.demos.mimo_v2_6_d_p.reference.weights import PackedExpert, WeightLoader, fp8_weight, qkv_weight


@dataclass
class MiMoConfig:
    hidden_size: int = 4096
    intermediate_size: int = 16384
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 48
    num_attention_heads: int = 64
    num_key_value_heads: int = 4
    head_dim: int = 192
    v_head_dim: int = 128
    swa_num_attention_heads: int = 64
    swa_num_key_value_heads: int = 8
    swa_head_dim: int = 192
    swa_v_head_dim: int = 128
    sliding_window: int = 128
    rope_theta: float = 1e7
    swa_rope_theta: float = 1e4
    partial_rotary_factor: float = 0.334
    attention_value_scale: float | None = 0.707
    add_full_attention_sink_bias: bool = False
    add_swa_attention_sink_bias: bool = True
    layernorm_epsilon: float = 1e-6
    vocab_size: int = 152576
    n_routed_experts: int = 256
    num_experts_per_tok: int = 8
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: bool = True
    routed_scaling_factor: float | None = None
    scoring_func: str = "sigmoid"
    topk_method: str = "noaux_tc"
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False
    hybrid_layer_pattern: tuple = ()
    moe_layer_freq: tuple = ()

    @classmethod
    def from_json(cls, path: str) -> "MiMoConfig":
        with open(path) as f:
            raw = json.load(f)
        known = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        cfg = cls(**known)
        cfg.hybrid_layer_pattern = tuple(cfg.hybrid_layer_pattern)
        cfg.moe_layer_freq = tuple(cfg.moe_layer_freq)
        rp = raw.get("rope_parameters") or {}
        if "rope_theta" in rp:
            cfg.rope_theta = rp["rope_theta"]
        cfg.partial_rotary_factor = rp.get("partial_rotary_factor", cfg.partial_rotary_factor)
        assert raw.get("attention_projection_layout") == "fused_qkv"
        assert cfg.hidden_act == "silu" and cfg.scoring_func == "sigmoid" and cfg.topk_method == "noaux_tc"
        assert not raw.get("n_shared_experts"), "shared experts not supported"
        assert rp.get("rope_type", "default") == "default"
        assert not raw.get("attention_bias", False)
        return cfg

    def is_sliding(self, i: int) -> bool:
        return self.hybrid_layer_pattern[i] == 1

    def is_moe(self, i: int) -> bool:
        return bool(self.moe_layer_freq[i])

    def attn_dims(self, i: int) -> tuple[int, int, int, int]:
        """(q heads, kv heads, qk head dim, v head dim) of layer i."""
        if self.is_sliding(i):
            return self.swa_num_attention_heads, self.swa_num_key_value_heads, self.swa_head_dim, self.swa_v_head_dim
        return self.num_attention_heads, self.num_key_value_heads, self.head_dim, self.v_head_dim

    def rope_dim(self, i: int) -> int:
        return self.rope_dim_for(self.is_sliding(i))

    def rope_dim_for(self, sliding: bool) -> int:
        return int((self.swa_head_dim if sliding else self.head_dim) * self.partial_rotary_factor)

    def has_sink(self, i: int) -> bool:
        return self.add_swa_attention_sink_bias if self.is_sliding(i) else self.add_full_attention_sink_bias


# --------------------------------------------------------------------------------------
# Functional building blocks (the per-op goldens)
# --------------------------------------------------------------------------------------


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """HF MiMoV2RMSNorm: w * (x * rsqrt(mean(x^2) + eps)) with the normalization in fp32, cast back before * w."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * y.to(x.dtype)


def rope_inv_freq(base: float, dim: int) -> torch.Tensor:
    return 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))


def rope_cos_sin(positions: torch.Tensor, inv_freq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate-half tables [S, rope_dim]: cat(freqs, freqs)."""
    freqs = positions.float()[:, None] * inv_freq[None, :].float()
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_partial_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [S, H, D]; cos/sin: [S, R] with R <= D: rotate the first R dims, pass the rest."""
    r = cos.shape[-1]
    xr, xp = x[..., :r], x[..., r:]
    xr = xr * cos[:, None] + rotate_half(xr) * sin[:, None]
    return torch.cat([xr, xp], dim=-1)


def swiglu_mlp(x, w_gate, w_up, w_down):
    return F.linear(F.silu(F.linear(x, w_gate)) * F.linear(x, w_up), w_down)


def chunk_attention(
    q: torch.Tensor,
    k_all: torch.Tensor,
    v_all: torch.Tensor,
    start: int,
    scale: float,
    window: int | None = None,
    sink: torch.Tensor | None = None,
    q_block: int = 512,
) -> torch.Tensor:
    """Causal (optionally sliding-window, optionally sink) GQA attention for queries at [start, start + Sq).

    q: [Hq, Sq, D]; k_all: [Hkv, >= start + Sq, D]; v_all: [Hkv, >= start + Sq, Dv] (state prefix + this chunk).
    The key range per query block is bounded by the window, so a sliding chunk never touches the whole prefix.
    sink: [Hq] logits appended as an extra softmax column whose probability is dropped (HF eager semantics).
    Returns [Hq, Sq, Dv].
    """
    hq, sq, d = q.shape
    hkv, dv = k_all.shape[0], v_all.shape[-1]
    rep = hq // hkv
    out = q.new_empty(hkv, rep, sq, dv)
    qg = q.view(hkv, rep, sq, d)
    for qs in range(0, sq, q_block):
        qe = min(sq, qs + q_block)
        kv_end = start + qe
        kv_start = max(0, start + qs - window + 1) if window else 0
        q_pos = torch.arange(start + qs, start + qe)[:, None]
        k_pos = torch.arange(kv_start, kv_end)[None, :]
        mask = k_pos <= q_pos
        if window:
            mask = mask & (k_pos > q_pos - window)
        k = k_all[:, None, kv_start:kv_end]
        v = v_all[:, None, kv_start:kv_end]
        qb = qg[:, :, qs:qe]
        if sink is None:
            out[:, :, qs:qe] = F.scaled_dot_product_attention(qb, k, v, attn_mask=mask, scale=scale)
        else:
            s = torch.matmul(qb, k.transpose(-1, -2)) * scale
            s = s.masked_fill(~mask, float("-inf"))
            sk = sink.view(hkv, rep, 1, 1).to(s.dtype).expand(hkv, rep, qe - qs, 1)
            s = torch.cat([s, sk], dim=-1)
            s = s - s.amax(dim=-1, keepdim=True)
            p = F.softmax(s, dim=-1, dtype=torch.float32).to(q.dtype)[..., :-1]
            out[:, :, qs:qe] = torch.matmul(p, v)
    return out.view(hq, sq, dv)


def route(x: torch.Tensor, gate_w: torch.Tensor, bias: torch.Tensor, cfg: MiMoConfig):
    """HF MiMoV2MoEGate (noaux_tc, sigmoid): (top-k weights [S, K] fp32, top-k expert ids [S, K])."""
    logits = F.linear(x.float(), gate_w.float())
    scores = logits.sigmoid()
    choice = scores + bias.float()[None, :]
    if cfg.n_group > 1:
        s, e = choice.shape
        group_scores = choice.view(s, cfg.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, k=cfg.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
        score_mask = group_mask.unsqueeze(-1).expand(s, cfg.n_group, e // cfg.n_group).reshape(s, -1)
        choice = choice.masked_fill(~score_mask.bool(), float("-inf"))
    ti = torch.topk(choice, k=cfg.num_experts_per_tok, dim=-1, sorted=False)[1]
    tw = scores.gather(1, ti)
    if cfg.num_experts_per_tok > 1 and cfg.norm_topk_prob:
        tw = tw / (tw.sum(dim=-1, keepdim=True) + 1e-20)
    tw = tw * (cfg.routed_scaling_factor if cfg.routed_scaling_factor is not None else 1.0)
    return tw, ti


def dense_routing(tw: torch.Tensor, ti: torch.Tensor, num_experts: int) -> torch.Tensor:
    """[S, E] fp32 routing matrix: the top-k weight at each selected expert, 0 elsewhere (sigmoid weights are > 0)."""
    return torch.zeros(tw.shape[0], num_experts, dtype=tw.dtype).scatter_(1, ti, tw)


EXPERT_ROW_BLOCK = 32  # per-expert GEMM rows are padded to a multiple of this (see experts_forward)


def experts_forward(x: torch.Tensor, routing: torch.Tensor, expert_weights) -> torch.Tensor:
    """sum over selected experts of routing[t, e] * expert_e(x[t]), accumulated in fp32 in expert-id order (the HF
    order), returned in x.dtype. ``expert_weights(e) -> (gate [I, H], up [I, H], down [H, I])``.

    Tokens are grouped per expert and each group is zero-padded to a multiple of EXPERT_ROW_BLOCK rows: MKL sgemm
    picks M-dependent kernels for small or odd M, so without padding a token's expert output would depend on how many
    other tokens chose that expert, and chunked prefill would stop matching one-shot."""
    out = torch.zeros(x.shape, dtype=routing.dtype)
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
        wg, wu, wd = expert_weights(e)
        y = swiglu_mlp(xe, wg.to(x.dtype), wu.to(x.dtype), wd.to(x.dtype))[:n]
        out.index_add_(0, tok, y * routing[tok, e, None])
    return out.to(x.dtype)


# --------------------------------------------------------------------------------------
# Weights
# --------------------------------------------------------------------------------------


@dataclass
class LayerWeights:
    input_norm: torch.Tensor
    post_attn_norm: torch.Tensor
    wqkv: torch.Tensor  # [q + k + v, H], global [q; k; v] order
    wo: torch.Tensor  # [H, Hq * Dv]
    sink: torch.Tensor | None  # [Hq]
    # dense layer
    w_gate: torch.Tensor | None = None
    w_up: torch.Tensor | None = None
    w_down: torch.Tensor | None = None
    # MoE layer
    router_w: torch.Tensor | None = None  # [E, H]
    router_bias: torch.Tensor | None = None  # [E] fp32
    experts: list | None = None  # per expert: (gate, up, down) dense tuple, or PackedExpert


def load_layer(loader: WeightLoader, cfg: MiMoConfig, i: int, dtype=torch.float32, eager_experts=True) -> LayerWeights:
    p = f"model.layers.{i}."
    g = lambda n: loader.get(p + n).to(dtype)  # noqa: E731
    hq, hkv, d, dv = cfg.attn_dims(i)
    w = LayerWeights(
        input_norm=g("input_layernorm.weight"),
        post_attn_norm=g("post_attention_layernorm.weight"),
        wqkv=qkv_weight(loader, p + "self_attn.", (hq * d, hkv * d, hkv * dv), dtype),
        wo=g("self_attn.o_proj.weight"),
        sink=g("self_attn.attention_sink_bias") if cfg.has_sink(i) else None,
    )
    if not cfg.is_moe(i):
        w.w_gate, w.w_up, w.w_down = (
            fp8_weight(loader, p + f"mlp.{n}.weight", dtype) for n in ("gate_proj", "up_proj", "down_proj")
        )
        return w
    w.router_w = g("mlp.gate.weight")
    w.router_bias = loader.get(p + "mlp.gate.e_score_correction_bias").float()
    w.experts = []
    for e in range(cfg.n_routed_experts):
        pe = PackedExpert(loader, p + f"mlp.experts.{e}.")
        w.experts.append(pe.weights(dtype) if eager_experts else pe)
    return w


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------


class State:
    """Full-length KV per layer: k[i] [Hkv, max_seq, D] (post-RoPE), v[i] [Hkv, max_seq, Dv] (x value scale)."""

    def __init__(self, cfg: MiMoConfig, layers: list[int], max_seq: int, dtype=torch.float32):
        self.k, self.v = {}, {}
        for i in layers:
            _, hkv, d, dv = cfg.attn_dims(i)
            self.k[i] = torch.zeros(hkv, max_seq, d, dtype=dtype)
            self.v[i] = torch.zeros(hkv, max_seq, dv, dtype=dtype)
        self.max_seq = max_seq


ATTN_STEPS = [
    Step("attn_norm", ("in",), "attn_norm", "norm"),
    Step("attention", ("attn_norm",), "attn_out", "attention", stateful=True),
    Step("attn_residual", ("in", "attn_out"), "h_mid", "residual"),
    Step("ffn_norm", ("h_mid",), "ffn_norm", "norm"),
]
DENSE_GRAPH = ATTN_STEPS + [
    Step("mlp", ("ffn_norm",), "mlp_out", "mlp"),
    Step("mlp_residual", ("h_mid", "mlp_out"), "out", "residual"),
]
MOE_GRAPH = ATTN_STEPS + [
    Step("router", ("ffn_norm",), "router", "router"),
    Step("experts", ("ffn_norm", "router"), "experts_out", "moe"),
    Step("ffn_residual", ("h_mid", "experts_out"), "out", "residual"),
]

# Up to this many MoE layers the experts are dequantized at load (fast goldens); beyond it (the 48-layer parity run)
# they stay mxfp4 and each expert is dequantized when it runs. Both give bit-identical weights.
EAGER_EXPERT_LAYERS = 8


class MiMoReference:
    """Implements models/demos/common/bringup/reference/interface.py. Keeps the requested layers resident."""

    def __init__(
        self,
        model_path: str,
        layers: list[int] | None = None,
        dtype=torch.float32,
        lm_head: bool = True,
        eager_experts: bool | None = None,
    ):
        self.cfg = MiMoConfig.from_json(os.path.join(model_path, "config.json"))
        self.loader = WeightLoader(model_path)
        self.dtype = dtype
        self.layer_ids = list(range(self.cfg.num_hidden_layers)) if layers is None else list(layers)
        n_moe = sum(self.cfg.is_moe(i) for i in self.layer_ids)
        self.eager_experts = n_moe <= EAGER_EXPERT_LAYERS if eager_experts is None else eager_experts
        self.embed = self.loader.get("model.embed_tokens.weight").to(dtype)
        self.final_norm_w = self.loader.get("model.norm.weight").to(dtype)
        self.lm_head = self.loader.get("lm_head.weight").to(dtype) if lm_head else None
        self.w = {i: load_layer(self.loader, self.cfg, i, dtype, self.eager_experts) for i in self.layer_ids}
        self.inv_freq = {
            s: rope_inv_freq(self.cfg.swa_rope_theta if s else self.cfg.rope_theta, self.cfg.rope_dim_for(s))
            for s in (True, False)
        }
        self._rope_cache = {}

    # ---- interface
    def new_state(self, max_seq: int) -> State:
        return State(self.cfg, self.layer_ids, max_seq, self.dtype)

    def state_tensors(self, state: State, layer: int, length: int) -> dict:
        return {"key": state.k[layer][:, :length].clone(), "value": state.v[layer][:, :length].clone()}

    def load_state(self, state: State, layer: int, tensors: dict, length: int) -> None:
        state.k[layer][:, :length] = tensors["key"][:, :length].to(state.k[layer].dtype)
        state.v[layer][:, :length] = tensors["value"][:, :length].to(state.v[layer].dtype)

    def block_graph(self, layer: int) -> list[Step]:
        return list(MOE_GRAPH if self.cfg.is_moe(layer) else DENSE_GRAPH)

    def chunk_context(self, layer: int, start: int, length: int, state) -> Ctx:
        sliding = self.cfg.is_sliding(layer)
        key = (sliding, start, length)
        if key not in self._rope_cache:
            self._rope_cache = {k: v for k, v in self._rope_cache.items() if k[1:] == key[1:]}
            cos, sin = rope_cos_sin(torch.arange(start, start + length), self.inv_freq[sliding])
            self._rope_cache[key] = (cos.to(self.dtype), sin.to(self.dtype))
        cos, sin = self._rope_cache[key]
        return Ctx(layer, start, length, state, extra={"cos": cos, "sin": sin, "sliding": sliding})

    def expert_weights(self, layer: int):
        experts = self.w[layer].experts
        if self.eager_experts:
            return lambda e: experts[e]
        return lambda e: experts[e].weights(self.dtype)

    def component(self, layer: int, name: str):
        cfg, w, eps = self.cfg, self.w[layer], self.cfg.layernorm_epsilon
        norm = lambda wt: lambda ctx, x: rms_norm(x, wt, eps)  # noqa: E731
        table = {
            "attn_norm": norm(w.input_norm),
            "attention": lambda ctx, x: self._attention(layer, ctx, x),
            "attn_residual": lambda ctx, a, b: a + b,
            "ffn_norm": norm(w.post_attn_norm),
        }
        if cfg.is_moe(layer):
            table.update(
                router=lambda ctx, x: dense_routing(*route(x, w.router_w, w.router_bias, cfg), cfg.n_routed_experts),
                experts=lambda ctx, x, r: experts_forward(x, r, self.expert_weights(layer)),
                ffn_residual=lambda ctx, a, b: a + b,
            )
        else:
            table.update(
                mlp=lambda ctx, x: swiglu_mlp(x, w.w_gate, w.w_up, w.w_down),
                mlp_residual=lambda ctx, a, b: a + b,
            )
        return table[name]

    def _attention(self, layer: int, ctx: Ctx, x: torch.Tensor) -> torch.Tensor:
        cfg, w = self.cfg, self.w[layer]
        s = x.shape[0]
        hq, hkv, d, dv = cfg.attn_dims(layer)
        cos, sin = ctx.extra["cos"], ctx.extra["sin"]
        qkv = F.linear(x, w.wqkv)
        q, k, v = qkv.split([hq * d, hkv * d, hkv * dv], dim=-1)
        v = v.reshape(s, hkv, dv)
        if cfg.attention_value_scale is not None:
            v = v * cfg.attention_value_scale
        q = apply_partial_rope(q.reshape(s, hq, d), cos, sin).transpose(0, 1)
        k = apply_partial_rope(k.reshape(s, hkv, d), cos, sin).transpose(0, 1)
        st, start = ctx.state, ctx.start
        end = start + s
        st.k[layer][:, start:end] = k
        st.v[layer][:, start:end] = v.transpose(0, 1)
        window = cfg.sliding_window if ctx.extra["sliding"] else None
        a = chunk_attention(q, st.k[layer][:, :end], st.v[layer][:, :end], start, d**-0.5, window, w.sink)
        return F.linear(a.transpose(0, 1).reshape(s, hq * dv), w.wo)

    @torch.no_grad()
    def forward_chunk(self, tokens, start, state, rec=noop, logits_last_n=0):
        """tokens [S] int at absolute positions [start, start+S). Returns (final_norm [S, H], logits [n, V] | None)."""
        tokens = tokens.long()
        h = F.embedding(tokens, self.embed)
        rec("embed", h)
        for i in self.layer_ids:
            ctx = self.chunk_context(i, start, tokens.shape[0], state)
            h = run_block(self.block_graph(i), lambda name, i=i: self.component(i, name), ctx, h, rec, prefix=f"L{i}.")
        out = rms_norm(h, self.final_norm_w, self.cfg.layernorm_epsilon)
        rec("final_norm", out)
        logits = None
        if logits_last_n and self.lm_head is not None:
            logits = self.logits(out[-logits_last_n:])
            rec("logits", logits)
        return out, logits

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden, self.lm_head)
