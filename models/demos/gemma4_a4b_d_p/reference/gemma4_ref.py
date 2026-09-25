# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Standalone torch CPU reference for Gemma-4 26B-A4B (text) chunked prefill.

Independent of HF transformers (HF is only the oracle in check_hf). Loads the safetensors checkpoint directly and
runs chunked causal prefill with an explicit full-length per-layer KV state. Every block runs through
``run_block``, so the block graph below is the code.

Block (30 layers, two attention types, every layer has dense MLP + MoE in parallel):
    in
    attn_norm      = rms(in) * w_in                                  input_layernorm
    attn_out       = attention(attn_norm)   [stateful]               q/k/v proj, q/k norm, v norm, RoPE, SDPA, o_proj
    attn_post_norm = rms(attn_out) * w                               post_attention_layernorm
    h_mid          = in + attn_post_norm
    ffn_norm       = rms(h_mid) * w                                  pre_feedforward_layernorm
    mlp_out        = down(gelu_tanh(gate(x)) * up(x))                dense MLP (2112)
    mlp_post_norm  = rms(mlp_out) * w                                post_feedforward_layernorm_1
    router         = dense routing weights [S, 128] from h_mid       (top-8, renormalized, * per_expert_scale)
    moe_norm       = rms(h_mid) * w                                  pre_feedforward_layernorm_2
    experts_out    = sum_e router[:, e] * expert_e(moe_norm)         128 experts (704), gelu_tanh
    moe_post_norm  = rms(experts_out) * w                            post_feedforward_layernorm_2
    ffn_sum        = mlp_post_norm + moe_post_norm
    ffn_out        = rms(ffn_sum) * w                                post_feedforward_layernorm
    out            = (h_mid + ffn_out) * layer_scalar

Attention:
    sliding (window 1024): 16 q heads x 256, 8 KV heads x 256, RoPE theta 1e4 (full rotate-half)
    global:                16 q heads x 512, 2 KV heads x 512, no v_proj: V = v_norm(k_proj(x)) (unscaled RMS),
                           K = rope(k_norm(k_proj(x))), proportional RoPE theta 1e6 on 25% (dims [0:64] + [256:320])
    attention scale 1.0 (q_norm/k_norm replace 1/sqrt(d)); softmax fp32.
State: key / value per layer, [Hkv, max_seq, D], K post-norm post-RoPE, V post-v_norm.
Embedding: embed[tokens] * sqrt(hidden). Logits: tied embedding, then 30 * tanh(logits / 30).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from safetensors import safe_open

from models.demos.common.bringup.reference.interface import Ctx, Step, noop, run_block

PREFIX = "model.language_model."


@dataclass
class Gemma4TextConfig:
    hidden_size: int = 2816
    intermediate_size: int = 2112
    moe_intermediate_size: int = 704
    num_hidden_layers: int = 30
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    num_global_key_value_heads: int = 2
    head_dim: int = 256
    global_head_dim: int = 512
    num_experts: int = 128
    top_k_experts: int = 8
    sliding_window: int = 1024
    rms_norm_eps: float = 1e-6
    vocab_size: int = 262144
    final_logit_softcapping: float | None = 30.0
    attention_k_eq_v: bool = True
    layer_types: tuple = ()
    rope_parameters: dict | None = None

    @classmethod
    def from_json(cls, path: str) -> "Gemma4TextConfig":
        with open(path) as f:
            raw = json.load(f)
        raw = raw.get("text_config", raw)
        known = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        cfg = cls(**known)
        cfg.layer_types = tuple(cfg.layer_types)
        assert raw.get("hidden_size_per_layer_input", 0) == 0, "per-layer inputs not supported"
        assert raw.get("num_kv_shared_layers", 0) == 0, "kv-shared layers not supported"
        assert raw.get("hidden_activation", "gelu_pytorch_tanh") == "gelu_pytorch_tanh"
        return cfg

    def is_sliding(self, i: int) -> bool:
        return self.layer_types[i] == "sliding_attention"

    def attn_dims(self, i: int) -> tuple[int, int]:
        """(num kv heads, head dim) of layer i."""
        if self.is_sliding(i):
            return self.num_key_value_heads, self.head_dim
        return (self.num_global_key_value_heads if self.attention_k_eq_v else self.num_key_value_heads), (
            self.global_head_dim or self.head_dim
        )


class WeightLoader:
    """Lazy safetensors accessor keyed by checkpoint tensor name."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        with open(os.path.join(model_path, "model.safetensors.index.json")) as f:
            self.weight_map: dict[str, str] = json.load(f)["weight_map"]
        self._handles = {}

    def get(self, name: str) -> torch.Tensor:
        fname = self.weight_map[name]
        if fname not in self._handles:
            self._handles[fname] = safe_open(os.path.join(self.model_path, fname), framework="pt")
        return self._handles[fname].get_tensor(name)

    def has(self, name: str) -> bool:
        return name in self.weight_map


# --------------------------------------------------------------------------------------
# Functional building blocks (the per-op goldens)
# --------------------------------------------------------------------------------------


def rms_norm(x: torch.Tensor, w: torch.Tensor | None, eps: float) -> torch.Tensor:
    """Gemma-4 RMSNorm: x * (mean(x^2) + eps)^-0.5 * w (plain w, not 1 + w); w=None for the unscaled variant."""
    xf = x.float()
    y = xf * torch.pow(xf.pow(2).mean(-1, keepdim=True) + eps, -0.5)
    if w is not None:
        y = y * w.float()
    return y.to(x.dtype)


def rope_inv_freq(cfg: Gemma4TextConfig, sliding: bool) -> tuple[torch.Tensor, int]:
    """Inverse frequencies (length head_dim/2) and head_dim. Global layers: proportional RoPE, zero freqs past 25%."""
    params = cfg.rope_parameters[("sliding" if sliding else "full") + "_attention"]
    base = params["rope_theta"]
    if sliding:
        dim = cfg.head_dim
        inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        return inv, dim
    dim = cfg.global_head_dim or cfg.head_dim
    assert params.get("rope_type", "default") == "proportional" and params.get("factor", 1.0) == 1.0
    angles = int(params.get("partial_rotary_factor", 1.0) * dim // 2)
    inv = 1.0 / (base ** (torch.arange(0, 2 * angles, 2, dtype=torch.int64).float() / dim))
    inv = torch.cat([inv, torch.zeros(dim // 2 - angles)])
    return inv, dim


def rope_cos_sin(positions: torch.Tensor, inv_freq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate-half tables [S, D]: cat(freqs, freqs)."""
    freqs = positions.float()[:, None] * inv_freq[None, :].float()
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [S, H, D]; cos/sin: [S, D]."""
    return x * cos[:, None] + rotate_half(x) * sin[:, None]


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


def gated_mlp(x, w_gate, w_up, w_down):
    return F.linear(gelu_tanh(F.linear(x, w_gate)) * F.linear(x, w_up), w_down)


def chunk_attention(
    q: torch.Tensor, k_all: torch.Tensor, v_all: torch.Tensor, start: int, window: int | None, q_block: int = 1024
) -> torch.Tensor:
    """Causal (optionally sliding-window) GQA attention, scale 1.0, for queries at [start, start+Sq).

    q: [Hq, Sq, D]; k_all / v_all: [Hkv, >= start+Sq, D] (state prefix + this chunk). Key range per query block is
    bounded by the window, so a chunk costs O(Sq * min(window, pos)) and never recomputes the prefix.
    """
    hq, sq, d = q.shape
    hkv = k_all.shape[0]
    rep = hq // hkv
    out = torch.empty_like(q)
    qg = q.view(hkv, rep, sq, d)
    og = out.view(hkv, rep, sq, d)
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
        og[:, :, qs:qe] = F.scaled_dot_product_attention(qg[:, :, qs:qe], k, v, attn_mask=mask, scale=1.0)
    return out


def route(x: torch.Tensor, w: "LayerWeights", cfg: Gemma4TextConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Gemma-4 router on the residual stream: returns (top-k weights [S, K], top-k expert ids [S, K])."""
    h = rms_norm(x, None, cfg.rms_norm_eps)
    h = h * w.router_scale * (cfg.hidden_size**-0.5)
    probs = F.softmax(F.linear(h, w.router_proj), dim=-1)
    tw, ti = torch.topk(probs, k=cfg.top_k_experts, dim=-1)
    tw = tw / tw.sum(dim=-1, keepdim=True)
    tw = tw * w.per_expert_scale[ti]
    return tw, ti


def dense_routing(tw: torch.Tensor, ti: torch.Tensor, num_experts: int) -> torch.Tensor:
    """[S, E] routing matrix: the top-k weight at each selected expert, 0 elsewhere."""
    r = torch.zeros(tw.shape[0], num_experts, dtype=tw.dtype)
    return r.scatter_(1, ti, tw)


def experts_forward(x: torch.Tensor, routing: torch.Tensor, gate_up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
    """sum over selected experts of routing[t, e] * expert_e(x[t]); tokens grouped per expert, experts in id order
    (the HF accumulation order). gate_up: [E, 2I, H] (gate rows first); down: [E, H, I]."""
    out = torch.zeros_like(x)
    tok_all, exp_all = (routing != 0).nonzero(as_tuple=True)
    order = torch.argsort(exp_all, stable=True)
    tok_all, exp_all = tok_all[order], exp_all[order]
    experts, counts = torch.unique_consecutive(exp_all, return_counts=True)
    off = 0
    for e, n in zip(experts.tolist(), counts.tolist()):
        tok = tok_all[off : off + n]
        off += n
        g, u = F.linear(x[tok], gate_up[e]).chunk(2, dim=-1)
        y = F.linear(gelu_tanh(g) * u, down[e])
        out.index_add_(0, tok, y * routing[tok, e, None])
    return out


# --------------------------------------------------------------------------------------
# Weights
# --------------------------------------------------------------------------------------


@dataclass
class LayerWeights:
    input_norm: torch.Tensor
    post_attn_norm: torch.Tensor
    pre_ffn_norm: torch.Tensor
    post_ffn_norm_1: torch.Tensor
    pre_ffn_norm_2: torch.Tensor
    post_ffn_norm_2: torch.Tensor
    post_ffn_norm: torch.Tensor
    layer_scalar: torch.Tensor
    wq: torch.Tensor
    wk: torch.Tensor
    wv: torch.Tensor | None  # None on global layers (V from k_proj)
    wo: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    w_gate: torch.Tensor
    w_up: torch.Tensor
    w_down: torch.Tensor
    router_proj: torch.Tensor
    router_scale: torch.Tensor
    per_expert_scale: torch.Tensor
    e_gate_up: torch.Tensor  # [E, 2I, H]
    e_down: torch.Tensor  # [E, H, I]


def load_layer(loader: WeightLoader, i: int, dtype=torch.float32) -> LayerWeights:
    p = f"{PREFIX}layers.{i}."
    g = lambda n: loader.get(p + n).to(dtype)  # noqa: E731
    return LayerWeights(
        input_norm=g("input_layernorm.weight"),
        post_attn_norm=g("post_attention_layernorm.weight"),
        pre_ffn_norm=g("pre_feedforward_layernorm.weight"),
        post_ffn_norm_1=g("post_feedforward_layernorm_1.weight"),
        pre_ffn_norm_2=g("pre_feedforward_layernorm_2.weight"),
        post_ffn_norm_2=g("post_feedforward_layernorm_2.weight"),
        post_ffn_norm=g("post_feedforward_layernorm.weight"),
        layer_scalar=g("layer_scalar"),
        wq=g("self_attn.q_proj.weight"),
        wk=g("self_attn.k_proj.weight"),
        wv=g("self_attn.v_proj.weight") if loader.has(p + "self_attn.v_proj.weight") else None,
        wo=g("self_attn.o_proj.weight"),
        q_norm=g("self_attn.q_norm.weight"),
        k_norm=g("self_attn.k_norm.weight"),
        w_gate=g("mlp.gate_proj.weight"),
        w_up=g("mlp.up_proj.weight"),
        w_down=g("mlp.down_proj.weight"),
        router_proj=g("router.proj.weight"),
        router_scale=g("router.scale"),
        per_expert_scale=g("router.per_expert_scale"),
        e_gate_up=g("experts.gate_up_proj"),
        e_down=g("experts.down_proj"),
    )


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------


class State:
    """Full-length KV per layer: k[i], v[i] of shape [Hkv, max_seq, D] (K post-norm + RoPE, V post-v_norm)."""

    def __init__(self, cfg: Gemma4TextConfig, layers: list[int], max_seq: int, dtype=torch.float32):
        self.k, self.v = {}, {}
        for i in layers:
            hkv, d = cfg.attn_dims(i)
            self.k[i] = torch.zeros(hkv, max_seq, d, dtype=dtype)
            self.v[i] = torch.zeros(hkv, max_seq, d, dtype=dtype)
        self.max_seq = max_seq


BLOCK_GRAPH = [
    Step("attn_norm", ("in",), "attn_norm", "norm"),
    Step("attention", ("attn_norm",), "attn_out", "attention", stateful=True),
    Step("post_attn_norm", ("attn_out",), "attn_post_norm", "norm"),
    Step("attn_residual", ("in", "attn_post_norm"), "h_mid", "residual"),
    Step("ffn_norm", ("h_mid",), "ffn_norm", "norm"),
    Step("mlp", ("ffn_norm",), "mlp_out", "mlp"),
    Step("post_mlp_norm", ("mlp_out",), "mlp_post_norm", "norm"),
    Step("router", ("h_mid",), "router", "router"),
    Step("moe_norm", ("h_mid",), "moe_norm", "norm"),
    Step("experts", ("moe_norm", "router"), "experts_out", "moe"),
    Step("post_moe_norm", ("experts_out",), "moe_post_norm", "norm"),
    Step("ffn_combine", ("mlp_post_norm", "moe_post_norm"), "ffn_sum", "residual"),
    Step("post_ffn_norm", ("ffn_sum",), "ffn_out", "norm"),
    Step("ffn_residual", ("h_mid", "ffn_out"), "out", "residual"),
]


class Gemma4Reference:
    """Implements models/demos/common/bringup/reference/interface.py. Keeps the requested layers resident."""

    def __init__(self, model_path: str, layers: list[int] | None = None, dtype=torch.float32, lm_head: bool = True):
        self.cfg = Gemma4TextConfig.from_json(os.path.join(model_path, "config.json"))
        self.loader = WeightLoader(model_path)
        self.dtype = dtype
        self.layer_ids = list(range(self.cfg.num_hidden_layers)) if layers is None else list(layers)
        self.embed = self.loader.get(PREFIX + "embed_tokens.weight").to(dtype)
        self.embed_scale = torch.tensor(self.cfg.hidden_size**0.5, dtype=torch.float32).to(dtype)
        self.final_norm_w = self.loader.get(PREFIX + "norm.weight").to(dtype)
        self.lm_head = self.embed if lm_head else None  # tied
        self.w = {i: load_layer(self.loader, i, dtype) for i in self.layer_ids}
        self.inv_freq = {s: rope_inv_freq(self.cfg, s)[0] for s in (True, False)}
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
        return list(BLOCK_GRAPH)

    def chunk_context(self, layer: int, start: int, length: int, state) -> Ctx:
        sliding = self.cfg.is_sliding(layer)
        key = (sliding, start, length)
        if key not in self._rope_cache:
            self._rope_cache = {k: v for k, v in self._rope_cache.items() if k[1:] == key[1:]}
            cos, sin = rope_cos_sin(torch.arange(start, start + length), self.inv_freq[sliding])
            self._rope_cache[key] = (cos.to(self.dtype), sin.to(self.dtype))
        cos, sin = self._rope_cache[key]
        return Ctx(layer, start, length, state, extra={"cos": cos, "sin": sin, "sliding": sliding})

    def component(self, layer: int, name: str):
        cfg, w, eps = self.cfg, self.w[layer], self.cfg.rms_norm_eps
        norm = lambda wt: lambda ctx, x: rms_norm(x, wt, eps)  # noqa: E731
        table = {
            "attn_norm": norm(w.input_norm),
            "attention": lambda ctx, x: self._attention(layer, ctx, x),
            "post_attn_norm": norm(w.post_attn_norm),
            "attn_residual": lambda ctx, a, b: a + b,
            "ffn_norm": norm(w.pre_ffn_norm),
            "mlp": lambda ctx, x: gated_mlp(x, w.w_gate, w.w_up, w.w_down),
            "post_mlp_norm": norm(w.post_ffn_norm_1),
            "router": lambda ctx, x: dense_routing(*route(x, w, cfg), cfg.num_experts),
            "moe_norm": norm(w.pre_ffn_norm_2),
            "experts": lambda ctx, x, r: experts_forward(x, r, w.e_gate_up, w.e_down),
            "post_moe_norm": norm(w.post_ffn_norm_2),
            "ffn_combine": lambda ctx, a, b: a + b,
            "post_ffn_norm": norm(w.post_ffn_norm),
            "ffn_residual": lambda ctx, a, b: (a + b) * w.layer_scalar,
        }
        return table[name]

    def _attention(self, layer: int, ctx: Ctx, x: torch.Tensor) -> torch.Tensor:
        cfg, w, eps = self.cfg, self.w[layer], self.cfg.rms_norm_eps
        s = x.shape[0]
        hkv, d = cfg.attn_dims(layer)
        cos, sin = ctx.extra["cos"], ctx.extra["sin"]
        q = rms_norm(F.linear(x, w.wq).view(s, cfg.num_attention_heads, d), w.q_norm, eps)
        q = apply_rope(q, cos, sin).transpose(0, 1)
        k_raw = F.linear(x, w.wk).view(s, hkv, d)
        v_raw = F.linear(x, w.wv).view(s, hkv, d) if w.wv is not None else k_raw
        k = apply_rope(rms_norm(k_raw, w.k_norm, eps), cos, sin).transpose(0, 1)
        v = rms_norm(v_raw, None, eps).transpose(0, 1)
        st, start = ctx.state, ctx.start
        end = start + s
        st.k[layer][:, start:end] = k
        st.v[layer][:, start:end] = v
        window = cfg.sliding_window if ctx.extra["sliding"] else None
        a = chunk_attention(q, st.k[layer][:, :end], st.v[layer][:, :end], start, window)
        return F.linear(a.transpose(0, 1).reshape(s, -1), w.wo)

    @torch.no_grad()
    def forward_chunk(self, tokens, start, state, rec=noop, logits_last_n=0):
        """tokens [S] int at absolute positions [start, start+S). Returns (final_norm [S, H], logits [n, V] | None)."""
        tokens = tokens.long()
        h = F.embedding(tokens, self.embed) * self.embed_scale
        rec("embed", h)
        for i in self.layer_ids:
            ctx = self.chunk_context(i, start, tokens.shape[0], state)
            h = run_block(self.block_graph(i), lambda name, i=i: self.component(i, name), ctx, h, rec, prefix=f"L{i}.")
        out = rms_norm(h, self.final_norm_w, self.cfg.rms_norm_eps)
        rec("final_norm", out)
        logits = None
        if logits_last_n and self.lm_head is not None:
            logits = self.logits(out[-logits_last_n:])
            rec("logits", logits)
        return out, logits

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = F.linear(hidden, self.lm_head)
        cap = self.cfg.final_logit_softcapping
        if cap:
            logits = torch.tanh(logits / cap) * cap
        return logits
