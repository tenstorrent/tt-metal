# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Standalone torch reference for ERNIE-4.5-21B-A3B prefill.

Independent of HF transformers (which is only used as an oracle in tests). Loads the
safetensors checkpoint lazily, one layer at a time, and runs chunked causal prefill
with an explicit per-layer KV cache so every intermediate is observable.

Block structure (28 decoder layers):
    h   = embed(tokens)
    for layer i:
        x   = rms_norm(h)                      L{i}.attn_norm
        q,k,v = x @ Wq, Wk, Wv                 L{i}.q / .k / .v   (k after RoPE = cache content)
        q,k = rope_interleaved(q,k, pos)
        a   = GQA causal attention(q, K[:end], V[:end])
        h   = h + a @ Wo                       L{i}.attn_out, L{i}.h_mid
        x   = rms_norm(h)                      L{i}.ffn_norm
        layer 0      : dense SwiGLU(12288)
        layers 1..27 : MoE = shared SwiGLU(2*1536) + sum_topk6 w_e * SwiGLU_e(1536)
        h   = h + mlp                          L{i}.out
    out = rms_norm(h); logits = out @ embed^T (tied)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from safetensors import safe_open

HF_MODEL_ID = "baidu/ERNIE-4.5-21B-A3B-PT"


@dataclass
class ErnieConfig:
    hidden_size: int = 2560
    intermediate_size: int = 12288
    moe_intermediate_size: int = 1536
    num_hidden_layers: int = 28
    num_attention_heads: int = 20
    num_key_value_heads: int = 4
    head_dim: int = 128
    moe_num_experts: int = 64
    moe_k: int = 6
    moe_num_shared_experts: int = 2
    moe_layer_start_index: int = 1
    moe_layer_end_index: int = 27
    moe_layer_interval: int = 1
    moe_norm_min: float = 1e-12
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    vocab_size: int = 103424
    tie_word_embeddings: bool = True

    @classmethod
    def from_json(cls, path: str) -> "ErnieConfig":
        with open(path) as f:
            raw = json.load(f)
        known = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        cfg = cls(**known)
        cfg.head_dim = raw.get("head_dim") or cfg.hidden_size // cfg.num_attention_heads
        return cfg

    def is_moe_layer(self, i: int) -> bool:
        return (i + 1) % self.moe_layer_interval == 0 and self.moe_layer_start_index <= i <= self.moe_layer_end_index


def resolve_model_path(model_path: str | None = None) -> str:
    """Local snapshot dir of the HF checkpoint (env ERNIE_MODEL_PATH overrides)."""
    model_path = model_path or os.environ.get("ERNIE_MODEL_PATH")
    if model_path:
        return model_path
    from huggingface_hub import snapshot_download

    return snapshot_download(HF_MODEL_ID, local_files_only=True)


class WeightLoader:
    """Lazy safetensors accessor keyed by checkpoint tensor name."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        with open(os.path.join(model_path, "model.safetensors.index.json")) as f:
            self.weight_map: dict[str, str] = json.load(f)["weight_map"]
        self._handles = {}

    def _handle(self, fname: str):
        if fname not in self._handles:
            self._handles[fname] = safe_open(os.path.join(self.model_path, fname), framework="pt")
        return self._handles[fname]

    def get(self, name: str) -> torch.Tensor:
        return self._handle(self.weight_map[name]).get_tensor(name)

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def names(self, prefix: str) -> list[str]:
        return [n for n in self.weight_map if n.startswith(prefix)]


# --------------------------------------------------------------------------------------
# Functional building blocks (shared with unit tests as the per-op golden)
# --------------------------------------------------------------------------------------


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (w.float() * xf).to(x.dtype)


def rope_cos_sin(positions: torch.Tensor, head_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Interleaved-pair RoPE tables, shape [S, head_dim]: element 2j and 2j+1 share freq j."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
    freqs = positions.float()[:, None] * inv_freq[None, :]  # [S, head_dim/2]
    cos = freqs.cos().repeat_interleave(2, dim=-1)
    sin = freqs.sin().repeat_interleave(2, dim=-1)
    return cos, sin


def rotate_interleaved(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [H, S, D]; cos/sin: [S, D]. Computed in fp32 like HF."""
    return (x.float() * cos + rotate_interleaved(x).float() * sin).to(x.dtype)


def swiglu(x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor) -> torch.Tensor:
    return F.linear(F.silu(F.linear(x, w_gate)) * F.linear(x, w_up), w_down)


def gqa_chunk_attention(
    q: torch.Tensor, k_all: torch.Tensor, v_all: torch.Tensor, start_pos: int, q_block: int = 1024
) -> torch.Tensor:
    """Causal GQA attention for a chunk of queries at absolute positions [start_pos, start_pos+Sq).

    q: [Hq, Sq, D]; k_all/v_all: [Hkv, start_pos+Sq, D] (cache prefix + this chunk).
    Processes query sub-blocks so memory stays bounded at 55k context.
    """
    hq, sq, d = q.shape
    hkv = k_all.shape[0]
    rep = hq // hkv
    scale = d**-0.5
    out = torch.empty_like(q)
    qg = q.view(hkv, rep, sq, d)
    for qs in range(0, sq, q_block):
        qe = min(sq, qs + q_block)
        kv_end = start_pos + qe  # keys beyond the last query in this block are never visible
        k = k_all[:, :kv_end]
        v = v_all[:, :kv_end]
        q_pos = torch.arange(start_pos + qs, start_pos + qe)
        k_pos = torch.arange(kv_end)
        mask = k_pos[None, :] <= q_pos[:, None]  # [bq, kv_end] True = visible
        o = F.scaled_dot_product_attention(
            qg[:, :, qs:qe], k[:, None], v[:, None], attn_mask=mask, scale=scale
        )  # [hkv, rep, bq, d]
        out.view(hkv, rep, sq, d)[:, :, qs:qe] = o
    return out


def moe_route(
    x: torch.Tensor, w_router: torch.Tensor, e_bias: torch.Tensor, top_k: int, norm_min: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Softmax router (fp32). Bias only affects *selection*; weights are renormalized softmax probs."""
    logits = F.linear(x.float(), w_router.float())
    probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    _, idx = torch.topk(probs + e_bias.float().view(-1), top_k, dim=-1)
    w = torch.gather(probs, -1, idx)
    w = w / torch.clamp(w.sum(-1, keepdim=True), min=norm_min)
    return logits, idx, w


# --------------------------------------------------------------------------------------
# Layer weights
# --------------------------------------------------------------------------------------


@dataclass
class LayerWeights:
    attn_norm: torch.Tensor
    ffn_norm: torch.Tensor
    wq: torch.Tensor
    wk: torch.Tensor
    wv: torch.Tensor
    wo: torch.Tensor
    is_moe: bool
    # dense MLP (layer 0) or shared experts (MoE layers)
    w_gate: torch.Tensor
    w_up: torch.Tensor
    w_down: torch.Tensor
    # routed experts (MoE layers only)
    router: torch.Tensor | None = None
    e_bias: torch.Tensor | None = None
    e_gate: torch.Tensor | None = None  # [E, I, H]
    e_up: torch.Tensor | None = None  # [E, I, H]
    e_down: torch.Tensor | None = None  # [E, H, I]


def load_layer(loader: WeightLoader, cfg: ErnieConfig, i: int, dtype=torch.float32) -> LayerWeights:
    p = f"model.layers.{i}."
    g = lambda n: loader.get(p + n).to(dtype)  # noqa: E731
    is_moe = cfg.is_moe_layer(i)
    kw = dict(
        attn_norm=g("input_layernorm.weight"),
        ffn_norm=g("post_attention_layernorm.weight"),
        wq=g("self_attn.q_proj.weight"),
        wk=g("self_attn.k_proj.weight"),
        wv=g("self_attn.v_proj.weight"),
        wo=g("self_attn.o_proj.weight"),
        is_moe=is_moe,
    )
    if not is_moe:
        kw.update(w_gate=g("mlp.gate_proj.weight"), w_up=g("mlp.up_proj.weight"), w_down=g("mlp.down_proj.weight"))
    else:
        kw.update(
            w_gate=g("mlp.shared_experts.gate_proj.weight"),
            w_up=g("mlp.shared_experts.up_proj.weight"),
            w_down=g("mlp.shared_experts.down_proj.weight"),
            router=loader.get(p + "mlp.gate.weight").float(),
            e_bias=_load_e_bias(loader, p),
        )
        E = cfg.moe_num_experts
        kw["e_gate"] = torch.stack([g(f"mlp.experts.{e}.gate_proj.weight") for e in range(E)])
        kw["e_up"] = torch.stack([g(f"mlp.experts.{e}.up_proj.weight") for e in range(E)])
        kw["e_down"] = torch.stack([g(f"mlp.experts.{e}.down_proj.weight") for e in range(E)])
    return LayerWeights(**kw)


def _load_e_bias(loader: WeightLoader, p: str) -> torch.Tensor:
    for n in ("mlp.moe_statics.e_score_correction_bias", "mlp.gate.moe_statics.e_score_correction_bias"):
        if loader.has(p + n):
            return loader.get(p + n).float().view(-1)
    raise KeyError(f"no e_score_correction_bias under {p}")


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------

Recorder = Callable[[str, torch.Tensor], None]


def _noop(name: str, t: torch.Tensor) -> None:
    pass


class KVCache:
    """Per-layer contiguous cache, K stored post-RoPE. k[i], v[i]: [Hkv, max_seq, D]."""

    def __init__(self, cfg: ErnieConfig, max_seq: int, dtype=torch.float32, layers: list[int] | None = None):
        layers = list(range(cfg.num_hidden_layers)) if layers is None else layers
        shape = (cfg.num_key_value_heads, max_seq, cfg.head_dim)
        self.k = {i: torch.zeros(shape, dtype=dtype) for i in layers}
        self.v = {i: torch.zeros(shape, dtype=dtype) for i in layers}
        self.max_seq = max_seq


class ErnieDecoderLayer:
    def __init__(self, cfg: ErnieConfig, idx: int, w: LayerWeights):
        self.cfg, self.idx, self.w = cfg, idx, w

    def attention(self, x, start_pos, cos, sin, cache: KVCache, rec: Recorder):
        cfg, w = self.cfg, self.w
        s = x.shape[0]
        q = F.linear(x, w.wq).view(s, cfg.num_attention_heads, cfg.head_dim).transpose(0, 1)
        k = F.linear(x, w.wk).view(s, cfg.num_key_value_heads, cfg.head_dim).transpose(0, 1)
        v = F.linear(x, w.wv).view(s, cfg.num_key_value_heads, cfg.head_dim).transpose(0, 1)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        rec(f"L{self.idx}.q", q)
        rec(f"L{self.idx}.k", k)
        rec(f"L{self.idx}.v", v)
        end = start_pos + s
        cache.k[self.idx][:, start_pos:end] = k.to(cache.k[self.idx].dtype)
        cache.v[self.idx][:, start_pos:end] = v.to(cache.v[self.idx].dtype)
        kk = cache.k[self.idx][:, :end].to(q.dtype)
        vv = cache.v[self.idx][:, :end].to(q.dtype)
        a = gqa_chunk_attention(q, kk, vv, start_pos)  # [Hq, S, D]
        rec(f"L{self.idx}.sdpa", a)
        a = a.transpose(0, 1).reshape(s, -1)
        return F.linear(a, w.wo)

    def mlp(self, x, rec: Recorder):
        w, cfg = self.w, self.cfg
        if not w.is_moe:
            return swiglu(x, w.w_gate, w.w_up, w.w_down)
        shared = swiglu(x, w.w_gate, w.w_up, w.w_down)
        logits, idx, rw = moe_route(x, w.router, w.e_bias, cfg.moe_k, cfg.moe_norm_min)
        rec(f"L{self.idx}.router_logits", logits)
        rec(f"L{self.idx}.topk_idx", idx)
        rec(f"L{self.idx}.topk_w", rw)
        routed = torch.zeros_like(x)
        for e in torch.unique(idx).tolist():
            tok, slot = torch.where(idx == e)
            y = swiglu(x[tok], w.e_gate[e], w.e_up[e], w.e_down[e])
            routed.index_add_(0, tok, y * rw[tok, slot, None].to(y.dtype))
        rec(f"L{self.idx}.shared_out", shared)
        rec(f"L{self.idx}.routed_out", routed)
        return shared + routed

    def forward(self, h, start_pos, cos, sin, cache: KVCache, rec: Recorder = _noop):
        cfg, w, i = self.cfg, self.w, self.idx
        rec(f"L{i}.in", h)
        x = rms_norm(h, w.attn_norm, cfg.rms_norm_eps)
        rec(f"L{i}.attn_norm", x)
        a = self.attention(x, start_pos, cos, sin, cache, rec)
        rec(f"L{i}.attn_out", a)
        h = h + a
        rec(f"L{i}.h_mid", h)
        x = rms_norm(h, w.ffn_norm, cfg.rms_norm_eps)
        rec(f"L{i}.ffn_norm", x)
        m = self.mlp(x, rec)
        rec(f"L{i}.mlp_out", m)
        h = h + m
        rec(f"L{i}.out", h)
        return h


class ErnieReference:
    """Full-model CPU reference. Keeps weights of the requested layers resident."""

    def __init__(
        self, model_path: str | None = None, dtype=torch.float32, layers: list[int] | None = None, lm_head=True
    ):
        self.model_path = resolve_model_path(model_path)
        self.cfg = ErnieConfig.from_json(os.path.join(self.model_path, "config.json"))
        self.loader = WeightLoader(self.model_path)
        self.dtype = dtype
        self.layer_ids = list(range(self.cfg.num_hidden_layers)) if layers is None else layers
        self.embed = self.loader.get("model.embed_tokens.weight").to(dtype)
        self.final_norm = self.loader.get("model.norm.weight").to(dtype)
        self.lm_head = None
        if lm_head:
            name = "lm_head.weight"
            self.lm_head = (
                self.loader.get(name).to(dtype)
                if (not self.cfg.tie_word_embeddings and self.loader.has(name))
                else self.embed
            )
        self.layers = {
            i: ErnieDecoderLayer(self.cfg, i, load_layer(self.loader, self.cfg, i, dtype)) for i in self.layer_ids
        }

    def new_cache(self, max_seq: int, dtype=None) -> KVCache:
        return KVCache(self.cfg, max_seq, dtype or self.dtype, self.layer_ids)

    @torch.no_grad()
    def forward_chunk(
        self,
        tokens: torch.Tensor,
        start_pos: int,
        cache: KVCache,
        rec: Recorder = _noop,
        logits_last_n: int = 0,
    ):
        """One prefill chunk. tokens: [S] int64 at absolute positions [start_pos, start_pos+S).
        Returns (final_normed_hidden [S, H], logits [n, V] or None)."""
        cfg = self.cfg
        s = tokens.shape[0]
        h = F.embedding(tokens, self.embed)
        rec("embed", h)
        cos, sin = rope_cos_sin(torch.arange(start_pos, start_pos + s), cfg.head_dim, cfg.rope_theta)
        for i in self.layer_ids:
            h = self.layers[i].forward(h, start_pos, cos, sin, cache, rec)
        out = rms_norm(h, self.final_norm, cfg.rms_norm_eps)
        rec("final_norm", out)
        logits = None
        if logits_last_n and self.lm_head is not None:
            logits = F.linear(out[-logits_last_n:], self.lm_head)
            rec("logits", logits)
        return out, logits

    @torch.no_grad()
    def prefill(self, tokens: torch.Tensor, chunk_size: int, rec_factory=None, logits_last_n: int = 1):
        """Chunked prefill over the whole prompt. rec_factory(chunk_idx) -> Recorder."""
        cache = self.new_cache(tokens.shape[0])
        outs, logits = [], None
        for c, start in enumerate(range(0, tokens.shape[0], chunk_size)):
            rec = rec_factory(c) if rec_factory else _noop
            o, logits = self.forward_chunk(tokens[start : start + chunk_size], start, cache, rec, logits_last_n)
            outs.append(o)
        return torch.cat(outs), logits, cache


def load_book_tokens(tokenizer, n_tokens: int) -> torch.Tensor:
    """Real long text: A Tale of Two Cities (Project Gutenberg), shipped in the repo."""
    import bz2

    path = os.path.join(os.path.dirname(__file__), "../../../tt_transformers/tests/tale-of-two-cities.txt.bz2")
    with bz2.open(os.path.normpath(path), "rt", encoding="utf-8") as f:
        text = f.read()
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if tokenizer.bos_token_id is not None:
        ids = [tokenizer.bos_token_id] + ids
    assert len(ids) >= n_tokens, f"book has {len(ids)} tokens < {n_tokens}"
    return torch.tensor(ids[:n_tokens], dtype=torch.int64)


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.double().flatten()
    b = b.double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    den = a.norm() * b.norm()
    return float((a @ b) / den) if den > 0 else float(torch.equal(a, b))
