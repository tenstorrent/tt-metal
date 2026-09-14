# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Self-contained torch reference for Llama-3.1-8B prefill.

Purity contract (``deepseek_v3_d_p/reference/kda/README.md``): **torch only** — no ``ttnn``, no
device code, no HuggingFace import at module load. It is the oracle every ``*_vs_ref`` PCC test
compares against, so it must be constructible and runnable on a host with no accelerator.

Storage dtype is **fp16** throughout (recipe §4: references and goldens are fp16 regardless of the
checkpoint dtype and of the ttnn dtypes under test) — weights, activations, cos/sin and the K/V this
dumps are all ``torch.float16``. Reductions that HF also upcasts (the RMSNorm variance and the
attention softmax) are computed in fp32 and cast back, which is what ``modeling_llama`` does for an
fp16 model; doing them in fp16 would make the reference, not the device, the least accurate thing in
the comparison.

Layout conventions, which the device side deliberately does NOT share:

* ``cos``/``sin`` are **HF half-split**: ``[c0..c63, c0..c63]``, paired with ``rotate_half``.
  The device uses the Meta *interleaved* layout and ``ttnn.experimental.rotary_embedding_llama``.
  The two produce the same rotation applied to differently-ordered columns; ``utils.rope_layout``
  carries the permutation between them, and that is the ONLY place the two conventions meet.
* K dumped by :meth:`Attention.forward` is **post-RoPE**, V is **raw** — matching the golden trace's
  ``k_is_post_rope`` / ``v_is_raw`` metadata.

Upstream provenance: ``transformers.models.llama.modeling_llama`` (transformers 5.12.1) —
``LlamaRMSNorm``, ``LlamaRotaryEmbedding`` + ``_compute_llama3_parameters``, ``rotate_half`` /
``apply_rotary_pos_emb``, ``repeat_kv``, ``LlamaMLP``, ``LlamaAttention``, ``LlamaDecoderLayer``,
``LlamaModel``. ``tests/torch_ref/test_reference_llama.py`` pins this file against those classes.
"""

from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple

import torch
from torch import nn

from .config import LlamaConfig

REF_DTYPE = torch.float16


# --------------------------------------------------------------------------------------------
# RoPE (llama3 rescaling), HF half-split layout
# --------------------------------------------------------------------------------------------
def freqs_rounded_to_fp16() -> bool:
    """``LLAMA_ROPE_FREQ_FP16=1`` rounds the rope inverse frequencies to float16. **Off by default.**

    This is not how Llama-3.1 is defined: HuggingFace keeps ``inv_freq`` in fp32 whatever the model
    dtype, and rounding it produces a phase error that grows *linearly with position* rather than
    staying bounded — about 1.7 rad by token 10240 on the fastest unscaled frequency. The knob exists
    because the shipped golden trace was generated that way, and it is the switch that turns the
    device's per-layer K PCC against that trace from ~0.98 into ~0.9997. Both the reference and the
    device read this one function, so they can never be on opposite sides of it.
    See ``tests/torch_ref/test_golden_trace_rope_precision.py``.
    """
    return os.getenv("LLAMA_ROPE_FREQ_FP16", "0").strip().lower() in ("1", "true", "yes", "on")


def llama3_inv_freq(cfg: LlamaConfig) -> torch.Tensor:
    """Per-frequency inverse wavelengths after the llama3 piecewise rescaling.

    Anchor: ``transformers.modeling_rope_utils._compute_llama3_parameters``. Three regimes keyed on
    the wavelength: high frequency (wavelen < ctx/high_factor) untouched, low frequency
    (wavelen > ctx/low_factor) divided by ``factor``, and a linear ramp between them.
    """
    dim = cfg.head_dim
    base = cfg.rope_theta
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    rs = cfg.rope_scaling
    if not rs:
        return inv_freq
    assert rs["rope_type"] == "llama3", f"only llama3 rope scaling is implemented, got {rs['rope_type']}"
    factor = float(rs["factor"])
    low_f = float(rs["low_freq_factor"])
    high_f = float(rs["high_freq_factor"])
    old_ctx = float(rs["original_max_position_embeddings"])

    low_wavelen = old_ctx / low_f
    high_wavelen = old_ctx / high_f
    wavelen = 2 * math.pi / inv_freq

    scaled = inv_freq / factor
    smooth = (old_ctx / wavelen - low_f) / (high_f - low_f)
    smoothed = (1 - smooth) * scaled + smooth * inv_freq

    out = torch.where(wavelen > low_wavelen, scaled, inv_freq)
    is_medium = (wavelen >= high_wavelen) & (wavelen <= low_wavelen)
    out = torch.where(is_medium, smoothed, out)
    return out.to(torch.float16).double() if freqs_rounded_to_fp16() else out


def rope_cos_sin(cfg: LlamaConfig, seq_len: int, start_pos: int = 0, dtype=REF_DTYPE):
    """HF half-split cos/sin for positions ``[start_pos, start_pos + seq_len)``.

    Returns ``(cos, sin)``, each ``[seq_len, head_dim]`` with the second half duplicating the first
    — the layout ``apply_rope`` below pairs with ``rotate_half``.
    """
    inv_freq = llama3_inv_freq(cfg)
    t = torch.arange(start_pos, start_pos + seq_len, dtype=torch.float64)
    freqs = torch.outer(t, inv_freq)  # [seq, head_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``[b, heads, seq, head_dim]`` by the HF half-split ``cos``/``sin`` ``[seq, head_dim]``."""
    c = cos.to(torch.float32).unsqueeze(0).unsqueeze(0)
    s = sin.to(torch.float32).unsqueeze(0).unsqueeze(0)
    xf = x.to(torch.float32)
    return ((xf * c) + (rotate_half(xf) * s)).to(x.dtype)


# --------------------------------------------------------------------------------------------
# Blocks
# --------------------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Plain RMSNorm (``x * rsqrt(mean(x^2) + eps) * w``) — no Gemma ``(1 + w)`` fold."""

    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=REF_DTYPE))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.to(torch.float32)
        var = xf.pow(2).mean(-1, keepdim=True)
        normed = xf * torch.rsqrt(var + self.eps)
        return (normed * self.weight.to(torch.float32)).to(x.dtype)


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Llama's SwiGLU: ``silu(gate) * up``. No clamp, no alpha — those belong to gpt-oss/M3."""
    return (torch.nn.functional.silu(gate.to(torch.float32)) * up.to(torch.float32)).to(gate.dtype)


class MLP(nn.Module):
    """Dense SwiGLU FFN, 4096 -> 14336 -> 4096, no biases."""

    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        h, i = cfg.hidden_size, cfg.intermediate_size
        self.gate_proj = nn.Linear(h, i, bias=cfg.mlp_bias, dtype=REF_DTYPE)
        self.up_proj = nn.Linear(h, i, bias=cfg.mlp_bias, dtype=REF_DTYPE)
        self.down_proj = nn.Linear(i, h, bias=cfg.mlp_bias, dtype=REF_DTYPE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """``[b, n_kv, s, d]`` -> ``[b, n_kv*n_rep, s, d]`` (GQA group expansion)."""
    if n_rep == 1:
        return x
    b, n_kv, s, d = x.shape
    return x[:, :, None].expand(b, n_kv, n_rep, s, d).reshape(b, n_kv * n_rep, s, d)


def causal_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    """Causal GQA attention in fp32, returned in ``q``'s dtype.

    ``q`` is ``[b, n_q, s_q, d]`` and ``k``/``v`` are ``[b, n_kv, s_kv, d]`` with ``s_kv >= s_q``;
    the query at local index ``i`` is taken to be at absolute position ``s_kv - s_q + i`` so the same
    function serves both the whole-sequence case and a later chunk attending an earlier prefix.
    """
    n_rep = q.shape[1] // k.shape[1]
    kk, vv = repeat_kv(k, n_rep).to(torch.float32), repeat_kv(v, n_rep).to(torch.float32)
    qq = q.to(torch.float32)
    s_q, s_kv = qq.shape[-2], kk.shape[-2]
    offset = s_kv - s_q
    scores = torch.matmul(qq, kk.transpose(-1, -2)) * scale
    pos_q = torch.arange(s_q).unsqueeze(-1) + offset
    pos_k = torch.arange(s_kv).unsqueeze(0)
    scores = scores.masked_fill(pos_k > pos_q, float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), vv).to(q.dtype)


class Attention(nn.Module):
    """GQA attention: 32 q heads / 8 kv heads, head_dim 128, full RoPE, no QK-norm, no biases."""

    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_attention_heads * d, bias=cfg.attention_bias, dtype=REF_DTYPE)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * d, bias=cfg.attention_bias, dtype=REF_DTYPE)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * d, bias=cfg.attention_bias, dtype=REF_DTYPE)
        self.o_proj = nn.Linear(cfg.num_attention_heads * d, cfg.hidden_size, bias=cfg.attention_bias, dtype=REF_DTYPE)
        self.scale = d**-0.5

    def project(self, x: torch.Tensor):
        """``[b, s, hidden]`` -> head-split ``q, k, v`` (``[b, heads, s, head_dim]``), pre-RoPE."""
        b, s, _ = x.shape
        d = self.cfg.head_dim
        q = self.q_proj(x).view(b, s, self.cfg.num_attention_heads, d).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.cfg.num_key_value_heads, d).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.cfg.num_key_value_heads, d).transpose(1, 2)
        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """Returns ``(out, k_post_rope, v_raw)`` where K/V cover THIS chunk only.

        ``past_kv`` is the accumulated ``(k, v)`` prefix for chunked prefill; the chunk's own K/V is
        appended before attention and the returned K/V is still only the new part, so a caller
        accumulating across chunks concatenates rather than de-duplicates.
        """
        b, s, _ = x.shape
        q, k, v = self.project(x)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        k_full, v_full = (k, v) if past_kv is None else (torch.cat([past_kv[0], k], 2), torch.cat([past_kv[1], v], 2))
        attn = causal_sdpa(q, k_full, v_full, self.scale)
        out = self.o_proj(attn.transpose(1, 2).reshape(b, s, -1))
        return out, k, v


class DecoderLayer(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, past_kv=None):
        attn_out, k, v = self.self_attn(self.input_layernorm(x), cos, sin, past_kv)
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, k, v


class LlamaReference(nn.Module):
    """Whole-model prefill reference: embedding -> N decoder layers -> final norm -> lm_head."""

    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, dtype=REF_DTYPE)
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False, dtype=REF_DTYPE)

    @torch.no_grad()
    def forward(
        self,
        token_ids: torch.Tensor,
        *,
        start_pos: int = 0,
        past_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        skip_lm_head: bool = False,
    ):
        """``token_ids`` ``[b, s]`` (int) -> ``(logits_or_hidden, kv)``.

        ``kv`` is one ``(k_post_rope, v_raw)`` per layer covering **this** chunk. ``past_kv`` carries
        the accumulated prefix for chunk N > 0 and ``start_pos`` shifts the RoPE positions with it,
        so a chunked host run and a one-shot host run are the same arithmetic.
        """
        s = token_ids.shape[-1]
        cos, sin = rope_cos_sin(self.cfg, s, start_pos=start_pos)
        x = self.embed_tokens(token_ids)
        kv = []
        for i, layer in enumerate(self.layers):
            x, k, v = layer(x, cos, sin, None if past_kv is None else past_kv[i])
            kv.append((k, v))
        if skip_lm_head:
            return x, kv
        return self.lm_head(self.norm(x)), kv

    @torch.no_grad()
    def forward_chunked(self, token_ids: torch.Tensor, chunk_size: int):
        """Prefill in ``chunk_size`` chunks, accumulating KV — the host mirror of the device's
        multi-chunk path. Returns ``(logits, kv)`` with each layer's K/V spanning the whole sequence."""
        s = token_ids.shape[-1]
        assert s % chunk_size == 0, f"seq {s} must be a multiple of chunk {chunk_size}"
        acc: List[Tuple[torch.Tensor, torch.Tensor]] = []
        logits = None
        for start in range(0, s, chunk_size):
            past = acc if start else None
            logits, kv = self.forward(token_ids[:, start : start + chunk_size], start_pos=start, past_kv=past)
            acc = [(k, v) if not acc else (torch.cat([acc[i][0], k], 2), torch.cat([acc[i][1], v], 2))
                   for i, (k, v) in enumerate(kv)]
        return logits, acc


def load_reference_from_state_dict(cfg: LlamaConfig, state_dict: dict) -> LlamaReference:
    """Build the reference and load an HF-key state dict (``model.*`` / ``lm_head.*``) into it.

    Keys are the checkpoint's own, minus the ``model.`` prefix on the backbone; everything is cast to
    fp16 (recipe §4). Loading is strict so a renamed or missing tensor fails here rather than showing
    up as a mystery PCC drop twelve tests later.
    """
    remapped = {}
    for k, v in state_dict.items():
        kk = k[len("model.") :] if k.startswith("model.") else k
        remapped[kk] = v.to(REF_DTYPE)
    model = LlamaReference(cfg)
    model.load_state_dict(remapped, strict=True)
    model.eval()
    return model
