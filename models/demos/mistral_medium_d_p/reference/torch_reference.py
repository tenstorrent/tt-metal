# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Self-contained torch reference for the Mistral-Medium-3.5 decoder — PURE TORCH, no ttnn.

Written in the **HF convention** (``rotate_half`` RoPE, ``[out, in]`` linear weights) and pinned to
``transformers.models.ministral3`` by ``tests/unit/test_reference_model.py``, which runs on a
plain dev box. The device tests then PCC the TT modules against *this*, so a TT failure is never
ambiguous about which side is wrong.

Deliberately small: the whole point is that Mistral-Medium's decoder is a plain dense
GQA + SwiGLU block, and this file is the evidence.
"""

from __future__ import annotations

import torch


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """MistralRMSNorm: normalize in fp32, scale by ``weight`` (no Gemma ``(1 + w)`` fold)."""
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (weight.float() * x).to(dtype)


def swiglu_mlp(x: torch.Tensor, gate_w: torch.Tensor, up_w: torch.Tensor, down_w: torch.Tensor) -> torch.Tensor:
    """MistralMLP: ``down(silu(gate(x)) * up(x))``, all bias-free."""
    return torch.nn.functional.silu(x @ gate_w.t()) * (x @ up_w.t()) @ down_w.t()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope_hf(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """cos/sin: ``[S, head_dim]`` (HF ``cat([half, half], -1)`` layout); t: ``[B, H, S, head_dim]``."""
    return t * cos + rotate_half(t) * sin


def gqa_attention(
    x: torch.Tensor,
    w: dict,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    n_q: int,
    n_kv: int,
    head_dim: int,
) -> torch.Tensor:
    """Dense causal GQA with full rotary. No bias, no QK-norm, no sinks, no sliding window.

    ``w`` holds HF-layout ``[out, in]`` matrices under keys ``q``/``k``/``v``/``o``.
    """
    B, S, _ = x.shape
    q = (x @ w["q"].t()).view(B, S, n_q, head_dim).transpose(1, 2)
    k = (x @ w["k"].t()).view(B, S, n_kv, head_dim).transpose(1, 2)
    v = (x @ w["v"].t()).view(B, S, n_kv, head_dim).transpose(1, 2)

    q = apply_rope_hf(q, cos, sin)
    k = apply_rope_hf(k, cos, sin)

    rep = n_q // n_kv
    k = k.repeat_interleave(rep, dim=1)
    v = v.repeat_interleave(rep, dim=1)

    scores = (q @ k.transpose(-1, -2)) * (head_dim**-0.5)
    scores = scores + torch.triu(torch.full((S, S), float("-inf"), dtype=scores.dtype), diagonal=1)
    probs = torch.softmax(scores, dim=-1)

    out = (probs @ v).transpose(1, 2).reshape(B, S, n_q * head_dim)
    return out @ w["o"].t()


def decoder_layer(
    x: torch.Tensor,
    w: dict,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    n_q: int,
    n_kv: int,
    head_dim: int,
    eps: float,
) -> torch.Tensor:
    """input_layernorm -> attn -> +residual -> post_attention_layernorm -> mlp -> +residual.

    ``w`` additionally holds ``input_layernorm`` / ``post_attention_layernorm`` gains and the
    ``gate``/``up``/``down`` MLP matrices.
    """
    h = x + gqa_attention(rms_norm(x, w["input_layernorm"], eps), w, cos, sin, n_q=n_q, n_kv=n_kv, head_dim=head_dim)
    return h + swiglu_mlp(rms_norm(h, w["post_attention_layernorm"], eps), w["gate"], w["up"], w["down"])


def random_layer_weights(hidden, n_q, n_kv, head_dim, ffn, *, seed=0, scale=0.02):
    """Random HF-layout weights for one decoder layer, shared by the reference and the TT modules."""
    g = torch.Generator().manual_seed(seed)

    def r(*shape):
        return torch.randn(*shape, generator=g) * scale

    return {
        "q": r(n_q * head_dim, hidden),
        "k": r(n_kv * head_dim, hidden),
        "v": r(n_kv * head_dim, hidden),
        "o": r(hidden, n_q * head_dim),
        "gate": r(ffn, hidden),
        "up": r(ffn, hidden),
        "down": r(hidden, ffn),
        # Norm gains sit near 1.0, as trained ones do.
        "input_layernorm": 1.0 + r(hidden),
        "post_attention_layernorm": 1.0 + r(hidden),
    }


def to_hf_state_dict(w: dict) -> dict:
    """``random_layer_weights`` -> an HF-named layer state dict (what the TT modules consume)."""
    return {
        "self_attn.q_proj.weight": w["q"],
        "self_attn.k_proj.weight": w["k"],
        "self_attn.v_proj.weight": w["v"],
        "self_attn.o_proj.weight": w["o"],
        "mlp.gate_proj.weight": w["gate"],
        "mlp.up_proj.weight": w["up"],
        "mlp.down_proj.weight": w["down"],
        "input_layernorm.weight": w["input_layernorm"],
        "post_attention_layernorm.weight": w["post_attention_layernorm"],
    }
