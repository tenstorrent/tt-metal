# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Standalone torch reference for Llama 3.1 8B (D1 oracle).

**Reference purity**: torch only. No ttnn, no device code, no ``transformers`` import — so this
module can be the oracle every PCC test measures against, and so a `transformers` version bump
cannot silently move the target. ``tests/torch/test_llama_reference.py`` pins it against upstream
``transformers.models.llama.modeling_llama`` so the two cannot drift.

Provenance (transformers 5.x ``modeling_llama.py``): ``LlamaRMSNorm``, ``LlamaMLP``,
``repeat_kv``, ``apply_rotary_pos_emb`` (rotate_half / "half_split" convention),
``LlamaAttention``, ``LlamaDecoderLayer``, ``LlamaModel``, and
``modeling_rope_utils._compute_llama3_parameters``.

Conventions that matter downstream:

* RoPE here is the **HF half-split** convention (``rotate_half``: ``[-x2, x1]`` over the two
  halves of the head). The on-device indexed RoPE uses the **Meta interleaved** convention with
  ``reverse_permute``-swizzled q/k projections; the two are equivalent up to that weight
  permutation, which ``tt/model_config.py`` applies at load time. Compare on-device K against this
  reference only after undoing the permutation (see ``meta_to_hf_head_perm``).
* Attention accumulates in fp32 regardless of the input dtype: this is an oracle, not a
  performance path.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import LlamaConfig

# ---------------------------------------------------------------------------------------------
# RoPE — llama3 smooth-ramp scaling
# ---------------------------------------------------------------------------------------------


def llama3_inv_freq(head_dim: int, config: LlamaConfig) -> tuple[torch.Tensor, float]:
    """Inverse frequencies with llama3 scaling + the attention factor.

    Mirrors ``transformers.modeling_rope_utils._compute_llama3_parameters``. Three wavelength
    regimes, blended by a smooth ramp:

      * wavelength shorter than ``orig_ctx / high_freq_factor``  -> pure extrapolation (unscaled)
      * wavelength longer  than ``orig_ctx / low_freq_factor``   -> pure interpolation (/factor)
      * in between                                               -> linear ramp between the two

    NOTE this is NOT YaRN. The donor (``gpt_oss_d_p/tt/rope.py``) implements YaRN, which is
    parameterised by beta_fast/beta_slow and carries an mscale; llama3 scaling has neither and its
    ``attention_factor`` is exactly 1.0. Substituting one for the other produces plausible-looking
    cos/sin and a long-context PCC collapse.
    """
    scaling = config.rope_scaling or {}
    factor = scaling.get("factor", 8.0)
    low_freq_factor = scaling.get("low_freq_factor", 1.0)
    high_freq_factor = scaling.get("high_freq_factor", 4.0)
    orig_ctx = scaling.get("original_max_position_embeddings", 8192)

    pos_freqs = config.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    inv_freq = 1.0 / pos_freqs

    low_freq_wavelen = orig_ctx / low_freq_factor
    high_freq_wavelen = orig_ctx / high_freq_factor
    wavelen = 2 * math.pi / inv_freq

    # Long wavelengths (low frequency): interpolate.
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    # Medium band: smooth ramp between interpolation and extrapolation.
    smooth = (orig_ctx / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed = (1 - smooth) * inv_freq_llama / factor + smooth * inv_freq_llama
    is_medium = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    inv_freq_llama = torch.where(is_medium, smoothed, inv_freq_llama)

    attention_factor = 1.0  # llama3 scaling applies no mscale
    return inv_freq_llama, attention_factor


def build_cos_sin_hf(seq_len: int, config: LlamaConfig, *, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """HF half-split cos/sin ``[1, seq_len, head_dim]`` — ``cat([f, f], dim=-1)``.

    This is the layout ``apply_rotary_pos_emb`` (rotate_half) consumes.
    """
    inv_freq, attn_factor = llama3_inv_freq(config.head_dim, config)
    pos = torch.arange(offset, offset + seq_len, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)  # [seq_len, head_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, head_dim]
    return (emb.cos() * attn_factor)[None], (emb.sin() * attn_factor)[None]


def build_cos_sin_meta(seq_len: int, config: LlamaConfig, *, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Meta interleaved cos/sin ``[1, 1, seq_len, head_dim]`` — ``[c0, c0, c1, c1, ...]``.

    The convention ``ttnn.rotary_embedding_llama`` / ``rotary_embedding_indexed`` expect, paired
    with ``reverse_permute``-swizzled q/k projection weights. Same underlying frequencies as
    :func:`build_cos_sin_hf`; only the interleave differs.
    """
    inv_freq, attn_factor = llama3_inv_freq(config.head_dim, config)
    pos = torch.arange(offset, offset + seq_len, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)  # [seq_len, head_dim/2]
    cos_h, sin_h = freqs.cos() * attn_factor, freqs.sin() * attn_factor
    cos = torch.stack([cos_h, cos_h], dim=-1).flatten(-2)[None, None]
    sin = torch.stack([sin_h, sin_h], dim=-1).flatten(-2)[None, None]
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim: int = 1):
    """HF half-split RoPE. q/k are ``[B, n_heads, S, head_dim]``; cos/sin ``[B, S, head_dim]``."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def meta_to_hf_head_perm(x: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Undo the Meta interleaved head swizzle on the last dim: ``[a0,b0,a1,b1,...] -> [a..., b...]``.

    ``convert_hf_qkv_to_meta_format`` reverse-permutes the q/k projection ROWS so the on-device
    interleaved RoPE reproduces HF's half-split rotation. Consequently on-device Q/K head vectors
    are the interleave of the two HF halves. Apply this to on-device K before PCC'ing it against
    the reference K, or the comparison measures the permutation and not the math.
    """
    assert x.shape[-1] == head_dim, f"last dim {x.shape[-1]} != head_dim {head_dim}"
    pair = x.reshape(*x.shape[:-1], head_dim // 2, 2)
    return torch.cat((pair[..., 0], pair[..., 1]), dim=-1)


# ---------------------------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------------------------


class LlamaRMSNorm(nn.Module):
    """Plain RMSNorm: ``x / rms(x) * weight``. No Gemma ``(1 + w)`` fold."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaMLP(nn.Module):
    """Dense gated SwiGLU: ``down(silu(gate(x)) * up(x))``. Three matrices, no bias."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand ``[B, n_kv, S, D]`` to ``[B, n_kv*n_rep, S, D]`` (GQA head broadcast)."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def causal_sdpa(q, k, v, *, scale: float, kv_offset: int = 0) -> torch.Tensor:
    """Explicit causal SDPA in fp32. ``kv_offset`` is the number of cached tokens preceding q.

    Query row ``i`` (global position ``kv_offset + i``) attends keys ``0 .. kv_offset + i``, which is
    what a chunked prefill step needs; ``kv_offset=0`` is the ordinary one-shot causal case.
    """
    q32, k32, v32 = q.float(), k.float(), v.float()
    attn = torch.matmul(q32, k32.transpose(-1, -2)) * scale  # [B, H, Sq, Sk]
    sq, sk = attn.shape[-2], attn.shape[-1]
    q_pos = torch.arange(sq, device=attn.device) + kv_offset
    k_pos = torch.arange(sk, device=attn.device)
    attn = attn.masked_fill(k_pos[None, :] > q_pos[:, None], float("-inf"))
    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, v32).to(q.dtype)


class LlamaAttention(nn.Module):
    """GQA attention: QKV proj -> head split -> full RoPE -> causal SDPA -> o_proj. No bias."""

    def __init__(self, config: LlamaConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = config.num_kv_groups
        self.scaling = self.head_dim**-0.5
        b = config.attention_bias
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=b)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=b)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=b)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=b)

    def project_qkv(self, hidden_states: torch.Tensor):
        """-> q ``[B, n_q, S, D]``, k/v ``[B, n_kv, S, D]`` (pre-RoPE)."""
        b, s, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    def forward(self, hidden_states, position_embeddings, *, past_kv=None, kv_offset=0):
        """One prefill step.

        Args:
            hidden_states: ``[B, S, hidden]``
            position_embeddings: ``(cos, sin)`` in HF half-split layout for THIS chunk's positions
            past_kv: optional ``(k, v)`` prefix already computed, ``[B, n_kv, kv_offset, D]``
            kv_offset: global position of this chunk's first token

        Returns:
            ``(attn_out [B, S, hidden], k_post_rope, v)`` — k/v are THIS chunk's, pre-concat, so a
            caller building a golden trace records exactly what the device writes to the cache.
        """
        cos, sin = position_embeddings
        q, k, v = self.project_qkv(hidden_states)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k_full, v_full = k, v
        if past_kv is not None:
            k_full = torch.cat([past_kv[0], k], dim=2)
            v_full = torch.cat([past_kv[1], v], dim=2)

        attn = causal_sdpa(
            q,
            repeat_kv(k_full, self.num_kv_groups),
            repeat_kv(v_full, self.num_kv_groups),
            scale=self.scaling,
            kv_offset=kv_offset,
        )
        b, _, s, _ = attn.shape
        attn = attn.transpose(1, 2).reshape(b, s, self.num_heads * self.head_dim)
        return self.o_proj(attn), k, v


class LlamaDecoderLayer(nn.Module):
    """Pre-norm decoder layer: ``x + attn(norm(x))``, then ``x + mlp(norm(x))``."""

    def __init__(self, config: LlamaConfig, layer_idx: int = 0):
        super().__init__()
        self.self_attn = LlamaAttention(config, layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, *, past_kv=None, kv_offset=0):
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        h, k, v = self.self_attn(h, position_embeddings, past_kv=past_kv, kv_offset=kv_offset)
        hidden_states = residual + h

        residual = hidden_states
        h = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(h)
        return hidden_states, k, v


class LlamaModel(nn.Module):
    """Embedding -> N decoder layers -> final norm -> lm_head."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, *, past_kvs=None, kv_offset=0, return_hidden_states=False):
        """Prefill forward.

        Returns ``(logits, kvs, hidden_states_per_layer)`` where ``kvs`` is a list of ``(k, v)``
        per layer for THIS chunk (post-RoPE K, raw V) — the golden trace the device KV cache is
        PCC'd against.
        """
        h = self.embed_tokens(input_ids)
        seq_len = h.shape[1]
        cos, sin = build_cos_sin_hf(seq_len, self.config, offset=kv_offset)
        cos, sin = cos.to(h.dtype), sin.to(h.dtype)

        kvs, snapshots = [], []
        for i, layer in enumerate(self.layers):
            past = past_kvs[i] if past_kvs is not None else None
            h, k, v = layer(h, (cos, sin), past_kv=past, kv_offset=kv_offset)
            kvs.append((k, v))
            if return_hidden_states:
                snapshots.append(h)
        h = self.norm(h)
        return self.lm_head(h), kvs, snapshots


def hf_to_meta_head_perm(x: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Apply the Meta interleaved head swizzle on the last dim: ``[a..., b...] -> [a0,b0,a1,b1,...]``.

    Inverse of :func:`meta_to_hf_head_perm`. Use it to bring an HF-convention Q/K into the layout the
    device holds — e.g. to compare a reference K against the on-device KV cache, whose contents are
    post-RoPE K computed from ``reverse_permute``d projection weights.
    """
    assert x.shape[-1] == head_dim, f"last dim {x.shape[-1]} != head_dim {head_dim}"
    a, b = x[..., : head_dim // 2], x[..., head_dim // 2 :]
    return torch.stack([a, b], dim=-1).flatten(-2)
