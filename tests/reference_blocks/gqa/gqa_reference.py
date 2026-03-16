# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Pure-PyTorch reference implementation of Grouped-Query Attention.

Two independent implementations are provided so that one can serve as the
golden reference for the other:

1. ``GQAReference``  – manual matmuls, explicit KV repeat, step-by-step RoPE.
2. ``GQAReferenceSdpa`` – delegates to ``torch.nn.functional.scaled_dot_product_attention``.

Both share the same weight matrices (Q/K/V/O projections) so that, given
identical inputs and weights, their outputs can be compared.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gqa_config import GQAConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * norm).to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_dim: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors.

    When ``rope_dim`` < ``head_dim`` (partial RoPE, e.g. GLM-4), only the
    first ``rope_dim`` dimensions are rotated and the rest are left unchanged.

    Shapes
    ------
    q, k : (batch, heads, seq_len, head_dim)
    cos, sin : (1, 1, seq_len, rope_dim)   *or broadcastable*
    """
    if rope_dim is not None and rope_dim < q.size(-1):
        q_rot, q_pass = q[..., :rope_dim], q[..., rope_dim:]
        k_rot, k_pass = k[..., :rope_dim], k[..., rope_dim:]
        q_rot = q_rot * cos + _rotate_half(q_rot) * sin
        k_rot = k_rot * cos + _rotate_half(k_rot) * sin
        q = torch.cat([q_rot, q_pass], dim=-1)
        k = torch.cat([k_rot, k_pass], dim=-1)
    else:
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
    return q, k


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match the number of query heads.

    (batch, kv_heads, seq, head_dim) -> (batch, q_heads, seq, head_dim)
    """
    if n_rep == 1:
        return x
    bs, n_kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(bs, n_kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(bs, n_kv_heads * n_rep, seq_len, head_dim)


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    theta: float = 10000.0,
    rope_dim: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute cos/sin tables for RoPE.

    Returns tensors of shape ``(1, 1, seq_len, rope_dim)``.
    """
    dim = rope_dim if rope_dim is not None else head_dim
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))
    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().unsqueeze(0).unsqueeze(0)
    sin = emb.sin().unsqueeze(0).unsqueeze(0)
    return cos, sin


def softcap(logits: torch.Tensor, cap: float) -> torch.Tensor:
    """Grok-2 style soft-capping of attention logits."""
    return cap * torch.tanh(logits / cap)


# ---------------------------------------------------------------------------
# Reference implementation 1 – manual matmuls
# ---------------------------------------------------------------------------


class GQAReference(nn.Module):
    """Grouped-Query Attention with explicit Q·K^T, softmax, ·V matmuls.

    Includes optional pre-attention RMSNorm, RoPE, QK-norm, softcapping,
    causal masking, and KV-cache for incremental decoding.
    """

    def __init__(self, config: GQAConfig, use_pre_norm: bool = True):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = config.num_kv_groups
        self.scaling = config.effective_scaling
        self.rope_dim = int(config.head_dim * config.rope_partial_factor) if config.rope_partial_factor < 1.0 else None
        self.attn_logit_softcapping = config.attn_logit_softcapping

        if use_pre_norm:
            self.pre_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.pre_norm = None

        self.q_proj = nn.Linear(config.hidden_size, config.q_proj_size, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.kv_proj_size, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.kv_proj_size, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.q_proj_size, config.hidden_size, bias=config.attention_bias)

        if config.use_qk_norm:
            self.q_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_residual: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Parameters
        ----------
        hidden_states : (batch, seq_len, hidden_size)
        attention_mask : (batch, 1, seq_len, kv_len) or None for causal
        cos, sin : RoPE tensors from ``build_rope_cache``
        kv_cache : optional (k_cache, v_cache) each (batch, kv_heads, cached_len, head_dim)
        use_residual : whether to add input as residual to output

        Returns
        -------
        output : (batch, seq_len, hidden_size)
        new_kv_cache : updated (k, v) cache tensors  (None when kv_cache not provided)
        """
        residual = hidden_states
        if self.pre_norm is not None:
            hidden_states = self.pre_norm(hidden_states)

        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        if cos is not None and sin is not None:
            q, k = apply_rotary_pos_emb(q, k, cos, sin, rope_dim=self.rope_dim)

        new_kv_cache = None
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            if k_cache.size(2) > 0:
                k = torch.cat([k_cache, k], dim=2)
                v = torch.cat([v_cache, v], dim=2)
            new_kv_cache = (k.detach(), v.detach())

        k_expanded = repeat_kv(k, self.num_kv_groups)
        v_expanded = repeat_kv(v, self.num_kv_groups)

        attn_weights = torch.matmul(q, k_expanded.transpose(2, 3)) * self.scaling

        if self.attn_logit_softcapping is not None:
            attn_weights = softcap(attn_weights, self.attn_logit_softcapping)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v_expanded)

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, seq_len, self.config.q_proj_size)
        output = self.o_proj(attn_output)

        if use_residual:
            output = residual + output

        return output, new_kv_cache


# ---------------------------------------------------------------------------
# Reference implementation 2 – using F.scaled_dot_product_attention
# ---------------------------------------------------------------------------


class GQAReferenceSdpa(nn.Module):
    """GQA using ``torch.nn.functional.scaled_dot_product_attention``.

    This serves as an independent second reference.  Shares the same
    projection structure so that weights can be copied from ``GQAReference``.
    """

    def __init__(self, config: GQAConfig, use_pre_norm: bool = True):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = config.num_kv_groups
        self.scaling = config.effective_scaling
        self.rope_dim = int(config.head_dim * config.rope_partial_factor) if config.rope_partial_factor < 1.0 else None

        if use_pre_norm:
            self.pre_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.pre_norm = None

        self.q_proj = nn.Linear(config.hidden_size, config.q_proj_size, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.kv_proj_size, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.kv_proj_size, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.q_proj_size, config.hidden_size, bias=config.attention_bias)

        if config.use_qk_norm:
            self.q_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_residual: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        if self.pre_norm is not None:
            hidden_states = self.pre_norm(hidden_states)

        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        if cos is not None and sin is not None:
            q, k = apply_rotary_pos_emb(q, k, cos, sin, rope_dim=self.rope_dim)

        new_kv_cache = None
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            if k_cache.size(2) > 0:
                k = torch.cat([k_cache, k], dim=2)
                v = torch.cat([v_cache, v], dim=2)
            new_kv_cache = (k.detach(), v.detach())

        k_expanded = repeat_kv(k, self.num_kv_groups)
        v_expanded = repeat_kv(v, self.num_kv_groups)

        attn_output = F.scaled_dot_product_attention(
            q,
            k_expanded,
            v_expanded,
            attn_mask=attention_mask,
            scale=self.scaling,
            is_causal=(attention_mask is None and kv_cache is None),
        )

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, seq_len, self.config.q_proj_size)
        output = self.o_proj(attn_output)

        if use_residual:
            output = residual + output

        return output, new_kv_cache


# ---------------------------------------------------------------------------
# Weight sharing utility
# ---------------------------------------------------------------------------


def copy_weights(src: nn.Module, dst: nn.Module) -> None:
    """Copy all matching named parameters from *src* to *dst* (in-place)."""
    src_dict = dict(src.named_parameters())
    for name, param in dst.named_parameters():
        if name in src_dict:
            param.data.copy_(src_dict[name].data)
