# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.experimental.audiox.reference.rotary import apply_rotary_pos_emb


class LayerNorm(nn.Module):
    """Bias-less LayerNorm matching audiox/models/transformer.py."""

    def __init__(self, dim: int, bias: bool = False):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        if bias:
            self.beta = nn.Parameter(torch.zeros(dim))
        else:
            self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, x.shape[-1:], weight=self.gamma, bias=self.beta)


class GLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, activation):
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x, gate = x.chunk(2, dim=-1)
        return x * self.act(gate)


class FeedForward(nn.Module):
    """SwiGLU feedforward: GLU(SiLU) -> Linear, matching the AudioX default
    (glu=True, use_conv=False, no_bias=False)."""

    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        inner_dim = int(dim * mult)
        self.ff = nn.Sequential(
            GLU(dim, inner_dim, nn.SiLU()),
            nn.Identity(),
            nn.Linear(inner_dim, dim),
            nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x)


class Attention(nn.Module):
    """Self-attention (fused QKV) or cross-attention (separate Q + KV) used by
    the AudioX continuous transformer. Skips paths AudioX never exercises
    (qk_norm, natten, masking, causal)."""

    def __init__(self, dim: int, dim_heads: int = 64, dim_context: Optional[int] = None):
        super().__init__()
        self.dim_heads = dim_heads
        self.num_heads = dim // dim_heads
        dim_kv = dim_context if dim_context is not None else dim
        self.kv_heads = dim_kv // dim_heads

        if dim_context is not None:
            self.to_q = nn.Linear(dim, dim, bias=False)
            self.to_kv = nn.Linear(dim_kv, dim_kv * 2, bias=False)
        else:
            self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.to_out = nn.Linear(dim, dim, bias=False)

    def forward(self, x, context=None, rotary_pos_emb=None):
        h, kv_h = self.num_heads, self.kv_heads
        has_context = context is not None
        kv_input = context if has_context else x

        if hasattr(self, "to_q"):
            q = self.to_q(x)
            q = rearrange(q, "b n (h d) -> b h n d", h=h)
            k, v = self.to_kv(kv_input).chunk(2, dim=-1)
            k = rearrange(k, "b n (h d) -> b h n d", h=kv_h)
            v = rearrange(v, "b n (h d) -> b h n d", h=kv_h)
        else:
            q, k, v = self.to_qkv(x).chunk(3, dim=-1)
            q = rearrange(q, "b n (h d) -> b h n d", h=h)
            k = rearrange(k, "b n (h d) -> b h n d", h=h)
            v = rearrange(v, "b n (h d) -> b h n d", h=h)

        if rotary_pos_emb is not None and not has_context:
            freqs, _ = rotary_pos_emb
            q_dtype, k_dtype = q.dtype, k.dtype
            q = apply_rotary_pos_emb(q.to(torch.float32), freqs.to(torch.float32))
            k = apply_rotary_pos_emb(k.to(torch.float32), freqs.to(torch.float32))
            q, k = q.to(q_dtype), k.to(k_dtype)

        if h != kv_h:
            repeats = h // kv_h
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class TransformerBlock(nn.Module):
    """Single AudioX continuous-transformer block on the prepend-conditioning
    path (no adaLN, no conformer, no qk_norm) — the configuration AudioX DiT
    actually uses."""

    def __init__(
        self,
        dim: int,
        dim_heads: int = 64,
        cross_attend: bool = False,
        dim_context: Optional[int] = None,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.cross_attend = cross_attend
        self.pre_norm = LayerNorm(dim)
        self.self_attn = Attention(dim, dim_heads=dim_heads)

        if cross_attend:
            self.cross_attend_norm = LayerNorm(dim)
            self.cross_attn = Attention(dim, dim_heads=dim_heads, dim_context=dim_context)

        self.ff_norm = LayerNorm(dim)
        self.ff = FeedForward(dim, mult=ff_mult)

    def forward(self, x, context=None, rotary_pos_emb=None):
        x = x + self.self_attn(self.pre_norm(x), rotary_pos_emb=rotary_pos_emb)
        if context is not None and self.cross_attend:
            x = x + self.cross_attn(self.cross_attend_norm(x), context=context)
        x = x + self.ff(self.ff_norm(x))
        return x
