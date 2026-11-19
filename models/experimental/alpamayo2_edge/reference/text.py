# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright 2026 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pure-torch reference of the Cosmos3 Edge text model, used for both the LM and the action expert.

Ported from the Cosmos3 Edge modeling code so that it carries no transformers dependency; it is the golden
the TT LM and expert are compared against. The expert is the same text model without `embed_tokens`; it
attends non-causally over the LM's KV cache, which both models share.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.experimental.alpamayo2_edge.reference.config import TextConfig


def mrope_cos_sin(
    position_ids: torch.Tensor,
    head_dim: int = 128,
    theta: float = 1e8,
    section: tuple = (24, 20, 20),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interleaved M-RoPE tables: [3, B, S] T/H/W positions -> fp32 (cos, sin), each [B, S, head_dim].

    Frequency f takes its position from H when f % 3 == 1 and f < 3 * section[1], from W when f % 3 == 2 and
    f < 3 * section[2], and from T otherwise, so T owns the tail beyond the interleaved block.
    """
    if sum(section) != head_dim // 2:
        raise ValueError(f"mrope section {section} does not cover head_dim {head_dim}")
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float, device=position_ids.device) / head_dim))
    freqs = position_ids.expand(3, -1, -1)[..., None].float() * inv_freq

    def recompose(table: torch.Tensor) -> torch.Tensor:
        out = table[0].clone()
        for axis, offset in ((1, 1), (2, 2)):
            idx = slice(offset, section[axis] * 3, 3)
            out[..., idx] = table[axis, ..., idx]
        return torch.cat((out, out), dim=-1)

    return recompose(freqs.cos()), recompose(freqs.sin())


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, heads, S, D]; cos, sin: [B, S, D]."""
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return x * cos + rotate_half(x) * sin


class KVCache:
    """Per-layer rotated K and raw V, each [B, kv_heads, T, head_dim], shared by the LM and the expert."""

    def __init__(self, num_layers: int):
        self.k: list[torch.Tensor | None] = [None] * num_layers
        self.v: list[torch.Tensor | None] = [None] * num_layers

    @property
    def seq_len(self) -> int:
        return 0 if self.k[0] is None else self.k[0].shape[2]

    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Append S rows to `layer` and return the layer's full K and V."""
        if self.k[layer] is None:
            self.k[layer], self.v[layer] = k, v
        else:
            self.k[layer] = torch.cat((self.k[layer], k), dim=2)
            self.v[layer] = torch.cat((self.v[layer], v), dim=2)
        return self.k[layer], self.v[layer]

    def crop(self, n: int) -> None:
        """Keep the first `n` positions of every layer."""
        if n < 0:
            raise ValueError(f"crop length must be >= 0, got {n}")
        for layer, (k, v) in enumerate(zip(self.k, self.v)):
            if k is not None:
                self.k[layer], self.v[layer] = k[:, :, :n], v[:, :, :n]

    def repeat_interleave(self, b: int) -> None:
        """Repeat every batch row `b` times in place, as for multiple samples per prompt."""
        for layer, (k, v) in enumerate(zip(self.k, self.v)):
            if k is not None:
                self.k[layer] = k.repeat_interleave(b, dim=0)
                self.v[layer] = v.repeat_interleave(b, dim=0)


def _linear(weights: dict, name: str, dtype: torch.dtype) -> nn.Linear:
    w = weights[f"{name}.weight"]
    layer = nn.Linear(w.shape[1], w.shape[0], bias=False)
    layer.weight.data = w.to(dtype)
    return layer


class RMSNorm(nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(weight.to(dtype), requires_grad=False)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(input_dtype)


class Attention(nn.Module):
    """GQA attention; q/k/v width is heads * head_dim, independent of the model width (expert: 2048 vs 512)."""

    def __init__(self, tcfg: TextConfig, weights: dict, prefix: str, layer: int, dtype: torch.dtype):
        super().__init__()
        self.layer = layer
        self.head_dim = tcfg.head_dim
        self.groups = tcfg.heads // tcfg.kv_heads
        self.scale = tcfg.head_dim**-0.5
        self.q_proj = _linear(weights, f"{prefix}q_proj", dtype)
        self.k_proj = _linear(weights, f"{prefix}k_proj", dtype)
        self.v_proj = _linear(weights, f"{prefix}v_proj", dtype)
        self.o_proj = _linear(weights, f"{prefix}o_proj", dtype)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        bias: torch.Tensor | None,
        cache: KVCache | None,
    ) -> torch.Tensor:
        bsz, seq, _ = x.shape
        shape = (bsz, seq, -1, self.head_dim)
        q = apply_rotary_pos_emb(self.q_proj(x).view(shape).transpose(1, 2), cos, sin)
        k = apply_rotary_pos_emb(self.k_proj(x).view(shape).transpose(1, 2), cos, sin)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        if cache is not None:
            k, v = cache.append(self.layer, k, v)

        k = k.repeat_interleave(self.groups, dim=1)
        v = v.repeat_interleave(self.groups, dim=1)
        scores = torch.matmul(q, k.transpose(2, 3)) * self.scale
        if bias is not None:
            scores = scores + bias
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(probs, v).transpose(1, 2).reshape(bsz, seq, -1)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, weights: dict, prefix: str, dtype: torch.dtype):
        super().__init__()
        self.fc1 = _linear(weights, f"{prefix}fc1", dtype)
        self.fc2 = _linear(weights, f"{prefix}fc2", dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.square(F.relu(self.fc1(x))))


class DecoderLayer(nn.Module):
    def __init__(self, tcfg: TextConfig, weights: dict, layer: int, dtype: torch.dtype):
        super().__init__()
        prefix = f"layers.{layer}."
        self.input_layernorm = RMSNorm(weights[f"{prefix}input_layernorm.weight"], tcfg.eps, dtype)
        self.post_attention_layernorm = RMSNorm(weights[f"{prefix}post_attention_layernorm.weight"], tcfg.eps, dtype)
        self.self_attn = Attention(tcfg, weights, f"{prefix}self_attn.", layer, dtype)
        self.mlp = MLP(weights, f"{prefix}mlp.", dtype)

    def forward(self, x, cos, sin, bias, cache):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, bias, cache)
        return x + self.mlp(self.post_attention_layernorm(x))


def attention_bias(
    seq: int,
    cached: int,
    attn_mask: torch.Tensor | None,
    causal: bool,
    dtype: torch.dtype,
    device: torch.device | None = None,
) -> torch.Tensor | None:
    """Additive bias over [S, cached + S] keys: the causal mask (offset by the cached prefix) plus `attn_mask`.

    The sum is clamped at the dtype minimum so a query with every key masked gets a uniform row, not NaN.
    """
    if not causal and attn_mask is None:
        return None
    min_value = torch.finfo(dtype).min
    bias = torch.zeros(1, 1, seq, cached + seq, dtype=dtype, device=device)
    if causal:
        keys = torch.arange(cached + seq, device=device)
        future = keys[None, :] > torch.arange(seq, device=device)[:, None] + cached
        bias = bias.masked_fill(future, min_value)
    if attn_mask is not None:
        bias = (bias + attn_mask.to(dtype)).clamp(min=min_value)
    return bias


class TextModel(nn.Module):
    """Cosmos3 Edge decoder stack; `has_embedding=False` builds the expert, which has no `embed_tokens`."""

    def __init__(self, tcfg: TextConfig, weights: dict, has_embedding: bool, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.tcfg = tcfg
        self.dtype = dtype
        self.embed_tokens = None
        if has_embedding:
            self.embed_tokens = nn.Embedding.from_pretrained(weights["embed_tokens.weight"].to(dtype))
        self.layers = nn.ModuleList(DecoderLayer(tcfg, weights, i, dtype) for i in range(tcfg.layers))
        self.norm = RMSNorm(weights["norm.weight"], tcfg.eps, dtype)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        cache: KVCache | None = None,
        attn_mask: torch.Tensor | None = None,
        causal: bool = True,
    ) -> torch.Tensor:
        """inputs_embeds [B, S, H], position_ids [3, B, S] -> [B, S, H] after the final norm.

        `attn_mask` is an additive mask broadcastable to [B, 1, S, cached + S]. With `causal=False` it is the
        only mask applied (expert).
        """
        x = inputs_embeds.to(self.dtype)
        seq = x.shape[1]
        cached = 0 if cache is None else cache.seq_len
        cos, sin = mrope_cos_sin(position_ids, self.tcfg.head_dim, self.tcfg.rope_theta, self.tcfg.mrope_section)
        cos, sin = cos.to(self.dtype), sin.to(self.dtype)
        bias = attention_bias(seq, cached, attn_mask, causal, self.dtype, x.device)
        for layer in self.layers:
            x = layer(x, cos, sin, bias, cache)
        return self.norm(x)


class LanguageModel(nn.Module):
    """Text model with embeddings plus the untied LM head."""

    def __init__(self, tcfg: TextConfig, weights: dict, lm_head_weights: dict, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.text = TextModel(tcfg, weights, has_embedding=True, dtype=dtype)
        w = lm_head_weights["weight"]
        self.lm_head = nn.Linear(w.shape[1], w.shape[0], bias=False)
        self.lm_head.weight.data = w.to(dtype)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden)
