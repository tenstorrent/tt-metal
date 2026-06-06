# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Self-contained float32 torch reference for the dots.ocr text-decoder prefill.

Matches Qwen2/dots.ocr semantics (half-half RoPE, GQA, SwiGLU, RMSNorm) so the
TTNN TP4 modules can be verified op-for-op without loading the full HF model.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.experimental.dots_ocr_tp4.tt.common import DotsOCRConfig


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rope_cos_sin(seq_len, head_dim, theta):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


class TorchRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        dt = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.variance_epsilon)
        return (self.weight * x.to(dt)).to(dt)


class TorchGQAAttention(nn.Module):
    def __init__(self, config: DotsOCRConfig):
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.hidden_size, config.q_size, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.kv_size, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.kv_size, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.q_size, config.hidden_size, bias=False)

    def forward(self, x):
        B, S, _ = x.shape
        cfg = self.config
        h, kvh, d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        q = self.q_proj(x).view(B, S, h, d).transpose(1, 2)
        k = self.k_proj(x).view(B, S, kvh, d).transpose(1, 2)
        v = self.v_proj(x).view(B, S, kvh, d).transpose(1, 2)
        cos, sin = rope_cos_sin(S, d, cfg.rope_theta)
        cos, sin = cos[None, None], sin[None, None]
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        rep = h // kvh
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d)
        attn = attn + torch.triu(torch.full((S, S), float("-inf")), diagonal=1)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, S, h * d)
        return self.o_proj(out)


class TorchSwiGLUMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class TorchDecoderBlock(nn.Module):
    def __init__(self, config: DotsOCRConfig):
        super().__init__()
        self.input_layernorm = TorchRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = TorchGQAAttention(config)
        self.post_attention_layernorm = TorchRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = TorchSwiGLUMLP(config.hidden_size, config.intermediate_size)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class TorchLMHead(nn.Module):
    """Final RMSNorm + LM head (H -> vocab), no bias (tie_word_embeddings=False)."""

    def __init__(self, config: DotsOCRConfig, vocab_size=151936):
        super().__init__()
        self.norm = TorchRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, vocab_size, bias=False)

    def forward(self, hidden, last_token_only=True):
        if last_token_only:
            hidden = hidden[:, -1:, :]
        return self.lm_head(self.norm(hidden))


class TorchDecoderStack(nn.Module):
    def __init__(self, config: DotsOCRConfig, num_layers=None):
        super().__init__()
        n = num_layers if num_layers is not None else config.num_hidden_layers
        self.layers = nn.ModuleList([TorchDecoderBlock(config) for _ in range(n)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
