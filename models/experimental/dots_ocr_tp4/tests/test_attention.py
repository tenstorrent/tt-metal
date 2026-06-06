# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TP4 GQA attention correctness vs a torch reference (dots.ocr text-decoder dims)."""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from models.experimental.dots_ocr_tp4.tt.attention import DotsOCRAttentionTP4
from models.experimental.dots_ocr_tp4.tt.common import DotsOCRConfig, from_replicated_to_torch, to_replicated
from models.experimental.dots_ocr_tp4.tests.common import device_params, resolve_mesh_shape


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _rope_cos_sin(seq_len, head_dim, theta):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv_freq)  # [S, head_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)  # [S, head_dim]
    return emb.cos(), emb.sin()  # [S, head_dim]


class TorchGQAAttention(nn.Module):
    """Reference Qwen2/dots.ocr GQA attention (half-half RoPE, causal)."""

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

        q = self.q_proj(x).view(B, S, h, d).transpose(1, 2)  # [B, h, S, d]
        k = self.k_proj(x).view(B, S, kvh, d).transpose(1, 2)  # [B, kvh, S, d]
        v = self.v_proj(x).view(B, S, kvh, d).transpose(1, 2)

        cos, sin = _rope_cos_sin(S, d, cfg.rope_theta)
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        # GQA: repeat kv heads to match q heads.
        rep = h // kvh
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

        scale = 1.0 / math.sqrt(d)
        attn = torch.matmul(q, k.transpose(-1, -2)) * scale
        causal = torch.triu(torch.full((S, S), float("-inf")), diagonal=1)
        attn = attn + causal
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # [B, h, S, d]
        out = out.transpose(1, 2).reshape(B, S, h * d)
        return self.o_proj(out)


@pytest.mark.parametrize("device_params", [device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [resolve_mesh_shape()], indirect=True)
@pytest.mark.parametrize("seq_len", [2816])
def test_dots_ocr_attention_tp4(mesh_device, seq_len):
    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    config = DotsOCRConfig()
    H = config.hidden_size

    torch_attn = TorchGQAAttention(config).eval()  # float32 reference
    x = torch.randn(1, seq_len, H, dtype=torch.bfloat16)
    torch_out = torch_attn(x.to(torch.float32))

    tt_attn = DotsOCRAttentionTP4.from_torch(mesh_device, config, torch_attn)
    x_tt = to_replicated(x, mesh_device, dtype=ttnn.bfloat16)

    out_tt = tt_attn.forward(x_tt)
    ttnn.synchronize_device(mesh_device)

    out_torch = from_replicated_to_torch(out_tt, mesh_device).to(torch.float32).reshape(torch_out.shape)

    assert_with_pcc(torch_out.to(torch.float32), out_torch, pcc=0.99)
