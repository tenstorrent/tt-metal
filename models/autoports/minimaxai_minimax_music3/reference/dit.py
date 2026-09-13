"""MiniMaxMusic3Transformer1DModel (flow-matching DiT) as a plain torch module. Keys match transformer/*.safetensors."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.autoports.minimaxai_minimax_music3.config import DiTConfig


class FourierEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(embedding_dim // 2, 1))

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        angles = 2.0 * math.pi * timestep.unsqueeze(-1) @ self.weight.T
        return torch.cat((angles.cos(), angles.sin()), dim=-1)


class TimestepEmbedding(nn.Module):
    """diffusers TimestepEmbedding(in, dim): linear_1 -> SiLU -> linear_2."""

    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample):
        return self.linear_2(F.silu(self.linear_1(sample)))


def rotary_tables(
    seq_len: int, rotary_dim: int, theta: float = 10000.0, device=None
) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, device=device).float() / rotary_dim))
    steps = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(steps, inv_freq)
    freqs = torch.cat((freqs, freqs), dim=-1)
    return freqs.cos().contiguous(), freqs.sin().contiguous()


def apply_partial_rotary(x: torch.Tensor, rotary_emb) -> torch.Tensor:
    # x: [batch, seq, heads, head_dim]; only the leading rotary dims rotate (rotate_half convention)
    cos, sin = rotary_emb
    rotary_dim = cos.shape[-1]
    cos = cos[:, None, :].to(x.dtype)
    sin = sin[:, None, :].to(x.dtype)
    rotated = x[..., :rotary_dim]
    h1, h2 = rotated.chunk(2, dim=-1)
    rotate_half = torch.cat((-h2, h1), dim=-1)
    rotated = rotated * cos + rotate_half * sin
    return torch.cat((rotated, x[..., rotary_dim:]), dim=-1)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        inner = heads * head_dim
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(dim, inner, bias=False)
        self.to_v = nn.Linear(dim, inner, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(inner, dim, bias=False), nn.Dropout(0.0)])

    def forward(self, x: torch.Tensor, rotary_emb) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.to_q(x).view(b, s, self.heads, self.head_dim)
        k = self.to_k(x).view(b, s, self.heads, self.head_dim)
        v = self.to_v(x).view(b, s, self.heads, self.head_dim)
        q = apply_partial_rotary(q, rotary_emb)
        k = apply_partial_rotary(k, rotary_emb)
        # dispatch_attention_fn default (native SDPA), full (non-causal) attention
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        out = out.transpose(1, 2).flatten(2, 3).to(q.dtype)
        return self.to_out[1](self.to_out[0](out))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, ff_inner_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, head_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff_in = nn.Linear(dim, ff_inner_dim * 2)
        self.ff_out = nn.Linear(ff_inner_dim, dim)

    def forward(self, x: torch.Tensor, rotary_emb) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rotary_emb)
        gate_states, gate = self.ff_in(self.norm2(x)).chunk(2, dim=-1)
        return x + self.ff_out(gate_states * F.silu(gate))


class Music3DiT(nn.Module):
    def __init__(self, cfg: DiTConfig = DiTConfig()):
        super().__init__()
        self.cfg = cfg
        inner = cfg.inner_dim
        self.time_proj = FourierEmbedding(cfg.fourier_embedding_dim)
        self.time_embed = TimestepEmbedding(cfg.fourier_embedding_dim, inner)
        self.preprocess_conv = nn.Conv1d(cfg.concat_channels, cfg.concat_channels, 1, bias=False)
        self.proj_in = nn.Linear(cfg.concat_channels, inner, bias=False)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(inner, cfg.num_attention_heads, cfg.attention_head_dim, cfg.ff_inner_dim)
                for _ in range(cfg.num_layers)
            ]
        )
        self.proj_out = nn.Linear(inner, cfg.in_channels, bias=False)
        self.postprocess_conv = nn.Conv1d(cfg.in_channels, cfg.in_channels, 1, bias=False)

    @property
    def dtype(self):
        return self.proj_in.weight.dtype

    def forward(
        self, hidden_states: torch.Tensor, timestep: torch.Tensor, encoder_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """hidden_states [B, 128, L] noisy latents; timestep [B] in [0,1] (0 = noise); encoder_hidden_states [B, L, 2048]."""
        zeros = torch.zeros_like(hidden_states)
        x = torch.cat((hidden_states, zeros, encoder_hidden_states.transpose(1, 2)), dim=1)
        x = self.preprocess_conv(x) + x
        x = x.transpose(1, 2)
        temb = self.time_embed(self.time_proj(timestep))
        x = self.proj_in(x)
        x = torch.cat((temb.unsqueeze(1), x), dim=1)
        rotary_emb = rotary_tables(x.shape[1], self.cfg.rotary_dim, device=x.device)
        for block in self.transformer_blocks:
            x = block(x, rotary_emb)
        x = self.proj_out(x[:, 1:]).transpose(1, 2)
        return self.postprocess_conv(x) + x

    @staticmethod
    def load(
        snapshot, dtype=torch.float32, device="cpu", cfg: DiTConfig | None = None, num_layers: int | None = None
    ) -> "Music3DiT":
        from safetensors.torch import load_file

        cfg = cfg or DiTConfig()
        if num_layers is not None:
            cfg = DiTConfig(**{**cfg.__dict__, "num_layers": num_layers})
        sd = {}
        for shard in sorted(Path(snapshot, "transformer").glob("*.safetensors")):
            sd.update(load_file(str(shard), device="cpu"))
        if num_layers is not None:
            sd = {
                k: v
                for k, v in sd.items()
                if not k.startswith("transformer_blocks.") or int(k.split(".")[1]) < num_layers
            }
        with torch.device("meta"):
            m = Music3DiT(cfg)
        m.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, strict=True, assign=True)
        return m.to(device).eval()
