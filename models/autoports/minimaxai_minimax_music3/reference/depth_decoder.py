"""MiniMaxMusic3RVQDepthDecoder (the 0.6B local LLM) as a plain torch module. Keys match rvq_depth_decoder/*.safetensors."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.autoports.minimaxai_minimax_music3.config import DepthConfig


class RMSNorm(nn.Module):
    """diffusers RMSNorm(dim, eps, elementwise_affine=True): fp32 variance, weight multiply in the weight dtype."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            x = x.to(self.weight.dtype)
        return x * self.weight


class DepthAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.to_q(x).view(b, s, self.heads, self.head_dim).transpose(1, 2)
        k = self.to_k(x).view(b, s, self.heads, self.head_dim).transpose(1, 2)
        v = self.to_v(x).view(b, s, self.heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.to_out(out.transpose(1, 2).flatten(2, 3).to(q.dtype))


class DepthBlock(nn.Module):
    def __init__(self, dim: int, heads: int, intermediate: int):
        super().__init__()
        self.input_layernorm = RMSNorm(dim)
        self.attn = DepthAttention(dim, heads)
        self.post_attention_layernorm = RMSNorm(dim)
        self.gate_proj = nn.Linear(dim, intermediate, bias=False)
        self.up_proj = nn.Linear(dim, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.input_layernorm(x))
        h = self.post_attention_layernorm(x)
        return x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))


class Music3DepthDecoder(nn.Module):
    def __init__(self, cfg: DepthConfig = DepthConfig()):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_size
        self.audio_embeddings = nn.Embedding(cfg.audio_vocab_size * (cfg.num_codebooks - 1), d)
        self.projection = nn.Linear(d, d, bias=False)
        self.pos_embedding = nn.Embedding(cfg.max_position_embeddings, d)
        self.layers = nn.ModuleList(
            [DepthBlock(d, cfg.num_attention_heads, cfg.intermediate_size) for _ in range(cfg.num_layers)]
        )
        self.norm = RMSNorm(d)
        self.audio_heads = nn.ModuleList(
            [nn.Linear(d, cfg.audio_vocab_size, bias=False) for _ in range(cfg.num_codebooks - 1)]
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """inputs_embeds [B, steps, D] (already projected) -> normalized hidden states [B, steps, D]."""
        positions = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        x = inputs_embeds + self.pos_embedding(positions).unsqueeze(0)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

    @staticmethod
    def load(snapshot, dtype=torch.float32, device="cpu", cfg: DepthConfig | None = None) -> "Music3DepthDecoder":
        from safetensors.torch import load_file

        cfg = cfg or DepthConfig()
        sd = load_file(str(Path(snapshot, "rvq_depth_decoder", "diffusion_pytorch_model.safetensors")), device="cpu")
        with torch.device("meta"):
            m = Music3DepthDecoder(cfg)
        m.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, strict=True, assign=True)
        return m.to(device).eval()
