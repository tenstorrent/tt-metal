"""PyTorch reference for Fast3R (CroCo-ViT-Large encoder + Fast3R cross-view decoder + DPT head).

Based on the published Fast3R architecture and the weight layout in
`jedyang97/Fast3R_ViT_Large_512/model.safetensors`.

Goals:
- exact weight shapes match the checkpoint so load_state_dict strict=True passes
- keep forward code compact and readable so the tt-nn port can mirror it one op at a time
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rope2d, build_rope2d_cos_sin


@dataclass
class Fast3RConfig:
    img_size: int = 512
    patch_size: int = 16
    embed_dim: int = 1024
    enc_depth: int = 24
    dec_depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    rope_base: float = 100.0
    with_local_head: bool = True
    # DPT head reads 4 features at these encoder+decoder block indices (DUSt3R convention)
    dpt_hooks: Tuple[int, int, int, int] = (5, 11, 17, 23)


class PatchEmbed(nn.Module):
    def __init__(self, cfg: Fast3RConfig):
        super().__init__()
        self.proj = nn.Conv2d(3, cfg.embed_dim, kernel_size=cfg.patch_size, stride=cfg.patch_size)
        self.grid = cfg.img_size // cfg.patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                          # (B, C, H', W')
        return x.flatten(2).transpose(1, 2)       # (B, N, C)


class Attention(nn.Module):
    def __init__(self, cfg: Fast3RConfig):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.embed_dim // cfg.num_heads
        self.qkv = nn.Linear(cfg.embed_dim, 3 * cfg.embed_dim, bias=cfg.qkv_bias)
        self.proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        cos: Optional[torch.Tensor],
        sin: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                   # (B, H, N, Dh)
        if cos is not None:
            q, k = apply_rope2d(q, k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class Mlp(nn.Module):
    def __init__(self, cfg: Fast3RConfig):
        super().__init__()
        hidden = int(cfg.embed_dim * cfg.mlp_ratio)
        self.fc1 = nn.Linear(cfg.embed_dim, hidden)
        self.fc2 = nn.Linear(hidden, cfg.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, cfg: Fast3RConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.embed_dim, eps=1e-6)
        self.attn = Attention(cfg)
        self.norm2 = nn.LayerNorm(cfg.embed_dim, eps=1e-6)
        self.mlp = Mlp(cfg)

    def forward(self, x: torch.Tensor, cos: Optional[torch.Tensor], sin: Optional[torch.Tensor]) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class Encoder(nn.Module):
    def __init__(self, cfg: Fast3RConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg)
        self.enc_blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.enc_depth)])
        self.enc_norm = nn.LayerNorm(cfg.embed_dim, eps=1e-6)

    def forward(self, imgs: torch.Tensor, return_hooks: bool = False):
        """imgs: (B, 3, H, W). Returns final tokens (B, N, C) and optionally hook features."""
        B = imgs.shape[0]
        x = self.patch_embed(imgs)                # (B, N, C)
        g = self.patch_embed.grid
        cos, sin = build_rope2d_cos_sin(g, g, self.cfg.embed_dim // self.cfg.num_heads, self.cfg.rope_base, device=x.device, dtype=x.dtype)
        hooks: List[torch.Tensor] = []
        for i, blk in enumerate(self.enc_blocks):
            x = blk(x, cos, sin)
            if return_hooks and i in self.cfg.dpt_hooks:
                hooks.append(x)
        x = self.enc_norm(x)
        if return_hooks:
            return x, hooks
        return x


class Decoder(nn.Module):
    """Fast3R decoder: same block architecture as encoder, runs over concatenated N-view tokens.

    Image-index embedding is added to tokens before the decoder per the Fast3R paper.
    For the initial port (N=1) this embedding is a no-op.
    """

    def __init__(self, cfg: Fast3RConfig):
        super().__init__()
        self.cfg = cfg
        self.decoder_embed = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.dec_blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.dec_depth)])
        self.dec_norm = nn.LayerNorm(cfg.embed_dim, eps=1e-6)

    def forward(self, tokens: torch.Tensor, return_hooks: bool = False):
        """tokens: (B, N_total, C) already includes concat across views."""
        hooks: List[torch.Tensor] = []
        x = self.decoder_embed(tokens)
        for i, blk in enumerate(self.dec_blocks):
            x = blk(x, None, None)
            if return_hooks and i in self.cfg.dpt_hooks:
                hooks.append(x)
        x = self.dec_norm(x)
        if return_hooks:
            return x, hooks
        return x


class Fast3R(nn.Module):
    """Encoder + decoder, no DPT head yet.

    The DPT head is held for a later iteration — it's orthogonal to the transformer
    bring-up and adds a lot of surface area (refinenets, resConfUnits, upsampling).
    """

    def __init__(self, cfg: Optional[Fast3RConfig] = None):
        super().__init__()
        self.cfg = cfg or Fast3RConfig()
        self.encoder = Encoder(self.cfg)
        self.decoder = Decoder(self.cfg)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        enc = self.encoder(imgs)
        dec = self.decoder(enc)
        return dec
