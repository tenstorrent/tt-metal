# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Self-contained torch CPU reference for the HaMeR per-frame regressor used
by Dyn-HaMR.

Adapted from ``geopavlakos/hamer`` (Pavlakos et al., ICCV 2023) with upstream
dependencies on ``timm``, ``pytorch_lightning``, ``yacs``, ``smplx`` and
``chumpy`` removed — this file needs only ``torch`` and ``einops`` so it runs
in the Tenstorrent venv without extra packages.  Random weights, deterministic
under ``torch.manual_seed``; used as the accuracy anchor for the tt-nn port.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class HamerConfig:
    # Image / patch
    img_size: tuple = (256, 192)
    patch_size: int = 16
    in_chans: int = 3
    # Backbone (ViT-H)
    embed_dim: int = 1280
    depth: int = 32
    num_heads: int = 16
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    # MANO head
    num_hand_joints: int = 15
    joint_rep_dim: int = 6     # 6-D continuous rotation per joint
    ief_iters: int = 1
    head_dim: int = 1024
    head_depth: int = 6
    head_heads: int = 8
    head_dim_head: int = 64
    head_mlp_dim: int = 1024

    @property
    def npose(self) -> int:
        return self.joint_rep_dim * (self.num_hand_joints + 1)  # 16 × 6 = 96


# --------------------------------------------------------------------------- #
# ViT-H backbone (no timm)
# --------------------------------------------------------------------------- #
class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, qkv_bias: bool) -> None:
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size: tuple, patch_size: int, in_chans: int, embed_dim: int) -> None:
        super().__init__()
        # HaMeR's upstream conv uses patch kernel with pad=4; preserve that geometry
        # (output H,W = 16,12 for 256×192 input with stride 16 and pad 4).
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, padding=4)
        self.img_h = img_size[0]
        self.img_w = img_size[1]

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        return x, (Hp, Wp)


class ViT(nn.Module):
    def __init__(self, cfg: HamerConfig) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(cfg.img_size, cfg.patch_size, cfg.in_chans, cfg.embed_dim)
        # Precompute expected patch grid shape for the configured resolution.
        ph = cfg.img_size[0] // cfg.patch_size + (2 * 4 // cfg.patch_size)
        pw = cfg.img_size[1] // cfg.patch_size + (2 * 4 // cfg.patch_size)
        # Actual Conv output count at 256×192 with stride 16 pad 4: H'=W'=17 / 13 — derive at runtime.
        num_patches = ((cfg.img_size[0] + 2 * 4 - cfg.patch_size) // cfg.patch_size + 1) * \
                      ((cfg.img_size[1] + 2 * 4 - cfg.patch_size) // cfg.patch_size + 1)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, cfg.embed_dim))
        self.blocks = nn.ModuleList([
            Block(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, cfg.qkv_bias)
            for _ in range(cfg.depth)
        ])
        self.last_norm = nn.LayerNorm(cfg.embed_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x, (Hp, Wp) = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]
        for blk in self.blocks:
            x = blk(x)
        x = self.last_norm(x)
        return x.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()


# --------------------------------------------------------------------------- #
# MANO transformer decoder head (cross-attn)
# --------------------------------------------------------------------------- #
class _FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int) -> None:
        super().__init__()
        inner = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_out = nn.Linear(inner, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in qkv)
        attn = (q @ k.transpose(-1, -2) * self.scale).softmax(dim=-1)
        out = rearrange(attn @ v, "b h n d -> b n (h d)")
        return self.to_out(out)


class _CrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, context_dim: Optional[int] = None) -> None:
        super().__init__()
        inner = dim_head * heads
        context_dim = context_dim or dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_kv = nn.Linear(context_dim, inner * 2, bias=False)
        self.to_out = nn.Linear(inner, dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q = self.to_q(x)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in (q, k, v))
        attn = (q @ k.transpose(-1, -2) * self.scale).softmax(dim=-1)
        out = rearrange(attn @ v, "b h n d -> b n (h d)")
        return self.to_out(out)


class _CrossAttnBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, context_dim: int) -> None:
        super().__init__()
        self.norm_sa = nn.LayerNorm(dim)
        self.sa = _SelfAttention(dim, heads, dim_head)
        self.norm_ca = nn.LayerNorm(dim)
        self.ca = _CrossAttention(dim, heads, dim_head, context_dim=context_dim)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = _FeedForward(dim, mlp_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.norm_sa(x))
        x = x + self.ca(self.norm_ca(x), context=context)
        x = x + self.ff(self.norm_ff(x))
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, num_tokens: int, token_dim: int, dim: int, depth: int,
                 heads: int, dim_head: int, mlp_dim: int, context_dim: int) -> None:
        super().__init__()
        self.to_token_embedding = nn.Linear(token_dim, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, dim) * 0.02)
        self.layers = nn.ModuleList([
            _CrossAttnBlock(dim, heads, dim_head, mlp_dim, context_dim) for _ in range(depth)
        ])

    def forward(self, tokens: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = self.to_token_embedding(tokens) + self.pos_embedding[:, : tokens.shape[1]]
        for blk in self.layers:
            x = blk(x, context)
        return x


def rot6d_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    """Convert 6-D continuous rotation to 3×3 rotation matrix (Zhou et al., CVPR'19)."""
    x = x.reshape(-1, 3, 2)
    a1 = x[..., 0]
    a2 = x[..., 1]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)  # (N, 3, 3)


class MANOHead(nn.Module):
    def __init__(self, cfg: HamerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.transformer = TransformerDecoder(
            num_tokens=1,
            token_dim=1,
            dim=cfg.head_dim,
            depth=cfg.head_depth,
            heads=cfg.head_heads,
            dim_head=cfg.head_dim_head,
            mlp_dim=cfg.head_mlp_dim,
            context_dim=cfg.embed_dim,
        )
        self.decpose = nn.Linear(cfg.head_dim, cfg.npose)
        self.decshape = nn.Linear(cfg.head_dim, 10)
        self.deccam = nn.Linear(cfg.head_dim, 3)
        # Mean-param buffers (upstream loads from a .npz; zeros give a stable
        # PCC anchor until real checkpoint weights land).
        self.register_buffer("init_hand_pose", torch.zeros(1, cfg.npose))
        self.register_buffer("init_betas", torch.zeros(1, 10))
        self.register_buffer("init_cam", torch.zeros(1, 3))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B = features.shape[0]
        context = rearrange(features, "b c h w -> b (h w) c")
        token = torch.zeros(B, 1, 1, device=features.device, dtype=features.dtype)
        pose = self.init_hand_pose.expand(B, -1)
        betas = self.init_betas.expand(B, -1)
        cam = self.init_cam.expand(B, -1)
        for _ in range(self.cfg.ief_iters):
            tok_out = self.transformer(token, context=context).squeeze(1)
            pose = self.decpose(tok_out) + pose
            betas = self.decshape(tok_out) + betas
            cam = self.deccam(tok_out) + cam
        rotmats = rot6d_to_rotmat(pose).view(B, self.cfg.num_hand_joints + 1, 3, 3)
        # Flatten to one tensor for a stable scalar PCC anchor.
        return torch.cat([rotmats.reshape(B, -1), betas, cam], dim=-1)  # (B, 16·9+10+3)=157


class Hamer(nn.Module):
    def __init__(self, cfg: Optional[HamerConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or HamerConfig()
        self.backbone = ViT(self.cfg)
        self.head = MANOHead(self.cfg)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(image)
        return self.head(feats)


def build_reference(seed: int = 0) -> Hamer:
    """Random-weight torch CPU reference; deterministic under the given seed."""
    torch.manual_seed(seed)
    model = Hamer()
    model.eval()
    return model


def sample_input(batch: int = 1, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed + 1)
    return torch.randn(batch, 3, 256, 192, generator=g)


def build_paired(tt_module, device_id: int = 0, seed: int = 0):
    """Harness hook — returns (ref_model, tt_model, sample_tensor).

    ``tt_module`` is the imported ``models.experimental.dyn_hamr.tt.hamer``
    namespace; it must expose ``build_from_reference(ref_model, device_id)``
    returning a callable with the same output contract as ``Hamer.forward``.
    """
    ref = build_reference(seed=seed)
    sample = sample_input(seed=seed)
    tt_model = tt_module.build_from_reference(ref, device_id=device_id)
    return ref, tt_model, sample
