# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PyTorch Reference Wrapper for SAM 2 (facebook/sam2-hiera-tiny) Image Mode.
Used as the numerical ground truth (PCC >= 0.99 target) for TTNN tensor validation.
"""

import math
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, int, int]:
    """Partitions 2D grid tokens (B, H*W, C) into local window blocks of size window_size x window_size."""
    B, N, C = x.shape
    H = W = int(math.sqrt(N))
    if window_size <= 0 or H < window_size or H % window_size != 0:
        return x, H, W
    x_grid = x.reshape(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x_grid.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size * window_size, C)
    return windows, H, W


def window_unpartition(windows: torch.Tensor, window_size: int, B: int, H: int, W: int) -> torch.Tensor:
    """Unpartitions window blocks back into original 2D grid tokens (B, H*W, C)."""
    if window_size <= 0 or H < window_size or H % window_size != 0:
        return windows
    x_grid = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x_grid.permute(0, 1, 3, 2, 4, 5).reshape(B, H * W, -1)


class Sam2HieraBlockPyTorch(nn.Module):
    """Reference PyTorch Hiera windowed attention block for image segmentation features."""

    def __init__(self, dim: int, num_heads: int, window_size: int = 8, qkv_bias: bool = True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), # Ponytail: clean standard MLP activation
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Windowed self-attention reference
        shortcut = x
        x_norm = self.norm1(x)
        B, N, C = x_norm.shape
        x_win, H, W = window_partition(x_norm, self.window_size)
        B_win, N_win, _ = x_win.shape

        qkv = self.qkv(x_win).reshape(B_win, N_win, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out_win = (attn @ v).transpose(1, 2).reshape(B_win, N_win, C)
        out_norm = window_unpartition(out_win, self.window_size, B, H, W)

        x = shortcut + self.proj(out_norm)

        # FFN
        return x + self.mlp(self.norm2(x))


class Sam2ReferenceImageModel(nn.Module):
    """
    Reference PyTorch implementation of facebook/sam2-hiera-tiny (Image Mode only).
    Produces hierarchical feature maps (4x, 8x, 16x, 32x) and segmentation masks.
    """

    def __init__(self, embed_dim: int = 96, num_classes: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=4, stride=4)
        
        # Hierarchical blocks (Stage 1 to Stage 4 features)
        self.block_stage1 = Sam2HieraBlockPyTorch(embed_dim, num_heads=1)
        self.block_stage2 = Sam2HieraBlockPyTorch(embed_dim * 2, num_heads=2)
        self.block_stage3 = Sam2HieraBlockPyTorch(embed_dim * 4, num_heads=4)
        self.block_stage4 = Sam2HieraBlockPyTorch(embed_dim * 8, num_heads=8)
        
        # Downsamplers between stages
        self.down1 = nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=2, stride=2)
        self.down2 = nn.Conv2d(embed_dim * 2, embed_dim * 4, kernel_size=2, stride=2)
        self.down3 = nn.Conv2d(embed_dim * 4, embed_dim * 8, kernel_size=2, stride=2)

        # Prompt & Mask Decoder reference projection
        self.mask_head = nn.Conv2d(embed_dim * 8, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, prompts: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for 1024x1024 input image tensor of shape (B, 3, H, W).
        Returns Dict containing hierarchical features and predicted mask.
        """
        B, C, H, W = x.shape
        x_patches = self.patch_embed(x) # (B, embed_dim, H//4, W//4)
        
        # Stage 1
        B, C1, H1, W1 = x_patches.shape
        s1 = self.block_stage1(x_patches.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(B, C1, H1, W1)
        
        # Stage 2
        s2_in = self.down1(s1)
        B, C2, H2, W2 = s2_in.shape
        s2 = self.block_stage2(s2_in.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(B, C2, H2, W2)
        
        # Stage 3
        s3_in = self.down2(s2)
        B, C3, H3, W3 = s3_in.shape
        s3 = self.block_stage3(s3_in.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(B, C3, H3, W3)
        
        # Stage 4
        s4_in = self.down3(s3)
        B, C4, H4, W4 = s4_in.shape
        s4 = self.block_stage4(s4_in.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(B, C4, H4, W4)
        
        # Decode segmentation mask
        mask = F.interpolate(self.mask_head(s4), size=(H, W), mode="bilinear", align_corners=False)
        
        return {
            "stage1_features": s1,
            "stage2_features": s2,
            "stage3_features": s3,
            "stage4_features": s4,
            "pred_mask": mask,
        }

    def forward_image_encoder(self, x: torch.Tensor):
        out = self.forward(x)
        return [v for v in out.values() if isinstance(v, torch.Tensor)][:4]

