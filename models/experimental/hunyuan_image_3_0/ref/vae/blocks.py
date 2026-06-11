# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Hunyuan VAE decoder building blocks (PyTorch reference)."""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.nn import functional as F

from models.experimental.hunyuan_image_3_0.ref.vae.common import (
    Conv3d,
    GN_EPS,
    MID_CHANNELS,
    NUM_GROUPS,
)


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int | None = None):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=NUM_GROUPS, num_channels=in_channels, eps=GN_EPS, affine=True)
        self.conv1 = Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=NUM_GROUPS, num_channels=out_channels, eps=GN_EPS, affine=True)
        self.conv2 = Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.nin_shortcut = (
            Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)
        h = swish(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)
        if self.nin_shortcut is not None:
            x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.norm = nn.GroupNorm(num_groups=NUM_GROUPS, num_channels=in_channels, eps=GN_EPS, affine=True)
        self.q = Conv3d(in_channels, in_channels, kernel_size=1)
        self.k = Conv3d(in_channels, in_channels, kernel_size=1)
        self.v = Conv3d(in_channels, in_channels, kernel_size=1)
        self.proj_out = Conv3d(in_channels, in_channels, kernel_size=1)

    def attention(self, x: Tensor) -> Tensor:
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        b, c, t, h_sp, w = q.shape
        q = rearrange(q, "b c f h w -> b 1 (f h w) c").contiguous()
        k = rearrange(k, "b c f h w -> b 1 (f h w) c").contiguous()
        v = rearrange(v, "b c f h w -> b 1 (f h w) c").contiguous()
        out = F.scaled_dot_product_attention(q, k, v)
        return rearrange(out, "b 1 (f h w) c -> b c f h w", f=t, h=h_sp, w=w, c=c, b=b)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


class MidBlock(nn.Module):
    """mid.block_1 -> mid.attn_1 -> mid.block_2 at [B, 1024, 1, 64, 64]."""

    def __init__(self, channels: int = MID_CHANNELS):
        super().__init__()
        self.block_1 = ResnetBlock(channels, channels)
        self.attn_1 = AttnBlock(channels)
        self.block_2 = ResnetBlock(channels, channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.block_1(x)
        x = self.attn_1(x)
        return self.block_2(x)
