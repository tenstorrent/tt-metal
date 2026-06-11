# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
HunyuanImage-3.0 VAE decoder — PyTorch reference (self-contained).

Implemented:
  conv_in:  [1, 32, 1, 64, 64] -> [1, 1024, 1, 64, 64]
  mid:      [1, 1024, 1, 64, 64] -> [1, 1024, 1, 64, 64]
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from models.experimental.hunyuan_image_3_0.ref.vae.blocks import MidBlock
from models.experimental.hunyuan_image_3_0.ref.vae.common import (
    BLOCK_IN_CHANNELS,
    Conv3d,
    LATENT_H,
    LATENT_W,
    Z_CHANNELS,
)
from models.experimental.hunyuan_image_3_0.ref.vae.weights import MODEL_DIR, load_prefixed_state_dict, load_tensors

__all__ = [
    "BLOCK_IN_CHANNELS",
    "ConvIn",
    "LATENT_H",
    "LATENT_W",
    "MidBlock",
    "Z_CHANNELS",
    "get_input",
    "get_mid_input",
    "load_conv_in",
    "load_mid",
    "run_conv_in_smoke_test",
    "run_mid_smoke_test",
]


class ConvIn(nn.Module):
    """3x3 Conv3d(32 -> 1024) plus channel repeat shortcut."""

    def __init__(self, z_channels: int = Z_CHANNELS, out_channels: int = BLOCK_IN_CHANNELS):
        super().__init__()
        self.repeats = out_channels // z_channels
        assert out_channels % z_channels == 0
        self.conv = Conv3d(z_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        return self.conv(z) + z.repeat_interleave(self.repeats, dim=1)


def load_conv_in(model_dir: Path = MODEL_DIR, dtype: torch.dtype = torch.float32) -> ConvIn:
    module = ConvIn()
    weights = load_tensors(
        model_dir,
        ["vae.decoder.conv_in.weight", "vae.decoder.conv_in.bias"],
    )
    module.conv.load_state_dict(
        {
            "weight": weights["vae.decoder.conv_in.weight"].to(dtype),
            "bias": weights["vae.decoder.conv_in.bias"].to(dtype),
        }
    )
    module.to(dtype=dtype)
    module.eval()
    return module


def load_mid(model_dir: Path = MODEL_DIR, dtype: torch.dtype = torch.float32) -> MidBlock:
    module = MidBlock()
    module.load_state_dict(load_prefixed_state_dict(model_dir, "vae.decoder.mid.", dtype=dtype))
    module.to(dtype=dtype)
    module.eval()
    return module


def get_input() -> Tensor:
    torch.manual_seed(42)
    return torch.randn(1, Z_CHANNELS, 1, LATENT_H, LATENT_W, dtype=torch.float32)


def get_mid_input() -> Tensor:
    """Deterministic activations at mid input shape [1, 1024, 1, 64, 64]."""
    torch.manual_seed(43)
    return torch.randn(1, BLOCK_IN_CHANNELS, 1, LATENT_H, LATENT_W, dtype=torch.float32)


@torch.no_grad()
def run_conv_in_smoke_test(model_dir: Path = MODEL_DIR) -> Tensor:
    conv_in = load_conv_in(model_dir)
    z = get_input()
    out = conv_in(z)
    print(f"conv_in: in={tuple(z.shape)} out={tuple(out.shape)} dtype={out.dtype}")
    print(f"  range=[{out.min().item():.4f}, {out.max().item():.4f}]")
    return out


@torch.no_grad()
def run_mid_smoke_test(model_dir: Path = MODEL_DIR) -> Tensor:
    mid = load_mid(model_dir)
    x = get_mid_input()
    out = mid(x)
    print(f"mid: in={tuple(x.shape)} out={tuple(out.shape)} dtype={out.dtype}")
    print(f"  range=[{out.min().item():.4f}, {out.max().item():.4f}]")
    return out


if __name__ == "__main__":
    run_conv_in_smoke_test()
    run_mid_smoke_test()
