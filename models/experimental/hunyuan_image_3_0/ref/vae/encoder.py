# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
HunyuanImage-3.0 VAE encoder — PyTorch reference (single file).

  conv_in:  [1, 3, 4, 1024, 1024] -> [1, 128, 4, 1024, 1024]
  down:     [1, 128, 4, 1024, 1024] -> [1, 1024, 1, 64, 64]
  mid:      [1, 1024, 1, 64, 64] -> [1, 1024, 1, 64, 64]
  head:     norm_out + conv_out -> [1, 64, 1, 64, 64]  (mu || log_sigma)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from models.experimental.hunyuan_image_3_0.ref.weights import MODEL_DIR, load_prefixed_state_dict, load_tensors

IN_CHANNELS = 3
Z_CHANNELS = 32
OUT_PARAM_CHANNELS = 2 * Z_CHANNELS
BLOCK_OUT_CHANNELS = (128, 256, 512, 1024, 1024)
MID_CHANNELS = 1024
PIXEL_T = 4
PIXEL_H = 1024
PIXEL_W = 1024
LATENT_T = 1
LATENT_H = 64
LATENT_W = 64
NUM_GROUPS = 32
GN_EPS = 1e-6
NUM_RES_BLOCKS = 2
FFACTOR_SPATIAL = 16
FFACTOR_TEMPORAL = 4


class DownLevelSpec(NamedTuple):
    level: int
    block_channels: int
    in_channels: int
    t: int
    h: int
    w: int
    has_downsample: bool
    downsample_out_channels: int | None
    add_temporal_downsample: bool


def encoder_down_level_specs(
    pixel_t: int = PIXEL_T,
    pixel_h: int = PIXEL_H,
    pixel_w: int = PIXEL_W,
) -> list[DownLevelSpec]:
    """Spatial/channel schedule for each encoder down level (post-conv_in)."""
    t, h, w = pixel_t, pixel_h, pixel_w
    block_in = BLOCK_OUT_CHANNELS[0]
    specs: list[DownLevelSpec] = []

    for i_level, ch in enumerate(BLOCK_OUT_CHANNELS):
        add_spatial = i_level < int(math.log2(FFACTOR_SPATIAL))
        add_temporal = add_spatial and i_level >= int(math.log2(FFACTOR_SPATIAL // FFACTOR_TEMPORAL))
        has_downsample = add_spatial or add_temporal
        downsample_out = (
            BLOCK_OUT_CHANNELS[i_level + 1] if has_downsample and i_level < len(BLOCK_OUT_CHANNELS) - 1 else None
        )

        specs.append(
            DownLevelSpec(
                level=i_level,
                block_channels=ch,
                in_channels=block_in,
                t=t,
                h=h,
                w=w,
                has_downsample=has_downsample and downsample_out is not None,
                downsample_out_channels=downsample_out,
                add_temporal_downsample=add_temporal,
            )
        )

        if has_downsample and downsample_out is not None:
            r1 = 2 if add_temporal else 1
            t //= r1
            h //= 2
            w //= 2
            block_in = downsample_out

    return specs


def encoder_head_shape(
    pixel_t: int = PIXEL_T,
    pixel_h: int = PIXEL_H,
    pixel_w: int = PIXEL_W,
) -> tuple[int, int, int, int]:
    """Return (T, H, W, C) at mid / encoder head input."""
    specs = encoder_down_level_specs(pixel_t, pixel_h, pixel_w)
    last = specs[-1]
    return last.t, last.h, last.w, last.block_channels


class Conv3d(nn.Conv3d):
    """Symmetric-padded Conv3d; chunks along T when activation memory exceeds ~2 GiB."""

    def forward(self, input: Tensor) -> Tensor:
        _b, c, t, h, w = input.shape
        memory_count = (c * t * h * w) * 2 / 1024**3
        if memory_count > 2:
            n_split = math.ceil(memory_count / 2)
            assert n_split >= 2
            chunks = torch.chunk(input, chunks=n_split, dim=-3)
            padded_chunks = []
            for i, chunk in enumerate(chunks):
                if self.padding[0] > 0:
                    padded_chunk = F.pad(
                        chunk,
                        (0, 0, 0, 0, self.padding[0], self.padding[0]),
                        mode="constant" if self.padding_mode == "zeros" else self.padding_mode,
                        value=0,
                    )
                    if i > 0:
                        padded_chunk[:, :, : self.padding[0]] = chunks[i - 1][:, :, -self.padding[0] :]
                    if i < len(chunks) - 1:
                        padded_chunk[:, :, -self.padding[0] :] = chunks[i + 1][:, :, : self.padding[0]]
                else:
                    padded_chunk = chunk
                padded_chunks.append(padded_chunk)
            padding_bak = self.padding
            self.padding = (0, self.padding[1], self.padding[2])
            outputs = [super().forward(padded_chunk) for padded_chunk in padded_chunks]
            self.padding = padding_bak
            return torch.cat(outputs, dim=-3)
        return super().forward(input)


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
    """mid.block_1 -> mid.attn_1 -> mid.block_2."""

    def __init__(self, channels: int = MID_CHANNELS):
        super().__init__()
        self.block_1 = ResnetBlock(channels, channels)
        self.attn_1 = AttnBlock(channels)
        self.block_2 = ResnetBlock(channels, channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.block_1(x)
        x = self.attn_1(x)
        return self.block_2(x)


class DownsampleDCAE(nn.Module):
    """Space-to-depth downsample with conv residual (matches Hunyuan autoencoder_kl_3d)."""

    def __init__(self, in_channels: int, out_channels: int, add_temporal_downsample: bool = True):
        super().__init__()
        factor = 2 * 2 * 2 if add_temporal_downsample else 1 * 2 * 2
        assert out_channels % factor == 0
        self.conv = Conv3d(in_channels, out_channels // factor, kernel_size=3, stride=1, padding=1)
        self.add_temporal_downsample = add_temporal_downsample
        self.group_size = factor * in_channels // out_channels

    def forward(self, x: Tensor) -> Tensor:
        r1 = 2 if self.add_temporal_downsample else 1
        h = self.conv(x)
        h = rearrange(h, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
        shortcut = rearrange(x, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
        b, c, t, height, width = shortcut.shape
        shortcut = shortcut.view(b, h.shape[1], self.group_size, t, height, width).mean(dim=2)
        return h + shortcut


class DownBlock(nn.Module):
    """One encoder down level: num_res_blocks ResnetBlocks + optional DownsampleDCAE."""

    def __init__(
        self,
        block_in: int,
        block_channels: int,
        *,
        has_downsample: bool,
        downsample_out_channels: int | None = None,
        add_temporal_downsample: bool = False,
        num_res_blocks: int = NUM_RES_BLOCKS,
    ):
        super().__init__()
        self.block = nn.ModuleList()
        in_ch = block_in
        for _ in range(num_res_blocks):
            self.block.append(ResnetBlock(in_ch, block_channels))
            in_ch = block_channels

        self.downsample = None
        if has_downsample:
            assert downsample_out_channels is not None
            self.downsample = DownsampleDCAE(
                in_ch, downsample_out_channels, add_temporal_downsample=add_temporal_downsample
            )

    def forward(self, x: Tensor) -> Tensor:
        for block in self.block:
            x = block(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class EncoderDown(nn.Module):
    """All encoder down levels (post-conv_in)."""

    def __init__(
        self,
        block_out_channels: tuple[int, ...] = BLOCK_OUT_CHANNELS,
        num_res_blocks: int = NUM_RES_BLOCKS,
        ffactor_spatial: int = FFACTOR_SPATIAL,
        ffactor_temporal: int = FFACTOR_TEMPORAL,
        downsample_match_channel: bool = True,
    ):
        super().__init__()
        self.down = nn.ModuleList()
        block_in = block_out_channels[0]
        for i_level, ch in enumerate(block_out_channels):
            add_spatial = i_level < int(math.log2(ffactor_spatial))
            add_temporal = add_spatial and i_level >= int(math.log2(ffactor_spatial // ffactor_temporal))
            has_downsample = add_spatial or add_temporal
            downsample_out = (
                block_out_channels[i_level + 1] if has_downsample and i_level < len(block_out_channels) - 1 else None
            )
            if has_downsample and not downsample_match_channel:
                downsample_out = block_in

            self.down.append(
                DownBlock(
                    block_in,
                    ch,
                    has_downsample=has_downsample and downsample_out is not None,
                    downsample_out_channels=downsample_out,
                    add_temporal_downsample=add_temporal,
                    num_res_blocks=num_res_blocks,
                )
            )
            if has_downsample and downsample_out is not None:
                block_in = downsample_out

    def forward(self, x: Tensor) -> Tensor:
        for down_block in self.down:
            x = down_block(x)
        return x


class ConvIn(nn.Module):
    """3x3 Conv3d(3 -> 128), no shortcut."""

    def __init__(self, in_channels: int = IN_CHANNELS, out_channels: int = BLOCK_OUT_CHANNELS[0]):
        super().__init__()
        self.conv = Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class EncoderHead(nn.Module):
    """norm_out -> swish -> conv_out + channel-group mean shortcut."""

    def __init__(
        self,
        in_channels: int = MID_CHANNELS,
        out_channels: int = OUT_PARAM_CHANNELS,
        z_channels: int = Z_CHANNELS,
    ):
        super().__init__()
        self.group_size = in_channels // out_channels
        assert in_channels == out_channels * self.group_size
        self.norm_out = nn.GroupNorm(num_groups=NUM_GROUPS, num_channels=in_channels, eps=GN_EPS, affine=True)
        self.conv_out = Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        shortcut = rearrange(x, "b (c r) f h w -> b c r f h w", r=self.group_size).mean(dim=2)
        h = swish(self.norm_out(x))
        return self.conv_out(h) + shortcut


class Encoder(nn.Module):
    """Full Hunyuan VAE encoder: conv_in -> down -> mid -> head."""

    def __init__(self) -> None:
        super().__init__()
        head_t, head_h, head_w, head_c = encoder_head_shape()
        del head_t, head_h, head_w
        self.conv_in = ConvIn()
        self.down = EncoderDown()
        self.mid = MidBlock(head_c)
        self.head = EncoderHead(head_c, OUT_PARAM_CHANNELS, Z_CHANNELS)

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv_in(x)
        h = self.down(h)
        h = self.mid(h)
        return self.head(h)


def _load_submodule(module: nn.Module, model_dir: Path, prefix: str, dtype: torch.dtype) -> nn.Module:
    module.load_state_dict(load_prefixed_state_dict(model_dir, prefix, dtype=dtype))
    module.to(dtype=dtype)
    module.eval()
    return module


def load_conv_in(model_dir: Path = MODEL_DIR, dtype: torch.dtype = torch.float32) -> ConvIn:
    module = ConvIn()
    weights = load_tensors(
        model_dir,
        ["vae.encoder.conv_in.weight", "vae.encoder.conv_in.bias"],
    )
    module.conv.load_state_dict(
        {
            "weight": weights["vae.encoder.conv_in.weight"].to(dtype),
            "bias": weights["vae.encoder.conv_in.bias"].to(dtype),
        }
    )
    module.to(dtype=dtype)
    module.eval()
    return module


def load_down_block(level: int, model_dir: Path = MODEL_DIR, dtype: torch.dtype = torch.float32) -> DownBlock:
    spec = encoder_down_level_specs()[level]
    module = DownBlock(
        spec.in_channels,
        spec.block_channels,
        has_downsample=spec.has_downsample,
        downsample_out_channels=spec.downsample_out_channels,
        add_temporal_downsample=spec.add_temporal_downsample,
    )
    return _load_submodule(module, model_dir, f"vae.encoder.down.{level}.", dtype)


def load_encoder_down(model_dir: Path = MODEL_DIR, dtype: torch.dtype = torch.float32) -> EncoderDown:
    module = EncoderDown()
    state = load_prefixed_state_dict(model_dir, "vae.encoder.down.", dtype=dtype)
    module.load_state_dict({f"down.{k}": v for k, v in state.items()})
    module.to(dtype=dtype)
    module.eval()
    return module


def load_mid(model_dir: Path = MODEL_DIR, dtype: torch.dtype = torch.float32) -> MidBlock:
    return _load_submodule(MidBlock(), model_dir, "vae.encoder.mid.", dtype)


def load_encoder_head(model_dir: Path = MODEL_DIR, dtype: torch.dtype = torch.float32) -> EncoderHead:
    _, _, _, head_c = encoder_head_shape()
    module = EncoderHead(head_c, OUT_PARAM_CHANNELS, Z_CHANNELS)
    weights = load_tensors(
        model_dir,
        [
            "vae.encoder.norm_out.weight",
            "vae.encoder.norm_out.bias",
            "vae.encoder.conv_out.weight",
            "vae.encoder.conv_out.bias",
        ],
    )
    module.norm_out.load_state_dict(
        {
            "weight": weights["vae.encoder.norm_out.weight"].to(dtype),
            "bias": weights["vae.encoder.norm_out.bias"].to(dtype),
        }
    )
    module.conv_out.load_state_dict(
        {
            "weight": weights["vae.encoder.conv_out.weight"].to(dtype),
            "bias": weights["vae.encoder.conv_out.bias"].to(dtype),
        }
    )
    module.to(dtype=dtype)
    module.eval()
    return module


def load_encoder(model_dir: Path = MODEL_DIR, dtype: torch.dtype = torch.float32) -> Encoder:
    module = Encoder()
    module.conv_in = load_conv_in(model_dir, dtype)
    module.down = load_encoder_down(model_dir, dtype)
    module.mid = load_mid(model_dir, dtype)
    module.head = load_encoder_head(model_dir, dtype)
    return module


@torch.no_grad()
def encode_pixels(x: Tensor, model_dir: Path = MODEL_DIR, dtype: torch.dtype = torch.float32) -> Tensor:
    return load_encoder(model_dir, dtype)(x.float())


def get_input() -> Tensor:
    torch.manual_seed(42)
    return torch.randn(1, IN_CHANNELS, PIXEL_T, PIXEL_H, PIXEL_W, dtype=torch.float32)


def get_conv_in_output_shape_input() -> Tensor:
    torch.manual_seed(43)
    return torch.randn(1, BLOCK_OUT_CHANNELS[0], PIXEL_T, PIXEL_H, PIXEL_W, dtype=torch.float32)


def get_down_level_input(level: int) -> Tensor:
    spec = encoder_down_level_specs()[level]
    torch.manual_seed(44 + level)
    return torch.randn(1, spec.in_channels, spec.t, spec.h, spec.w, dtype=torch.float32)


def get_encoder_down_input() -> Tensor:
    return get_conv_in_output_shape_input()


def get_mid_input() -> Tensor:
    head_t, head_h, head_w, head_c = encoder_head_shape()
    torch.manual_seed(55)
    return torch.randn(1, head_c, head_t, head_h, head_w, dtype=torch.float32)


def get_encoder_head_input() -> Tensor:
    return get_mid_input()


@torch.no_grad()
def run_conv_in_smoke_test(model_dir: Path = MODEL_DIR) -> Tensor:
    conv_in = load_conv_in(model_dir)
    x = get_input()
    out = conv_in(x)
    print(f"conv_in: in={tuple(x.shape)} out={tuple(out.shape)} dtype={out.dtype}")
    return out
