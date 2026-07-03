# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
HunyuanImage-3.0 VAE decoder — PyTorch reference (single file).

  conv_in:  [1, 32, 1, 64, 64] -> [1, 1024, 1, 64, 64]
  mid:      [1, 1024, 1, 64, 64] -> [1, 1024, 1, 64, 64]
  up:       [1, 1024, 1, 64, 64] -> [1, 128, 4, 1024, 1024]
  tail:     norm_out + conv_out -> [1, 3, 4, 1024, 1024]
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

Z_CHANNELS = 32
BLOCK_IN_CHANNELS = 1024
MID_CHANNELS = 1024
LATENT_T = 1
LATENT_H = 64
LATENT_W = 64
NUM_GROUPS = 32
GN_EPS = 1e-6

# HunyuanImage-3.0 VAE config (encoder block_out_channels reversed for decoder).
ENCODER_BLOCK_CHANNELS = (128, 256, 512, 1024, 1024)
DECODER_BLOCK_CHANNELS = tuple(reversed(ENCODER_BLOCK_CHANNELS))
NUM_RES_BLOCKS = 2
FFACTOR_SPATIAL = 16
FFACTOR_TEMPORAL = 4
OUT_CHANNELS = 3


class UpLevelSpec(NamedTuple):
    level: int
    block_channels: int
    in_channels: int
    t: int
    h: int
    w: int
    has_upsample: bool
    upsample_out_channels: int | None
    add_temporal_upsample: bool


def decoder_up_level_specs(
    latent_t: int = LATENT_T,
    latent_h: int = LATENT_H,
    latent_w: int = LATENT_W,
) -> list[UpLevelSpec]:
    """Spatial/channel schedule for each decoder up level (post-mid)."""
    t, h, w = latent_t, latent_h, latent_w
    block_in = DECODER_BLOCK_CHANNELS[0]
    specs: list[UpLevelSpec] = []

    for i_level, ch in enumerate(DECODER_BLOCK_CHANNELS):
        add_spatial = i_level < int(math.log2(FFACTOR_SPATIAL))
        add_temporal = i_level < int(math.log2(FFACTOR_TEMPORAL))
        has_upsample = add_spatial or add_temporal
        upsample_out = DECODER_BLOCK_CHANNELS[i_level + 1] if has_upsample else None

        specs.append(
            UpLevelSpec(
                level=i_level,
                block_channels=ch,
                in_channels=block_in,
                t=t,
                h=h,
                w=w,
                has_upsample=has_upsample,
                upsample_out_channels=upsample_out,
                add_temporal_upsample=add_temporal,
            )
        )

        if has_upsample:
            r1 = 2 if add_temporal else 1
            t *= r1
            h *= 2
            w *= 2
            block_in = upsample_out  # type: ignore[assignment]

    return specs


def decoder_tail_shape(
    latent_t: int = LATENT_T,
    latent_h: int = LATENT_H,
    latent_w: int = LATENT_W,
) -> tuple[int, int, int, int]:
    """Return (T, H, W, C) at norm_out / conv_out input."""
    specs = decoder_up_level_specs(latent_t, latent_h, latent_w)
    last = specs[-1]
    return last.t, last.h, last.w, last.in_channels


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


class UpsampleDCAE(nn.Module):
    """Depth-to-space upsample with conv residual (matches Hunyuan autoencoder_kl_3d)."""

    def __init__(self, in_channels: int, out_channels: int, add_temporal_upsample: bool = True):
        super().__init__()
        factor = 2 * 2 * 2 if add_temporal_upsample else 1 * 2 * 2
        self.conv = Conv3d(in_channels, out_channels * factor, kernel_size=3, stride=1, padding=1)
        self.add_temporal_upsample = add_temporal_upsample
        self.repeats = factor * out_channels // in_channels
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Tensor:
        r1 = 2 if self.add_temporal_upsample else 1
        h = self.conv(x)
        h = rearrange(h, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2, c=self.out_channels)
        shortcut = x.repeat_interleave(self.repeats, dim=1)
        shortcut = rearrange(
            shortcut, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2, c=self.out_channels
        )
        return h + shortcut


class UpBlock(nn.Module):
    """One decoder up level: (num_res_blocks+1) ResnetBlocks + optional UpsampleDCAE."""

    def __init__(
        self,
        block_in: int,
        block_channels: int,
        *,
        has_upsample: bool,
        upsample_out_channels: int | None = None,
        add_temporal_upsample: bool = False,
        num_res_blocks: int = NUM_RES_BLOCKS,
    ):
        super().__init__()
        self.block = nn.ModuleList()
        in_ch = block_in
        for _ in range(num_res_blocks + 1):
            self.block.append(ResnetBlock(in_ch, block_channels))
            in_ch = block_channels

        self.upsample = None
        if has_upsample:
            assert upsample_out_channels is not None
            self.upsample = UpsampleDCAE(in_ch, upsample_out_channels, add_temporal_upsample=add_temporal_upsample)

    def forward(self, x: Tensor) -> Tensor:
        for block in self.block:
            x = block(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class DecoderUp(nn.Module):
    """All decoder up levels (post-mid)."""

    def __init__(
        self,
        block_out_channels: tuple[int, ...] = DECODER_BLOCK_CHANNELS,
        num_res_blocks: int = NUM_RES_BLOCKS,
        ffactor_spatial: int = FFACTOR_SPATIAL,
        ffactor_temporal: int = FFACTOR_TEMPORAL,
        upsample_match_channel: bool = True,
        block_in: int | None = None,
    ):
        super().__init__()
        self.up = nn.ModuleList()
        in_ch = block_out_channels[0] if block_in is None else block_in
        for i_level, ch in enumerate(block_out_channels):
            add_spatial = i_level < int(math.log2(ffactor_spatial))
            add_temporal = i_level < int(math.log2(ffactor_temporal))
            has_upsample = add_spatial or add_temporal
            upsample_out = block_out_channels[i_level + 1] if has_upsample else None
            if has_upsample and not upsample_match_channel:
                upsample_out = in_ch

            self.up.append(
                UpBlock(
                    in_ch,
                    ch,
                    has_upsample=has_upsample,
                    upsample_out_channels=upsample_out,
                    add_temporal_upsample=add_temporal,
                    num_res_blocks=num_res_blocks,
                )
            )
            if has_upsample:
                in_ch = upsample_out  # type: ignore[assignment]

    def forward(self, x: Tensor) -> Tensor:
        for up_block in self.up:
            x = up_block(x)
        return x


class NormOut(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=NUM_GROUPS, num_channels=channels, eps=GN_EPS, affine=True)

    def forward(self, x: Tensor) -> Tensor:
        return swish(self.norm(x))


class ConvOut(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = OUT_CHANNELS):
        super().__init__()
        self.conv = Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class DecoderTail(nn.Module):
    """norm_out -> conv_out."""

    def __init__(self, in_channels: int, out_channels: int = OUT_CHANNELS):
        super().__init__()
        self.norm_out = NormOut(in_channels)
        self.conv_out = ConvOut(in_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv_out(self.norm_out(x))


class ConvIn(nn.Module):
    """3x3 Conv3d(32 -> 1024) plus channel repeat shortcut."""

    def __init__(self, z_channels: int = Z_CHANNELS, out_channels: int = BLOCK_IN_CHANNELS):
        super().__init__()
        self.repeats = out_channels // z_channels
        assert out_channels % z_channels == 0
        self.conv = Conv3d(z_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        return self.conv(z) + z.repeat_interleave(self.repeats, dim=1)


def _load_submodule(module: nn.Module, model_dir: Path, prefix: str, dtype: torch.dtype) -> nn.Module:
    module.load_state_dict(load_prefixed_state_dict(model_dir, prefix, dtype=dtype))
    module.to(dtype=dtype)
    module.eval()
    return module


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
    return _load_submodule(MidBlock(), model_dir, "vae.decoder.mid.", dtype)


def load_up_block(level: int, model_dir: Path = MODEL_DIR, dtype: torch.dtype = torch.float32) -> UpBlock:
    spec = decoder_up_level_specs()[level]
    module = UpBlock(
        spec.in_channels,
        spec.block_channels,
        has_upsample=spec.has_upsample,
        upsample_out_channels=spec.upsample_out_channels,
        add_temporal_upsample=spec.add_temporal_upsample,
    )
    return _load_submodule(module, model_dir, f"vae.decoder.up.{level}.", dtype)


def load_decoder_up(model_dir: Path = MODEL_DIR, dtype: torch.dtype = torch.float32) -> DecoderUp:
    module = DecoderUp()
    state = load_prefixed_state_dict(model_dir, "vae.decoder.up.", dtype=dtype)
    module.load_state_dict({f"up.{k}": v for k, v in state.items()})
    module.to(dtype=dtype)
    module.eval()
    return module


def load_norm_out(model_dir: Path = MODEL_DIR, dtype: torch.dtype = torch.float32) -> NormOut:
    _, _, _, tail_c = decoder_tail_shape()
    module = NormOut(tail_c)
    weights = load_tensors(
        model_dir,
        ["vae.decoder.norm_out.weight", "vae.decoder.norm_out.bias"],
    )
    module.norm.load_state_dict(
        {
            "weight": weights["vae.decoder.norm_out.weight"].to(dtype),
            "bias": weights["vae.decoder.norm_out.bias"].to(dtype),
        }
    )
    module.to(dtype=dtype)
    module.eval()
    return module


def load_conv_out(model_dir: Path = MODEL_DIR, dtype: torch.dtype = torch.float32) -> ConvOut:
    tail_t, tail_h, tail_w, tail_c = decoder_tail_shape()
    del tail_t, tail_h, tail_w
    module = ConvOut(tail_c, OUT_CHANNELS)
    weights = load_tensors(
        model_dir,
        ["vae.decoder.conv_out.weight", "vae.decoder.conv_out.bias"],
    )
    module.conv.load_state_dict(
        {
            "weight": weights["vae.decoder.conv_out.weight"].to(dtype),
            "bias": weights["vae.decoder.conv_out.bias"].to(dtype),
        }
    )
    module.to(dtype=dtype)
    module.eval()
    return module


def load_decoder_tail(model_dir: Path = MODEL_DIR, dtype: torch.dtype = torch.float32) -> DecoderTail:
    _, _, _, tail_c = decoder_tail_shape()
    module = DecoderTail(tail_c, OUT_CHANNELS)
    module.norm_out = load_norm_out(model_dir, dtype)
    module.conv_out = load_conv_out(model_dir, dtype)
    return module


class Decoder(nn.Module):
    """Full Hunyuan VAE decoder: conv_in -> mid -> up -> tail."""

    def __init__(self) -> None:
        super().__init__()
        tail_t, tail_h, tail_w, tail_c = decoder_tail_shape()
        del tail_t, tail_h, tail_w
        self.conv_in = ConvIn()
        self.mid = MidBlock()
        self.up = DecoderUp()
        self.tail = DecoderTail(tail_c, OUT_CHANNELS)

    def forward(self, z: Tensor) -> Tensor:
        h = self.conv_in(z)
        h = self.mid(h)
        h = self.up(h)
        return self.tail(h)


def load_decoder(model_dir: Path = MODEL_DIR, dtype: torch.dtype = torch.float32) -> Decoder:
    module = Decoder()
    module.conv_in = load_conv_in(model_dir, dtype)
    module.mid = load_mid(model_dir, dtype)
    module.up = load_decoder_up(model_dir, dtype)
    module.tail = load_decoder_tail(model_dir, dtype)
    return module


@torch.no_grad()
def decode_latent(z: Tensor, model_dir: Path = MODEL_DIR, dtype: torch.dtype = torch.float32) -> Tensor:
    return load_decoder(model_dir, dtype)(z.float())


def vae_decode_output_to_rgb(image_bcthw: Tensor) -> Tensor:
    """Decoder output BCTHW in [-1, 1] -> RGB BCHW in [0, 1].

    Single-frame latent decode (T_in=1) upsamples to T_out=4; HF AutoencoderKLConv3D
    keeps the last temporal frame only (``decoded[:, :, -1:]``).
    """
    return (image_bcthw[:, :, -1] / 2 + 0.5).clamp(0, 1)


def tensor_to_preview_image(image_bcthw: Tensor, *, frame: int = -1) -> Tensor:
    """Last temporal frame -> uint8 HWC for PIL (matches HF decode when T=1)."""
    x = image_bcthw[0, :, frame].clamp(-1, 1)
    x = ((x + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 0).cpu()
    return x


def get_input() -> Tensor:
    torch.manual_seed(42)
    return torch.randn(1, Z_CHANNELS, LATENT_T, LATENT_H, LATENT_W, dtype=torch.float32)


def get_mid_input() -> Tensor:
    torch.manual_seed(43)
    return torch.randn(1, BLOCK_IN_CHANNELS, LATENT_T, LATENT_H, LATENT_W, dtype=torch.float32)


def get_up_level_input(level: int) -> Tensor:
    spec = decoder_up_level_specs()[level]
    torch.manual_seed(44 + level)
    return torch.randn(1, spec.in_channels, spec.t, spec.h, spec.w, dtype=torch.float32)


def get_decoder_up_input() -> Tensor:
    return get_mid_input()


def get_decoder_tail_input() -> Tensor:
    tail_t, tail_h, tail_w, tail_c = decoder_tail_shape()
    torch.manual_seed(60)
    return torch.randn(1, tail_c, tail_t, tail_h, tail_w, dtype=torch.float32)


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
