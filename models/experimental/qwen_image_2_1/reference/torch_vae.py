# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Compact torch reference for the Qwen-Image-2.1 VAE *decoder*, single frame.

`AutoencoderKLQwenImage21` is a 3D (video) VAE, but a still image is one frame and every temporal
mechanism degenerates:

  * `QwenImage21CausalConv3d` squeezes the time axis away and is a plain `nn.Conv2d` with the
    symmetric spatial padding it stashed in `_padding`.
  * `_decode` runs the decoder once with `feat_cache` freshly cleared (all-`None`) and
    `first_chunk=True`. Every cache slot therefore takes its "first chunk" branch, which for the
    `upsample3d` resamplers means `feat_cache[idx] = "Rep"` and **no** `time_conv` at all.
  * `QwenImage21DupUp3D` keeps only frame `factor_t - 1`, which collapses to a pure channel/pixel
    gather (see `dup_up_fast` and the literal `dup_up_reference` it is checked against).

So the whole decoder is 2D and this module works on `[B, C, H, W]` tensors. Parameter names and
shapes match the checkpoint exactly (`post_quant_conv.*`, `decoder.*`), so `load_state_dict` takes
the safetensors state dict unmodified.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.config import VAE, VAEConfig

# --------------------------------------------------------------------------------------- DupUp3D


def dup_up_reference(
    x5: torch.Tensor, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 2, first_chunk: bool = True
) -> torch.Tensor:
    """Literal transcription of `QwenImage21DupUp3D.forward` (5D `[B, C, T, H, W]` in and out)."""
    factor = factor_t * factor_s * factor_s
    assert out_channels * factor % in_channels == 0
    repeats = out_channels * factor // in_channels

    x = x5.repeat_interleave(repeats, dim=1)
    x = x.view(x.size(0), out_channels, factor_t, factor_s, factor_s, x.size(2), x.size(3), x.size(4))
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
    x = x.view(x.size(0), out_channels, x.size(2) * factor_t, x.size(4) * factor_s, x.size(6) * factor_s)
    if first_chunk:
        x = x[:, :, factor_t - 1 :, :, :]
    return x


def dup_up_fast(x: torch.Tensor, in_channels: int, out_channels: int, factor_t: int) -> torch.Tensor:
    """`dup_up_reference` for one frame and `factor_s == 2`, as a 4D `[B, C, H, W]` gather.

    Writing `j = co * factor + a * 4 + b * 2 + d` for the index into the channel-repeated tensor
    (`a` the temporal offset, `b` the row offset, `d` the column offset) and `c_in = j // repeats`,
    the kept frame `a = factor_t - 1` gives:

      * `factor_t == 2`, `in == out`     -> `c_in = co`          (nearest 2x upsample)
      * `factor_t == 2`, `in == 2 * out` -> `c_in = 2 * co + 1`  (odd channels, nearest 2x upsample)
      * `factor_t == 1`, `in == 2 * out` -> `c_in = 2 * co + b`  (row parity picks the channel)
    """
    if factor_t == 2:
        src = x if in_channels == out_channels else x[:, 1::2]
        assert src.shape[1] == out_channels, (in_channels, out_channels)
        return src.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)

    assert factor_t == 1 and in_channels == 2 * out_channels, (in_channels, out_channels, factor_t)
    b, _, h, w = x.shape
    rows = torch.stack([x[:, 0::2], x[:, 1::2]], dim=3)  # [B, Co, H, 2, W]
    rows = rows.reshape(b, out_channels, 2 * h, w)
    return rows.repeat_interleave(2, dim=3)


# ----------------------------------------------------------------------------------------- layers


class RMSNormCF(nn.Module):
    """`QwenImage21RMS_norm(channel_first=True)` applied to the channel dim of `[B, C, H, W]`.

    `images` only selects the stored `gamma` shape ((C, 1, 1) vs (C, 1, 1, 1)); it is kept so the
    checkpoint tensors load without reshaping.
    """

    def __init__(self, dim: int, images: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones((dim, 1, 1) if images else (dim, 1, 1, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(x.float(), dim=1).to(x.dtype)
        return normalized * self.scale * self.gamma.reshape(1, self.dim, 1, 1)


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.norm1 = RMSNormCF(in_dim, images=False)
        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, padding=1)
        self.norm2 = RMSNormCF(out_dim, images=False)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = nn.Conv2d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_shortcut(x)
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(F.silu(self.norm2(x)))
        return x + h


class AttentionBlock(nn.Module):
    """Single-head self-attention over the H*W spatial positions."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.norm = RMSNormCF(dim, images=True)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        b, c, h, w = x.shape
        qkv = self.to_qkv(self.norm(x))
        qkv = qkv.reshape(b, 1, c * 3, h * w).permute(0, 1, 3, 2)  # [B, 1, HW, 3C]
        q, k, v = qkv.chunk(3, dim=-1)
        out = F.scaled_dot_product_attention(q, k, v)  # [B, 1, HW, C]
        out = out.squeeze(1).permute(0, 2, 1).reshape(b, c, h, w)
        return self.proj(out) + identity


class MidBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.attentions = nn.ModuleList([AttentionBlock(dim)])
        self.resnets = nn.ModuleList([ResidualBlock(dim, dim), ResidualBlock(dim, dim)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnets[0](x)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            x = resnet(attn(x))
        return x


class Resample(nn.Module):
    """`QwenImage21Resample` in upsample mode, single frame: nearest-exact 2x then a 3x3 conv.

    `time_conv` exists in the checkpoint for `upsample3d` but is never applied to the first (only)
    frame, so it is registered purely so `load_state_dict` stays strict.
    """

    def __init__(self, dim: int, out_dim: int, temporal: bool) -> None:
        super().__init__()
        self.resample = nn.Sequential(
            nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"), nn.Conv2d(dim, out_dim, 3, padding=1)
        )
        if temporal:
            self.time_conv = nn.Conv2d(dim, dim * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x.float(), scale_factor=2.0, mode="nearest-exact").type_as(x)
        return self.resample[1](x)


class ResidualUpBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_res_blocks: int, temporal_upsample: bool, up_flag: bool) -> None:
        super().__init__()
        self.in_dim, self.out_dim, self.up_flag = in_dim, out_dim, up_flag
        self.factor_t = 2 if (up_flag and temporal_upsample) else 1
        blocks, cur = [], in_dim
        for _ in range(num_res_blocks + 1):
            blocks.append(ResidualBlock(cur, out_dim))
            cur = out_dim
        self.resnets = nn.ModuleList(blocks)
        if up_flag:
            self.upsampler = Resample(out_dim, out_dim, temporal=temporal_upsample)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        for resnet in self.resnets:
            x = resnet(x)
        if not self.up_flag:
            return x
        x = self.upsampler(x)
        return x + dup_up_fast(skip, self.in_dim, self.out_dim, self.factor_t)


class Decoder3d(nn.Module):
    def __init__(self, cfg: VAEConfig = VAE) -> None:
        super().__init__()
        dim, mult = cfg.decoder_base_dim, list(cfg.dim_mult)
        dims = [dim * u for u in [mult[-1]] + mult[::-1]]  # [1152, 1152, 1152, 576, 288, 144]
        self.dims = dims
        self.conv_in = nn.Conv2d(cfg.z_dim, dims[0], 3, padding=1)
        self.mid_block = MidBlock(dims[0])
        blocks = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            up_flag = i != len(mult) - 1
            blocks.append(
                ResidualUpBlock(
                    in_dim,
                    out_dim,
                    cfg.num_res_blocks,
                    temporal_upsample=cfg.temporal_upsample[i] if up_flag else False,
                    up_flag=up_flag,
                )
            )
        self.up_blocks = nn.ModuleList(blocks)
        self.norm_out = RMSNormCF(dims[-1], images=False)
        self.conv_out = nn.Conv2d(dims[-1], cfg.out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        x = self.mid_block(x)
        for up_block in self.up_blocks:
            x = up_block(x)
        return self.conv_out(F.silu(self.norm_out(x)))


class QwenImageVAEDecoderTorch(nn.Module):
    """`post_quant_conv` + decoder + the final clamp, i.e. `AutoencoderKLQwenImage21.decode`."""

    def __init__(self, cfg: VAEConfig = VAE) -> None:
        super().__init__()
        self.cfg = cfg
        self.post_quant_conv = nn.Conv2d(cfg.z_dim, cfg.z_dim, 1)
        self.decoder = Decoder3d(cfg)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """`z`: `[B, 64, H, W]` (or `[B, 64, 1, H, W]`) -> `[B, 4, 16H, 16W]` clamped to [-1, 1]."""
        if z.dim() == 5:
            assert z.shape[2] == 1, "single-frame decoder"
            z = z[:, :, 0]
        return torch.clamp(self.decoder(self.post_quant_conv(z)), -1.0, 1.0)


# ------------------------------------------------------------------------------------- weight load


def state_dict_from_ckpt(ckpt, dtype: torch.dtype = torch.float32) -> dict:
    """Pull the `post_quant_conv.*` / `decoder.*` tensors out of a `LazyCheckpoint`."""
    wanted = ("post_quant_conv.", "decoder.")
    return {k: ckpt.get(k, dtype) for k in ckpt.keys() if k.startswith(wanted)}


def load_decoder(ckpt=None, dtype: torch.dtype = torch.float32, cfg: VAEConfig = VAE) -> QwenImageVAEDecoderTorch:
    if ckpt is None:
        from ..common.weights import vae_ckpt

        ckpt = vae_ckpt()
    model = QwenImageVAEDecoderTorch(cfg).to(dtype)
    missing, unexpected = model.load_state_dict(state_dict_from_ckpt(ckpt, dtype), strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ------------------------------------------------------------------------------------------ pixels


def to_pil(out: torch.Tensor, background: Sequence[float] = (1.0, 1.0, 1.0)):
    """`[1, 4, H, W]` in [-1, 1] -> (RGBA image, RGB composited over `background`)."""
    from PIL import Image

    x = (out.detach().float()[0].permute(1, 2, 0) / 2 + 0.5).clamp(0, 1)  # [H, W, 4]
    rgba = Image.fromarray((x * 255).round().to(torch.uint8).numpy(), mode="RGBA")
    alpha = x[..., 3:4]
    bg = torch.tensor(background, dtype=x.dtype).reshape(1, 1, 3)
    rgb = x[..., :3] * alpha + bg * (1 - alpha)
    return rgba, Image.fromarray((rgb * 255).round().to(torch.uint8).numpy(), mode="RGB")


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation between two tensors, computed in fp64 over the flattened values."""
    a = a.detach().reshape(-1).to(torch.float64)
    b = b.detach().reshape(-1).to(torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom == 0:
        return 1.0 if torch.equal(a, b) else 0.0
    return float((a @ b) / denom)


def load_golden(path: Optional[str] = None):
    """Load `goldens/vae.pt` -> (vae_in `[1, 64, 1, 64, 64]`, vae_out `[1, 4, 1024, 1024]`), or None."""
    import os

    from ..common.config import GOLDENS_DIR

    path = path or os.path.join(GOLDENS_DIR, "vae.pt")
    if not os.path.isfile(path):
        return None
    d = torch.load(path, map_location="cpu")
    return d["vae_in"], d["vae_out"]
