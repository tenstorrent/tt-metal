# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image-2.1 VAE encoder on one Blackhole p150 with TTNN ops.

`AutoencoderKLQwenImage21` is a 3D VAE, but a still image is a single frame and every temporal
mechanism degenerates (see `reference/torch_vae_encoder.py` for the derivation): the causal 3D
convolutions are plain 2D convolutions, the `downsample3d` `time_conv` is skipped on the first
chunk, and `QwenImage21AvgDown3D` collapses to a 2x2 average pool with a fixed channel mapping. So
this is a 2D encoder:

    image [1, 4, 1024, 1024] -> conv_in -> 5 down blocks -> mid -> head -> quant_conv
                             -> [1, 128, 64, 64] = mean | logvar;  the pipeline uses mean

It shares its building blocks with the decoder (`tt/vae.py`): activations are channels-last bf16 in
DRAM carried as `[1, 1, H*W, C]` with `(h, w)` tracked alongside, channel counts are zero-padded to
a multiple of 32 in the convolution weights, and the RMS norms fold `sqrt(c / c_padded)` into gamma.
The encoder's own channel counts (96, 192, 384, 768, 128) are all multiples of 32, so only the
4-channel RGBA input is actually padded.

Two things the decoder never needed and this module adds:

  * `TTConvStrided`, a `TTConv2d` that also takes a stride and the 4-tuple
    `(top, bottom, left, right)` padding, for the 3x3 stride-2 downsamplers whose reference is
    `ZeroPad2d((0, 1, 0, 1))` followed by an unpadded convolution.
  * `avg_down_weight`, which expresses the parameter-free `AvgDown3D` shortcut as a fixed 2x2
    stride-2 convolution, so it runs as one `ttnn.conv2d` instead of a pool plus a channel gather.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import torch

import ttnn

from ..common.weights import LazyCheckpoint
from ..reference.torch_vae_encoder import ENC, EncoderConfig
from .vae import DRAM, TTConv2d, TTMidBlock, TTResBlock, TTRmsNorm, VAEPrecision, align32

__all__ = [
    "ENC",
    "EncoderConfig",
    "QwenImageVAEEncoder",
    "TTConvStrided",
    "TTResidualDownBlock",
    "VAEPrecision",
    "avg_down_weight",
]


# ------------------------------------------------------------------------------------ strided conv


class TTConvStrided(TTConv2d):
    """`TTConv2d` with a stride and (optionally asymmetric) padding.

    `padding` is `(pad_h, pad_w)` or `(top, bottom, left, right)`, exactly what `ttnn.conv2d` takes,
    so `nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(..., stride=2))` becomes one op with
    `padding=(0, 1, 0, 1)`. Weight preparation and the per-`(h, w)` device-weight cache are the
    parent's.
    """

    def __init__(
        self,
        dev,
        weight: torch.Tensor,
        bias: torch.Tensor,
        *,
        stride: Tuple[int, int],
        padding: Tuple[int, ...],
        prec: VAEPrecision,
        pad_out_channels: bool = True,
        output_layout: ttnn.Layout = ttnn.TILE_LAYOUT,
    ) -> None:
        super().__init__(
            dev,
            weight,
            bias,
            padding=0,
            prec=prec,
            pad_out_channels=pad_out_channels,
            output_layout=output_layout,
        )
        self.stride = tuple(stride)
        self.padding = tuple(padding)

    def __call__(self, x: ttnn.Tensor, h: int, w: int) -> Tuple[ttnn.Tensor, int, int]:
        x = ttnn.reshape(x, (1, h, w, self.in_c))
        n_slices = self.prec.slice_overrides.get((h, w, self.in_c, self.out_c))
        slice_config = (
            ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceWidth, num_slices=n_slices) if n_slices else None
        )
        wt, bt = self.prepared.get((h, w), (self.host_w, self.host_b))
        out, (oh, ow), (wt, bt) = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=wt,
            bias_tensor=bt,
            device=self.dev,
            in_channels=self.in_c,
            out_channels=self.out_c,
            batch_size=1,
            input_height=h,
            input_width=w,
            kernel_size=self.kernel,
            stride=self.stride,
            padding=self.padding,
            dilation=(1, 1),
            groups=1,
            dtype=self.prec.act_dtype,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            slice_config=slice_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        self.prepared[(h, w)] = (wt, bt)
        return ttnn.reshape(out, (1, 1, oh * ow, self.out_c)), oh, ow


# ------------------------------------------------------------------------------------ AvgDown3D


def avg_down_weight(in_c: int, out_c: int, factor_t: int) -> torch.Tensor:
    """The `AvgDown3D` shortcut of a down block as a 2x2 stride-2 convolution weight.

    `reference.torch_vae_encoder.avg_down_fast` shows the shortcut is a 2x2 average pool whose
    result lands in output channel `c` (`factor_t == 1`) or `2 * c + 1` (`factor_t == 2`, where the
    even channels average the zero-padded time slot and are therefore identically zero). Both are a
    convolution with `1 / 4` on the four taps of one input channel, so the whole shortcut is one
    `ttnn.conv2d` with this fixed weight and a zero bias.
    """
    w = torch.zeros(out_c, in_c, 2, 2, dtype=torch.float32)
    if factor_t == 1:
        assert in_c == out_c, (in_c, out_c)
        rows = torch.arange(in_c)
    else:
        assert factor_t == 2 and out_c == 2 * in_c, (in_c, out_c, factor_t)
        rows = 2 * torch.arange(in_c) + 1
    w[rows, torch.arange(in_c)] = 0.25
    return w


# ---------------------------------------------------------------------------------------- blocks


class TTResidualDownBlock:
    """`num_res_blocks` residual blocks, an optional 3x3 stride-2 downsampler, and the `AvgDown3D`
    shortcut of the block *input* added to the result.

    Unlike the decoder's up blocks this is `num_res_blocks` blocks, not `num_res_blocks + 1`, and
    the last block (`down_flag=False`) keeps its resolution and adds the input back unchanged.
    """

    def __init__(
        self,
        dev,
        ckpt: LazyCheckpoint,
        prefix: str,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        temporal: bool,
        down_flag: bool,
        prec: VAEPrecision,
    ) -> None:
        self.in_dim, self.out_dim, self.down_flag = in_dim, out_dim, down_flag
        self.factor_t = 2 if temporal else 1
        self.resnets = [TTResBlock(dev, ckpt, f"{prefix}resnets.{i}.", prec) for i in range(num_res_blocks)]
        if down_flag:
            self.down_conv = TTConvStrided(
                dev,
                ckpt.get(f"{prefix}downsampler.resample.1.weight", torch.float32),
                ckpt.get(f"{prefix}downsampler.resample.1.bias", torch.float32),
                stride=(2, 2),
                padding=(0, 1, 0, 1),  # nn.ZeroPad2d((left, right, top, bottom)) = (0, 1, 0, 1)
                prec=prec,
            )
            self.avg_conv = TTConvStrided(
                dev,
                avg_down_weight(in_dim, out_dim, self.factor_t),
                torch.zeros(out_dim),
                stride=(2, 2),
                padding=(0, 0),
                prec=prec,
            )
        else:
            # factor_t == factor_s == 1 and in_dim == out_dim: the shortcut is the identity.
            self.down_conv = None
            self.avg_conv = None

    def __call__(self, x: ttnn.Tensor, h: int, w: int) -> Tuple[ttnn.Tensor, int, int]:
        y, yh, yw = x, h, w
        for i, resnet in enumerate(self.resnets):
            # The first resnet must not free the block input: it is the AvgDown3D shortcut source.
            y, yh, yw = resnet(y, yh, yw, keep_input=(i == 0))
        if self.down_conv is not None:
            down, yh, yw = self.down_conv(y, yh, yw)
            ttnn.deallocate(y)
            y = down

        if self.avg_conv is None:
            out = ttnn.add(y, x)
            ttnn.deallocate(y)
            ttnn.deallocate(x)
            return out, yh, yw

        short, sh, sw = self.avg_conv(x, h, w)
        ttnn.deallocate(x)
        assert (sh, sw) == (yh, yw), ((sh, sw), (yh, yw))
        out = ttnn.add(y, short)
        ttnn.deallocate(y)
        ttnn.deallocate(short)
        return out, yh, yw


# ---------------------------------------------------------------------------------------- encoder


class QwenImageVAEEncoder:
    """The encoder + `quant_conv` half of `AutoencoderKLQwenImage21.encode`, single frame.

    `encode` is the host-in, host-out entry point, `encode_device` the device-only one, and
    `warmup(h, w)` materializes every prepared convolution weight for one resolution so the
    pipeline can allocate all device buffers before it captures any trace.
    """

    def __init__(
        self,
        device,
        ckpt: Optional[LazyCheckpoint] = None,
        prec: Optional[VAEPrecision] = None,
        cfg: EncoderConfig = ENC,
    ) -> None:
        if ckpt is None:
            from ..common.weights import vae_ckpt

            ckpt = vae_ckpt()
        self.dev = device
        self.cfg = cfg
        self.prec = prec or VAEPrecision()
        g = lambda k: ckpt.get(k, torch.float32)

        dims = cfg.dims  # [96, 96, 192, 384, 768, 768]
        self.dims = dims
        self.in_c_padded = align32(cfg.in_channels)

        self.conv_in = TTConv2d(
            device, g("encoder.conv_in.weight"), g("encoder.conv_in.bias"), padding=1, prec=self.prec
        )
        self.down_blocks: List[TTResidualDownBlock] = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            down_flag = i != len(cfg.dim_mult) - 1
            self.down_blocks.append(
                TTResidualDownBlock(
                    device,
                    ckpt,
                    f"encoder.down_blocks.{i}.",
                    in_dim,
                    out_dim,
                    cfg.num_res_blocks,
                    temporal=cfg.temperal_downsample[i] if down_flag else False,
                    down_flag=down_flag,
                    prec=self.prec,
                )
            )
        self.mid_block = TTMidBlock(device, ckpt, "encoder.mid_block.", dims[-1], self.prec)
        self.norm_out = TTRmsNorm(device, g("encoder.norm_out.gamma"), self.prec)
        self.conv_out = TTConv2d(
            device, g("encoder.conv_out.weight"), g("encoder.conv_out.bias"), padding=1, prec=self.prec
        )
        self.quant_conv = TTConv2d(device, g("quant_conv.weight"), g("quant_conv.bias"), padding=0, prec=self.prec)
        self._warm: Dict[Tuple[int, int], bool] = {}

    # ----------------------------------------------------------------------------- host <-> device
    def _to_nhwc(self, image: torch.Tensor) -> torch.Tensor:
        """`[1, 4, H, W]` / `[1, 4, 1, H, W]` -> `[1, 1, H*W, 32]` with the pad channels zeroed."""
        if image.dim() == 5:
            assert image.shape[2] == 1, "single-frame encoder"
            image = image[:, :, 0]
        assert image.dim() == 4, image.shape
        b, c, h, w = image.shape
        assert b == 1, "batch 1 only"
        assert c == self.cfg.in_channels, (c, self.cfg.in_channels)
        flat = torch.zeros(1, 1, h * w, self.in_c_padded, dtype=torch.float32)
        flat[0, 0, :, :c] = image.float()[0].permute(1, 2, 0).reshape(h * w, c)
        return flat

    def upload(self, image: torch.Tensor) -> Tuple[ttnn.Tensor, int, int]:
        h, w = int(image.shape[-2]), int(image.shape[-1])
        dev_x = ttnn.from_torch(
            self._to_nhwc(image).to(torch.bfloat16),
            dtype=self.prec.act_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.dev,
            memory_config=DRAM,
        )
        return dev_x, h, w

    def _download(self, out: ttnn.Tensor, h: int, w: int) -> torch.Tensor:
        """device `[1, 1, H*W, 128]` -> torch `[1, 128, H, W]` fp32."""
        t = ttnn.to_torch(out).float().reshape(1, h, w, -1)[..., : self.cfg.latent_channels]
        return t.permute(0, 3, 1, 2).contiguous()

    # -------------------------------------------------------------------------------------- device
    def forward(self, x: ttnn.Tensor, h: int, w: int) -> Tuple[ttnn.Tensor, int, int]:
        """Device-only encode: `[1, 1, H*W, 32]` TILE bf16 -> `[1, 1, H/16*W/16, 128]` TILE bf16.

        `x` is left allocated so a traced call can keep reusing the same persistent input buffer.
        """
        y, h, w = self.conv_in(x, h, w)
        for down_block in self.down_blocks:
            y, h, w = down_block(y, h, w)
        y, h, w = self.mid_block(y, h, w)
        n = self.norm_out(y)
        ttnn.deallocate(y)
        s = ttnn.silu(n)
        ttnn.deallocate(n)
        y, h, w = self.conv_out(s, h, w)
        ttnn.deallocate(s)
        out, h, w = self.quant_conv(y, h, w)
        ttnn.deallocate(y)
        return out, h, w

    def encode_device(self, x: ttnn.Tensor, h: int, w: int) -> Tuple[ttnn.Tensor, int, int]:
        """Alias of `forward`, for callers that keep the latent on device."""
        return self.forward(x, h, w)

    # -------------------------------------------------------------------------------------- warmup
    def warmup(self, h: int, w: int) -> None:
        """Run one encode at `(h, w)` so every `ttnn.conv2d` caches its prepared device weights.

        The pipeline calls this at init: buffers allocated after a trace capture get clobbered, so
        everything the encode touches has to exist beforehand.
        """
        if self._warm.get((h, w)):
            return
        x = ttnn.from_torch(
            torch.zeros(1, 1, h * w, self.in_c_padded, dtype=torch.bfloat16),
            dtype=self.prec.act_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.dev,
            memory_config=DRAM,
        )
        out, _, _ = self.forward(x, h, w)
        ttnn.deallocate(out)
        ttnn.deallocate(x)
        ttnn.synchronize_device(self.dev)
        self._warm[(h, w)] = True

    # ---------------------------------------------------------------------------------- host entry
    def encode(self, image: torch.Tensor, *, return_logvar: bool = False):
        """Encode one RGBA image in [-1, 1] to the posterior mode.

        `image` is a torch `[1, 4, H, W]` or `[1, 4, 1, H, W]` tensor. Returns the mode
        `[1, 64, H/16, W/16]` (fp32), or `(mode, logvar)` when `return_logvar` is set.
        """
        x, h, w = self.upload(image)
        out, oh, ow = self.forward(x, h, w)
        params = self._download(out, oh, ow)
        ttnn.deallocate(out)
        ttnn.deallocate(x)
        self._warm[(h, w)] = True
        mean, logvar = torch.chunk(params, 2, dim=1)
        if not return_logvar:
            return mean
        return mean, torch.clamp(logvar, -30.0, 20.0)

    # ---------------------------------------------------------------------------------- timing aid
    def time_encode(self, image: torch.Tensor, iters: int = 1) -> Tuple[torch.Tensor, float]:
        """Return (last mode, best wall-clock seconds) for `iters` encodes."""
        best = float("inf")
        out = None
        for _ in range(iters):
            ttnn.synchronize_device(self.dev)
            t0 = time.perf_counter()
            out = self.encode(image)
            ttnn.synchronize_device(self.dev)
            best = min(best, time.perf_counter() - t0)
        return out, best
