# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ACE-Step v1.5 VAE decoder (Oobleck / stable-audio 1D autoencoder) on TTNN.

Turns the DiT's clean audio latents `[B, 64, T]` into a 48 kHz stereo waveform
`[B, 2, T*1920]` — the final stage that makes generated music audible so SongEval can score it.

Reference: diffusers `AutoencoderOobleck.decoder` (autoencoder_oobleck.py). Structure:

    conv1 : Conv1d(64 -> 2048, k7, pad3)                              (weight_norm)
    block[i] for i in 0..4 (upsampling_ratios [2,4,4,6,10] reversed): OobleckDecoderBlock
        snake1  : Snake1d
        conv_t1 : ConvTranspose1d(in -> out, k=2s, stride=s, pad=ceil(s/2))   (weight_norm)
        res_unit{1,2,3} : OobleckResidualUnit(out, dilation in {1,3,9})
            snake1 -> conv1(k7, dilation d, "same") -> snake2 -> conv2(k1) -> +center-cropped residual
    snake1 : Snake1d(128)
    conv2  : Conv1d(128 -> 2, k7, pad3, no bias)                      (weight_norm)

REUSE: this is built entirely from validated TTTv2 audio primitives
(`models/tt_dit/layers/audio_ops`): `Conv1dViaConv3d`, `ConvTranspose1dViaConv3d`, `SnakeBeta`
(with `alpha_logscale=True` to match the diffusers `logscale=True` Snake1d). Each primitive was
PCC-verified against the genuine Oobleck weights (>=0.9999). We only compose them here; no new
device kernels. weight_norm is already folded into the reference `.weight` by diffusers.

Data flows in `[B, T, C]` ROW_MAJOR (the layout the TTTv2 audio ops expect); callers transpose
from/to the reference `[B, C, T]` at the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass


import ttnn
from models.tt_dit.layers.audio_ops import Conv1dViaConv3d, ConvTranspose1dViaConv3d, SnakeBeta
from models.tt_dit.layers.module import Module, ModuleList


def _row_major(x: ttnn.Tensor) -> ttnn.Tensor:
    """SnakeBeta upcasts to TILE internally; conv ops require ROW_MAJOR input. Pull back."""
    if x.layout != ttnn.ROW_MAJOR_LAYOUT:
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    return x


@dataclass
class OobleckVAEConfig:
    """Oobleck decoder dims (diffusers AutoencoderOobleck config.json)."""

    decoder_input_channels: int = 64
    decoder_channels: int = 128
    audio_channels: int = 2
    channel_multiples: tuple[int, ...] = (1, 2, 4, 8, 16)
    # Decoder upsampling strides = downsampling_ratios reversed (diffusers: [2,4,4,6,10] -> [10,6,4,4,2]).
    upsampling_ratios: tuple[int, ...] = (10, 6, 4, 4, 2)
    dtype: ttnn.DataType = ttnn.float32  # audio decode wants fp32 (matches vocoder_ltx)

    @classmethod
    def from_diffusers(cls, vae_config) -> "OobleckVAEConfig":
        g = vae_config.get if hasattr(vae_config, "get") else (lambda k, d: getattr(vae_config, k, d))
        # The decoder upsamples by the REVERSED downsampling_ratios (diffusers builds it that way).
        down = list(g("downsampling_ratios", (2, 4, 4, 6, 10)))
        return cls(
            decoder_input_channels=g("decoder_input_channels", 64),
            decoder_channels=g("decoder_channels", 128),
            audio_channels=g("audio_channels", 2),
            channel_multiples=tuple(g("channel_multiples", (1, 2, 4, 8, 16))),
            upsampling_ratios=tuple(reversed(down)),
        )


class OobleckResidualUnit(Module):
    """snake1 -> conv1(k7, dilation) -> snake2 -> conv2(k1) -> center-cropped residual."""

    def __init__(self, dim: int, dilation: int, *, mesh_device, dtype=ttnn.float32):
        super().__init__()
        self.snake1 = SnakeBeta(dim, alpha_logscale=True, mesh_device=mesh_device, dtype=dtype)
        self.conv1 = Conv1dViaConv3d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=7,
            dilation=dilation,
            padding_mode="zeros",
            bias=True,
            mesh_device=mesh_device,
            dtype=dtype,
        )
        self.snake2 = SnakeBeta(dim, alpha_logscale=True, mesh_device=mesh_device, dtype=dtype)
        self.conv2 = Conv1dViaConv3d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=1,
            padding_mode="zeros",
            bias=True,
            mesh_device=mesh_device,
            dtype=dtype,
        )

    def forward(self, x_BTC: ttnn.Tensor) -> ttnn.Tensor:
        # "same" padding keeps T constant, so the residual add is a plain elementwise add.
        h = _row_major(self.snake1(x_BTC))
        h = self.conv1(h)
        h = _row_major(self.snake2(h))
        h = self.conv2(h)
        return ttnn.add(x_BTC, h)


class OobleckDecoderBlock(Module):
    """snake1 -> conv_t1 (upsample) -> res_unit1/2/3 (dilations 1,3,9)."""

    def __init__(self, input_dim: int, output_dim: int, stride: int, *, mesh_device, dtype=ttnn.float32):
        super().__init__()
        self.snake1 = SnakeBeta(input_dim, alpha_logscale=True, mesh_device=mesh_device, dtype=dtype)
        self.conv_t1 = ConvTranspose1dViaConv3d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=2 * stride,
            stride=stride,
            bias=True,
            mesh_device=mesh_device,
            dtype=dtype,
        )
        self.res_unit1 = OobleckResidualUnit(output_dim, 1, mesh_device=mesh_device, dtype=dtype)
        self.res_unit2 = OobleckResidualUnit(output_dim, 3, mesh_device=mesh_device, dtype=dtype)
        self.res_unit3 = OobleckResidualUnit(output_dim, 9, mesh_device=mesh_device, dtype=dtype)

    def forward(self, x_BTC: ttnn.Tensor) -> ttnn.Tensor:
        h = _row_major(self.snake1(x_BTC))
        h = self.conv_t1(h)
        h = self.res_unit1(h)
        h = self.res_unit2(h)
        h = self.res_unit3(h)
        return h


class OobleckDecoder(Module):
    """Full Oobleck VAE decoder: latents [B,T,64] -> waveform [B,T*1920,2] (BTC layout)."""

    def __init__(self, config: OobleckVAEConfig, *, mesh_device, dtype=None):
        super().__init__()
        dtype = dtype or config.dtype
        self.config = config
        self.mesh_device = mesh_device
        self.dtype = dtype

        ch = config.decoder_channels
        mults = [1] + list(config.channel_multiples)
        strides = list(config.upsampling_ratios)

        self.conv1 = Conv1dViaConv3d(
            in_channels=config.decoder_input_channels,
            out_channels=ch * mults[-1],
            kernel_size=7,
            padding_mode="zeros",
            bias=True,
            mesh_device=mesh_device,
            dtype=dtype,
        )
        self.block = ModuleList()
        for i, stride in enumerate(strides):
            self.block.append(
                OobleckDecoderBlock(
                    input_dim=ch * mults[len(strides) - i],
                    output_dim=ch * mults[len(strides) - i - 1],
                    stride=stride,
                    mesh_device=mesh_device,
                    dtype=dtype,
                )
            )
        self.snake1 = SnakeBeta(ch, alpha_logscale=True, mesh_device=mesh_device, dtype=dtype)
        self.conv2 = Conv1dViaConv3d(
            in_channels=ch,
            out_channels=config.audio_channels,
            kernel_size=7,
            padding_mode="zeros",
            bias=False,
            mesh_device=mesh_device,
            dtype=dtype,
        )

    def forward(self, latents_BTC: ttnn.Tensor) -> ttnn.Tensor:
        h = self.conv1(latents_BTC)
        for blk in self.block:
            h = blk(h)
        h = _row_major(self.snake1(h))
        h = self.conv2(h)
        return h
