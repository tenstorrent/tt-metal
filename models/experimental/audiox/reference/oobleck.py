# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""AudioX Oobleck VAE blocks (reference).

Mirrors ``audiox/models/autoencoders.py`` for the inference path AudioX
exercises. The encoder downsamples raw stereo audio into latent features,
and the decoder upsamples the latent ``[B, 64, T]`` produced by the DiT
back into the audio waveform ``[B, 2, T*2048]``.

Upstream uses ``WNConv1d``/``WNConvTranspose1d`` from the ``dac`` library;
both are simply ``torch.nn.utils.weight_norm`` wrappers around standard
Conv1d/ConvTranspose1d. We use the in-tree ``weight_norm`` so the
parameter names match upstream's checkpoint (``weight_g`` / ``weight_v``)
and the pretrained loader can drop straight in.

AudioX's HF config disables ``antialias_activation`` and ``final_tanh``
and sets ``use_snake=True`` on every layer, so this module supports only
that path; the alias-free wrapper and the tanh tail are not ported."""

import math

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class SnakeBeta(nn.Module):
    """Periodic activation ``x + (1/beta) * sin(alpha*x)^2`` with per-channel
    learned ``alpha``/``beta`` stored in log-space (matching upstream
    ``alpha_logscale=True``). Init to zero so the activation starts as
    identity-ish + small periodic perturbation after exp."""

    def __init__(self, in_features: int, eps: float = 1e-9):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(in_features))
        self.beta = nn.Parameter(torch.zeros(in_features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] — broadcast alpha/beta over batch and time.
        alpha = torch.exp(self.alpha).unsqueeze(0).unsqueeze(-1)
        beta = torch.exp(self.beta).unsqueeze(0).unsqueeze(-1)
        return x + (1.0 / (beta + self.eps)) * torch.sin(x * alpha).pow(2)


class ResidualUnit(nn.Module):
    """Snake -> dilated 7-tap conv -> Snake -> 1-tap conv, plus residual.
    Output channels equal input channels in the upstream config (always
    called as in==out)."""

    def __init__(self, channels: int, dilation: int):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.act1 = SnakeBeta(channels)
        self.conv1 = weight_norm(nn.Conv1d(channels, channels, kernel_size=7, dilation=dilation, padding=padding))
        self.act2 = SnakeBeta(channels)
        self.conv2 = weight_norm(nn.Conv1d(channels, channels, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(x))
        h = self.conv2(self.act2(h))
        return x + h


class DecoderBlock(nn.Module):
    """Snake -> WNConvTranspose1d (the upsample) -> 3 dilated residual
    units (dilations 1, 3, 9). Upstream packs these in ``nn.Sequential``;
    we keep them as named attributes so the TT port can address each
    op directly."""

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.act = SnakeBeta(in_channels)
        # ConvTranspose1d with kernel=2*stride/padding=ceil(stride/2) is
        # the upstream choice; gives a clean factor-of-stride upsample.
        self.upsample = weight_norm(
            nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            )
        )
        self.res1 = ResidualUnit(out_channels, dilation=1)
        self.res2 = ResidualUnit(out_channels, dilation=3)
        self.res3 = ResidualUnit(out_channels, dilation=9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(self.act(x))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.res1 = ResidualUnit(in_channels, dilation=1)
        self.res2 = ResidualUnit(in_channels, dilation=3)
        self.res3 = ResidualUnit(in_channels, dilation=9)
        self.act = SnakeBeta(in_channels)
        self.downsample = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        return self.downsample(self.act(x))


class OobleckEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        channels: int = 128,
        latent_dim: int = 64,
        c_mults=(1, 2, 4, 8, 16),
        strides=(2, 4, 4, 8, 8),
    ):
        super().__init__()
        c_mults = (1,) + tuple(c_mults)
        depth = len(c_mults)

        self.encoded_channels = latent_dim
        self.in_conv = weight_norm(nn.Conv1d(in_channels, c_mults[0] * channels, kernel_size=7, padding=3))
        self.blocks = nn.ModuleList(
            [
                EncoderBlock(
                    in_channels=c_mults[i] * channels,
                    out_channels=c_mults[i + 1] * channels,
                    stride=strides[i],
                )
                for i in range(depth - 1)
            ]
        )
        self.out_act = SnakeBeta(c_mults[-1] * channels)
        self.out_conv = weight_norm(nn.Conv1d(c_mults[-1] * channels, latent_dim, kernel_size=3, padding=1))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(x)
        for block in self.blocks:
            x = block(x)
        return self.out_conv(self.out_act(x))


class OobleckDecoder(nn.Module):
    """Five-stage upsampler. AudioX HF config:
    ``c_mults=[1,2,4,8,16]``, ``strides=[2,4,4,8,8]``, ``channels=128``,
    ``latent_dim=64``, ``out_channels=2``. Total upsampling factor =
    ``prod(strides) = 2048``, so a 237-frame latent decodes to ~485k
    samples (~10s of stereo audio at 44.1 kHz)."""

    def __init__(
        self,
        out_channels: int = 2,
        channels: int = 128,
        latent_dim: int = 64,
        c_mults=(1, 2, 4, 8, 16),
        strides=(2, 4, 4, 8, 8),
    ):
        super().__init__()
        # Upstream prepends a 1x to c_mults so the final block lands on
        # `channels` width; track the depth here so naming follows upstream.
        c_mults = (1,) + tuple(c_mults)
        depth = len(c_mults)

        self.in_conv = weight_norm(nn.Conv1d(latent_dim, c_mults[-1] * channels, kernel_size=7, padding=3))

        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    in_channels=c_mults[i] * channels,
                    out_channels=c_mults[i - 1] * channels,
                    stride=strides[i - 1],
                )
                for i in range(depth - 1, 0, -1)
            ]
        )

        self.out_act = SnakeBeta(c_mults[0] * channels)
        self.out_conv = weight_norm(
            nn.Conv1d(c_mults[0] * channels, out_channels, kernel_size=7, padding=3, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(x)
        for block in self.blocks:
            x = block(x)
        return self.out_conv(self.out_act(x))
