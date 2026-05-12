# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Torch reference of the Oobleck VAE decoder (Stable Audio).

The structure and parameter names are kept bit-for-bit aligned with
``diffusers.models.autoencoders.autoencoder_oobleck`` so the same checkpoint
state dict loads here. ``weight_norm`` is applied to convolutions to preserve
the ``weight_g``/``weight_v`` (or modern ``parametrizations``) keys in the
state dict; the TTNN port fuses these into a single weight tensor before
moving to device.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


@dataclass(frozen=True)
class OobleckDecoderConfig:
    """Configuration matching ``AutoencoderOobleck`` defaults for ACE-Step."""

    decoder_channels: int = 128
    decoder_input_channels: int = 64
    audio_channels: int = 2
    upsampling_ratios: tuple[int, ...] = (8, 8, 4, 4, 2)
    channel_multiples: tuple[int, ...] = (1, 2, 4, 8, 16)
    sampling_rate: int = 44100


class Snake1d(nn.Module):
    """Channel-wise Snake activation: ``x + (1/beta) * sin(alpha*x)**2``.

    With ``logscale=True`` (Oobleck default) the stored parameters are the
    log-domain values; runtime form is ``alpha = exp(alpha_log)`` and
    ``beta = exp(beta_log)``.
    """

    def __init__(self, hidden_dim: int, logscale: bool = True):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1, hidden_dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, hidden_dim, 1))
        self.logscale = logscale

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        alpha = self.alpha.exp() if self.logscale else self.alpha
        beta = self.beta.exp() if self.logscale else self.beta
        hidden_states = hidden_states.reshape(shape[0], shape[1], -1)
        hidden_states = hidden_states + (beta + 1e-9).reciprocal() * torch.sin(alpha * hidden_states).pow(2)
        return hidden_states.reshape(shape)


class OobleckResidualUnit(nn.Module):
    """``Snake -> Conv1d(k=7, dilated) -> Snake -> Conv1d(k=1)`` residual."""

    def __init__(self, dimension: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.snake1 = Snake1d(dimension)
        self.conv1 = weight_norm(nn.Conv1d(dimension, dimension, kernel_size=7, dilation=dilation, padding=pad))
        self.snake2 = Snake1d(dimension)
        self.conv2 = weight_norm(nn.Conv1d(dimension, dimension, kernel_size=1))

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        output_tensor = self.conv1(self.snake1(hidden_state))
        output_tensor = self.conv2(self.snake2(output_tensor))
        padding = (hidden_state.shape[-1] - output_tensor.shape[-1]) // 2
        if padding > 0:
            hidden_state = hidden_state[..., padding:-padding]
        return hidden_state + output_tensor


class OobleckDecoderBlock(nn.Module):
    """One upsampling stage: ``Snake -> ConvTranspose1d -> 3 x residual units``."""

    def __init__(self, input_dim: int, output_dim: int, stride: int = 1):
        super().__init__()
        self.snake1 = Snake1d(input_dim)
        self.conv_t1 = weight_norm(
            nn.ConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            )
        )
        self.res_unit1 = OobleckResidualUnit(output_dim, dilation=1)
        self.res_unit2 = OobleckResidualUnit(output_dim, dilation=3)
        self.res_unit3 = OobleckResidualUnit(output_dim, dilation=9)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.snake1(hidden_state)
        hidden_state = self.conv_t1(hidden_state)
        hidden_state = self.res_unit1(hidden_state)
        hidden_state = self.res_unit2(hidden_state)
        hidden_state = self.res_unit3(hidden_state)
        return hidden_state


class OobleckDecoder(nn.Module):
    """Full Oobleck decoder: ``conv1 -> N upsample blocks -> snake -> conv2``.

    Input is ``[batch, decoder_input_channels, latent_frames]`` and output is
    ``[batch, audio_channels, latent_frames * prod(upsampling_ratios)]``.
    """

    def __init__(
        self,
        *,
        channels: int,
        input_channels: int,
        audio_channels: int,
        upsampling_ratios,
        channel_multiples,
    ):
        super().__init__()
        strides = list(upsampling_ratios)
        cm = [1] + list(channel_multiples)

        self.conv1 = weight_norm(nn.Conv1d(input_channels, channels * cm[-1], kernel_size=7, padding=3))

        block: list[nn.Module] = []
        for stride_index, stride in enumerate(strides):
            block.append(
                OobleckDecoderBlock(
                    input_dim=channels * cm[len(strides) - stride_index],
                    output_dim=channels * cm[len(strides) - stride_index - 1],
                    stride=stride,
                )
            )
        self.block = nn.ModuleList(block)

        self.snake1 = Snake1d(channels)
        self.conv2 = weight_norm(nn.Conv1d(channels, audio_channels, kernel_size=7, padding=3, bias=False))

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.conv1(hidden_state)
        for layer in self.block:
            hidden_state = layer(hidden_state)
        hidden_state = self.snake1(hidden_state)
        hidden_state = self.conv2(hidden_state)
        return hidden_state


def build_decoder_from_config(cfg: OobleckDecoderConfig) -> OobleckDecoder:
    """Construct an ``OobleckDecoder`` from a ``OobleckDecoderConfig``."""
    return OobleckDecoder(
        channels=cfg.decoder_channels,
        input_channels=cfg.decoder_input_channels,
        audio_channels=cfg.audio_channels,
        upsampling_ratios=cfg.upsampling_ratios,
        channel_multiples=cfg.channel_multiples,
    )
