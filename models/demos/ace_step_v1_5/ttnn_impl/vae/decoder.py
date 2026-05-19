# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of the full ``OobleckDecoder``."""

from __future__ import annotations

from .._ttnn import get_ttnn
from .block import TtOobleckDecoderBlock, _strip_prefix
from .conv1d import TtConv1d
from .snake import TtSnake1d
from .weight_utils import fused_oobleck_decoder_weights


def _require_ttnn():
    ttnn = get_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is required for ace_step_v1_5.ttnn_impl.vae")
    return ttnn


class TtOobleckDecoder:
    """TTNN port of the Stable-Audio Oobleck VAE decoder.

    Input ``[B, T_latent, decoder_input_channels]`` (row-major) ->
    output ``[B, T_audio, audio_channels]`` (row-major), where
    ``T_audio = T_latent * prod(upsampling_ratios)``.
    """

    def __init__(
        self,
        *,
        state_dict: dict,
        device,
        decoder_prefix: str = "decoder.",
        channels: int = 128,
        input_channels: int = 64,
        audio_channels: int = 2,
        upsampling_ratios=(8, 8, 4, 4, 2),
        channel_multiples=(1, 2, 4, 8, 16),
        activation_dtype=None,
        weights_dtype=None,
    ) -> None:
        ttnn = _require_ttnn()
        self.ttnn = ttnn
        self.device = device
        self.channels = int(channels)
        self.input_channels = int(input_channels)
        self.audio_channels = int(audio_channels)
        self.upsampling_ratios = list(upsampling_ratios)
        cm = [1] + list(channel_multiples)
        assert (
            len(cm) == len(upsampling_ratios) + 1
        ), f"channel_multiples length {len(channel_multiples)} must equal len(upsampling_ratios)={len(upsampling_ratios)}"
        self.activation_dtype = activation_dtype or getattr(ttnn, "bfloat16", None)
        self.weights_dtype = weights_dtype or getattr(ttnn, "bfloat16", None)
        if self.activation_dtype is None or self.weights_dtype is None:
            raise RuntimeError("TTNN build missing bfloat16; supply activation_dtype/weights_dtype")

        weights = fused_oobleck_decoder_weights(
            state_dict, upsampling_ratios=self.upsampling_ratios, decoder_prefix=decoder_prefix
        )

        self.conv1 = TtConv1d(
            weight_host=weights["conv1.weight"],
            bias_host=weights.get("conv1.bias"),
            in_channels=self.input_channels,
            out_channels=self.channels * cm[-1],
            kernel_size=7,
            stride=1,
            padding=3,
            dilation=1,
            device=device,
            activation_dtype=self.activation_dtype,
            weights_dtype=self.weights_dtype,
        )

        self.blocks = []
        for i, stride in enumerate(self.upsampling_ratios):
            in_dim = self.channels * cm[len(self.upsampling_ratios) - i]
            out_dim = self.channels * cm[len(self.upsampling_ratios) - i - 1]
            block_weights = _strip_prefix(weights, f"block.{i}.")
            self.blocks.append(
                TtOobleckDecoderBlock(
                    weights=block_weights,
                    input_dim=in_dim,
                    output_dim=out_dim,
                    stride=stride,
                    device=device,
                    activation_dtype=self.activation_dtype,
                    weights_dtype=self.weights_dtype,
                )
            )

        self.snake1 = TtSnake1d(
            alpha_host=weights["snake1.alpha"],
            beta_host=weights["snake1.beta"],
            device=device,
            dtype=self.activation_dtype,
        )
        self.conv2 = TtConv1d(
            weight_host=weights["conv2.weight"],
            bias_host=weights.get("conv2.bias"),
            in_channels=self.channels,
            out_channels=self.audio_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            dilation=1,
            device=device,
            activation_dtype=self.activation_dtype,
            weights_dtype=self.weights_dtype,
        )

    def __call__(self, x):
        """Decode latents to raw audio. Input: ``[B, T, C_in]`` row-major; output: ``[B, T_out, C_audio]``."""
        ttnn = self.ttnn
        if len(x.shape) != 3:
            raise ValueError(f"TtOobleckDecoder expects rank-3 [B,T,C], got {x.shape}")
        # Ensure ROW_MAJOR layout: ttnn.slice on TILE tensors requires 32-aligned boundaries,
        # which chunk windows never guarantee, so we must be row-major before any indexing.
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        # Ensure bfloat16: conv weights were packed with input_dtype=activation_dtype (bfloat16).
        # Diffusion latents arrive as float32; typecast to bfloat16 for correct conv computation.
        if x.dtype != self.activation_dtype:
            x = ttnn.typecast(x, self.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x = self.conv1(x)
        for block in self.blocks:
            x = block(x)
        x = self.snake1(x)
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = self.conv2(x)
        return x

    def forward(self, x):
        return self(x)
