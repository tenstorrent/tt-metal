# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of the full ``OobleckDecoder``."""

from __future__ import annotations

import os

import ttnn
from ..math_perf_env import (
    ace_step_flush_device_profiler,
    ace_step_profiler_flush_every_layer,
    ace_step_vae_activation_storage_dtype,
    ace_step_vae_ensure_interleaved,
    ace_step_vae_host_weight_staging_dtype,
)
from .block import TtOobleckDecoderBlock, _strip_prefix
from .conv1d import TtConv1d
from .snake import TtSnake1d
from .weight_utils import fused_oobleck_decoder_weights


def _vae_trace_enabled() -> bool:
    # conv2d_L1 internally calls ttnn::prim::move (kernel-binary write) inside the trace-capture
    # window, which is forbidden.  The second call triggers a program-cache miss because the
    # in_buf clone has a different device address, making the L1-shard move program hash differ
    # from warmup.  Disable until the VAE conv path is made trace-compatible.
    return os.environ.get("ACE_STEP_VAE_TRACE", "0") not in ("0", "false", "False")


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
        self.ttnn = ttnn
        self.device = device
        self.channels = int(channels)
        self.input_channels = int(input_channels)
        self.audio_channels = int(audio_channels)
        # Stereo head (audio_channels=2) is not TP-4 column-shardable; VAE weights stay
        # device-local / replicated — never use ShardTensorToMesh on out_channels here.
        self.upsampling_ratios = list(upsampling_ratios)
        cm = [1] + list(channel_multiples)
        assert (
            len(cm) == len(upsampling_ratios) + 1
        ), f"channel_multiples length {len(channel_multiples)} must equal len(upsampling_ratios)={len(upsampling_ratios)}"
        self.activation_dtype = activation_dtype or ace_step_vae_activation_storage_dtype(ttnn)
        self.weights_dtype = weights_dtype or ace_step_vae_host_weight_staging_dtype(ttnn)

        from models.experimental.ace_step_v1_5.utils.tt_device import (
            ace_step_device_num_chips,
            ace_step_synchronize_device,
        )

        weights = fused_oobleck_decoder_weights(
            state_dict, upsampling_ratios=self.upsampling_ratios, decoder_prefix=decoder_prefix
        )

        print("[ace_step_v1_5] VAE: loading conv_in …", flush=True)
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
        num_blocks = len(self.upsampling_ratios)
        for i, stride in enumerate(self.upsampling_ratios):
            print(f"[ace_step_v1_5] VAE: upsample block {i + 1}/{num_blocks} …", flush=True)
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
        if ace_step_device_num_chips(device) > 1:
            ace_step_synchronize_device(ttnn, device)
        print("[ace_step_v1_5] VAE: decoder ready", flush=True)
        ace_step_flush_device_profiler(device)

        self._profiler_flush_every = ace_step_profiler_flush_every_layer()
        self._profiler_layer_idx = 0

        # Per-shape trace cache: (B, T, C) -> (trace_id, in_buf, out_buf).
        # Populated on the 2nd call with a given shape (1st call is warmup so programs compile
        # before capture); replayed on the 3rd+ call.  ``deallocate_activation`` is only triggered
        # for L1 tensors (not DRAM), so the DRAM ``in_buf`` survives the trace intact.
        self._trace_cache: dict = {}
        self._shape_warmup: set = set()
        self._trace_api: bool | None = None  # lazily checked

    def _has_trace_api(self) -> bool:
        if self._trace_api is None:
            ttnn = self.ttnn
            self._trace_api = all(
                hasattr(ttnn, name)
                for name in (
                    "begin_trace_capture",
                    "end_trace_capture",
                    "execute_trace",
                    "clone",
                    "copy",
                    "record_event",
                    "wait_for_event",
                )
            )
        return self._trace_api  # type: ignore[return-value]

    def _maybe_flush_device_profiler(self) -> None:
        every = int(getattr(self, "_profiler_flush_every", 0) or 0)
        if every <= 0:
            return
        self._profiler_layer_idx = int(getattr(self, "_profiler_layer_idx", 0)) + 1
        if self._profiler_layer_idx % every == 0:
            ace_step_flush_device_profiler(self.device)

    def _forward(self, x):
        """Core decode computation without normalization; used for both eager and traced paths."""
        ttnn = self.ttnn
        dram_mc = ttnn.DRAM_MEMORY_CONFIG
        x = self.conv1(x)
        x = ace_step_vae_ensure_interleaved(ttnn, x, memory_config=dram_mc)
        self._maybe_flush_device_profiler()
        for block in self.blocks:
            x = block(x)
            x = ace_step_vae_ensure_interleaved(ttnn, x, memory_config=dram_mc)
            self._maybe_flush_device_profiler()
        x = self.snake1(x)
        self._maybe_flush_device_profiler()
        x = self.conv2(x)
        self._maybe_flush_device_profiler()
        return x

    def __call__(self, x):
        """Decode latents to raw audio. Input: ``[B, T, C_in]`` row-major; output: ``[B, T_out, C_audio]``."""
        ttnn = self.ttnn
        if len(x.shape) != 3:
            raise ValueError(f"TtOobleckDecoder expects rank-3 [B,T,C], got {x.shape}")
        # Ensure ROW_MAJOR layout: ttnn.slice on TILE tensors requires 32-aligned boundaries,
        # which chunk windows never guarantee, so we must be row-major before any indexing.
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        # Ensure storage dtype (BF16 ROW_MAJOR): conv uses compute dtype internally when opt-in env is set.
        # Diffusion latents arrive as float32.
        if x.dtype != self.activation_dtype:
            x = ttnn.typecast(x, self.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        if not (_vae_trace_enabled() and self._has_trace_api()):
            return self._forward(x)

        key = (int(x.shape[0]), int(x.shape[1]), int(x.shape[2]))

        cached = self._trace_cache.get(key)
        if cached is not None:
            # Replay: copy new latent data into the persistent input buffer, then replay trace.
            tid, in_buf, out_buf = cached
            ttnn.copy(x, in_buf)
            write_event = ttnn.record_event(self.device, 1)
            ttnn.wait_for_event(0, write_event)
            ttnn.execute_trace(self.device, tid, cq_id=0, blocking=True)
            return out_buf

        if key not in self._shape_warmup:
            # First call with this shape: run eagerly so all programs compile before capture.
            self._shape_warmup.add(key)
            return self._forward(x)

        # Second call with this shape: capture trace against a persistent input buffer.
        in_buf = ttnn.clone(x, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=self.activation_dtype)
        tid = ttnn.begin_trace_capture(self.device, cq_id=0)
        out_buf = self._forward(in_buf)
        ttnn.end_trace_capture(self.device, tid, cq_id=0)
        self._trace_cache[key] = (tid, in_buf, out_buf)
        return out_buf

    def forward(self, x):
        return self(x)
