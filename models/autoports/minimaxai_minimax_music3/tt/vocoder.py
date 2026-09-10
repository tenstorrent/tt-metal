# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-Music3's Flow-VAE vocoder (``MiniMaxMusic3Vocoder``) on one Blackhole chip (stage 07).

Reference: ``reference/vocoder_ref.py`` (diffusers ``minimax_music3_vocoder.py``, ``weight_norm`` folded at load).
Modelled on ``models/tt_dit/models/audio_vae/vocoder_ltx.py``: every 1D conv is ``models.tt_dit.layers.audio_ops``
``Conv1dViaConv3d`` (``ttnn.experimental.conv3d`` with a ``(k, 1, 1)`` kernel over a ``(B, T, 1, 1, C)`` row-major
tensor), the upsamplers are ``ConvTranspose1dViaConv3d`` (zero-stuffing + flipped-kernel conv; the reference's
``padding = ceil(stride / 2)`` equals the class's ``(k - stride) // 2`` for the even strides 8 / 8 / 4 / 2) and the
activations are ``Snake``. The structure mirrors the torch module 1:1 so the folded state dict loads by name.

Layout: the two audio channels are the two folded 64-channel latent streams, so the batch is ``2 * B`` streams of
``(T, C)`` rows; latents ``[B, 128, L]`` -> ``(2B, L, 64)`` on device -> ... -> ``(2B, 512 L, 1)`` -> ``[B, 2, 512 L]``.
Channel counts: 64 -> 1024 -> 1536 -> 768 -> 384 -> 192 -> 96 -> 1 (``conv_out`` pads its single output channel to
32 and slices it back). Dtype: fp32 by default (the LTX port found bf16 accumulation degrades through a long conv
chain); ``dtype=ttnn.bfloat16`` is the faster candidate measured in ``doc/optimize/README.md``.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch

import ttnn
from models.autoports.minimaxai_minimax_music3.reference import vocoder_ref as V
from models.tt_dit.layers.audio_ops import Conv1dViaConv3d, ConvTranspose1dViaConv3d, Snake, _AlignedOutConv1d
from models.tt_dit.layers.module import Module, ModuleList


def _snake(act: Snake, x: ttnn.Tensor) -> ttnn.Tensor:
    """``Snake`` upcasts to TILE layout internally (its alpha is a TILE parameter); the convs want ROW_MAJOR."""
    y = act(x)
    if y.layout != ttnn.ROW_MAJOR_LAYOUT:
        y2 = ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(y)
        y = y2
    return y


class _ResidualUnit(Module):
    def __init__(self, dim: int, dilation: int, *, mesh_device, dtype):
        super().__init__()
        self.snake1 = Snake(dim, mesh_device=mesh_device, dtype=dtype)
        self.conv1 = Conv1dViaConv3d(
            dim, dim, kernel_size=7, dilation=dilation, padding_mode="zeros", mesh_device=mesh_device, dtype=dtype
        )
        self.snake2 = Snake(dim, mesh_device=mesh_device, dtype=dtype)
        self.conv2 = Conv1dViaConv3d(
            dim, dim, kernel_size=1, padding_mode="zeros", mesh_device=mesh_device, dtype=dtype
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        h = _snake(self.snake1, x)
        h2 = self.conv1(h)
        ttnn.deallocate(h)
        h = _snake(self.snake2, h2)
        ttnn.deallocate(h2)
        h2 = self.conv2(h)
        ttnn.deallocate(h)
        out = ttnn.add(x, h2)
        ttnn.deallocate(h2)
        return out


class _Block(Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int, *, mesh_device, dtype):
        super().__init__()
        self.snake1 = Snake(input_dim, mesh_device=mesh_device, dtype=dtype)
        self.conv_t1 = ConvTranspose1dViaConv3d(
            input_dim, output_dim, kernel_size=2 * stride, stride=stride, mesh_device=mesh_device, dtype=dtype
        )
        assert self.conv_t1.external_pad_each == 2 * stride - 1 - stride // 2
        # torch: ConvTranspose1d(padding=ceil(stride / 2)); the class assumes (k - stride) // 2 = stride // 2.
        assert math.ceil(stride / 2) == stride // 2, stride
        self.res_unit1 = _ResidualUnit(output_dim, 1, mesh_device=mesh_device, dtype=dtype)
        self.res_unit2 = _ResidualUnit(output_dim, 3, mesh_device=mesh_device, dtype=dtype)
        self.res_unit3 = _ResidualUnit(output_dim, 9, mesh_device=mesh_device, dtype=dtype)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        h = _snake(self.snake1, x)
        up = self.conv_t1(h)
        ttnn.deallocate(h)
        for unit in (self.res_unit1, self.res_unit2, self.res_unit3):
            nxt = unit(up)
            ttnn.deallocate(up)
            up = nxt
        return up


class TTVocoder(Module):
    """``MiniMaxMusic3Vocoder`` on device; call with ``[B, 128, L]`` fp32 latents -> ``[B, 2, 512 L]`` fp32 host."""

    def __init__(
        self,
        mesh_device,
        *,
        latent_channels: int = 128,
        decoder_input_dim: int = 1024,
        decoder_hidden_dim: int = 1536,
        upsampling_ratios: Sequence[int] = (8, 8, 4, 2),
        sampling_rate: int = V.SAMPLING_RATE,
        dtype=ttnn.float32,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.latent_channels = latent_channels
        self.sampling_rate = sampling_rate
        self.hop_length = int(math.prod(upsampling_ratios))
        self.dec_in_proj = Conv1dViaConv3d(
            latent_channels // 2,
            decoder_input_dim,
            kernel_size=1,
            padding_mode="zeros",
            mesh_device=mesh_device,
            dtype=dtype,
        )
        self.conv_in = Conv1dViaConv3d(
            decoder_input_dim,
            decoder_hidden_dim,
            kernel_size=7,
            padding_mode="zeros",
            mesh_device=mesh_device,
            dtype=dtype,
        )
        blocks = []
        output_dim = decoder_hidden_dim
        for index, stride in enumerate(upsampling_ratios):
            input_dim = decoder_hidden_dim // (2**index)
            output_dim = decoder_hidden_dim // (2 ** (index + 1))
            blocks.append(_Block(input_dim, output_dim, stride, mesh_device=mesh_device, dtype=dtype))
        self.blocks = ModuleList(blocks)
        self.snake_out = Snake(output_dim, mesh_device=mesh_device, dtype=dtype)
        self.conv_out = _AlignedOutConv1d(
            output_dim, 1, kernel_size=7, padding_mode="zeros", mesh_device=mesh_device, dtype=dtype
        )
        self.timings: Dict[str, float] = {}

    @classmethod
    def from_pretrained(cls, mesh_device, weights_dir: Optional[Path] = None, *, dtype=ttnn.float32) -> "TTVocoder":
        root = V.weights_root(weights_dir)
        cfg = json.loads((root / "vocoder" / "config.json").read_text())
        t0 = time.time()
        model = cls(
            mesh_device,
            latent_channels=cfg["latent_channels"],
            decoder_input_dim=cfg["decoder_input_dim"],
            decoder_hidden_dim=cfg["decoder_hidden_dim"],
            upsampling_ratios=tuple(cfg["upsampling_ratios"]),
            sampling_rate=int(cfg["sampling_rate"]),
            dtype=dtype,
        )
        sd = {k: v.float() for k, v in V.load_vocoder_state_dict(root).items()}
        model.load_torch_state_dict(sd)
        ttnn.synchronize_device(mesh_device)
        model.load_seconds = time.time() - t0
        return model

    def _host_input(self, latents: torch.Tensor) -> torch.Tensor:
        batch, channels, length = latents.shape
        assert channels == self.latent_channels, latents.shape
        return latents.float().reshape(batch * 2, self.latent_channels // 2, length).transpose(1, 2).contiguous()

    def input_buffer(self, latents: torch.Tensor) -> ttnn.Tensor:
        """A device input tensor ``(2B, L, 64)`` row-major holding ``latents`` (persistent buffer for tracing)."""
        return ttnn.from_torch(
            self._host_input(latents), dtype=self.dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh_device
        )

    def write_input(self, buf: ttnn.Tensor, latents: torch.Tensor) -> None:
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(self._host_input(latents), dtype=self.dtype, layout=ttnn.ROW_MAJOR_LAYOUT), buf
        )

    @staticmethod
    def read_waveform(out: ttnn.Tensor, batch: int) -> torch.Tensor:
        """``(2B, S, C_pad)`` device output -> host fp32 ``[B, 2, S]`` (channel 0 of the padded conv_out)."""
        return ttnn.to_torch(out).float()[..., 0].reshape(batch, 2, -1)

    def forward(self, latents: torch.Tensor) -> ttnn.Tensor:
        """``[B, 128, L]`` host -> ``(2B, 512 L, 32)`` device row-major (channel 0 = the waveform)."""
        h = self.input_buffer(latents)
        out = self.forward_from_device(h)
        ttnn.deallocate(h)
        return out

    def forward_from_device(self, h: ttnn.Tensor) -> ttnn.Tensor:
        """Device-only forward over an ``input_buffer`` tensor (kept), the traceable graph."""
        t = time.perf_counter()
        h2 = self.dec_in_proj(h)
        h = self.conv_in(h2)
        ttnn.deallocate(h2)
        for block in self.blocks:
            h = block(h)
        h2 = _snake(self.snake_out, h)
        ttnn.deallocate(h)
        h = self.conv_out(h2)
        ttnn.deallocate(h2)
        out = ttnn.tanh(h)
        ttnn.deallocate(h)
        self.timings["dispatch_s"] = time.perf_counter() - t
        return out

    def __call__(self, latents: torch.Tensor) -> torch.Tensor:
        t0 = time.perf_counter()
        out = self.forward(latents)
        wav = self.read_waveform(out, latents.shape[0])
        ttnn.deallocate(out)
        self.timings["total_s"] = time.perf_counter() - t0
        return wav

    def release(self) -> None:
        self.deallocate_weights()
