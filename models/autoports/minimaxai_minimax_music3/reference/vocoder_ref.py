# Copyright 2026 The MiniMax Team and The HuggingFace Team. All rights reserved.
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MiniMax-Music3's Flow-VAE waveform decoder (``MiniMaxMusic3Vocoder``) as plain torch, plus the
crop-and-stitch of ``MiniMaxMusic3VocoderDecodeStep``.

Vendored from diffusers ``models/autoencoders/minimax_music3_vocoder.py`` and
``modular_pipelines/minimax_music3/decoders.py`` (Apache-2.0) so the pipeline does not import
diffusers. Differences from the diffusers module, none of which change the arithmetic:

* the ``weight_norm`` parametrization is folded at load time: the checkpoint stores ``weight_g`` /
  ``weight_v`` and the module recomputes ``weight = g * v / ||v||`` (norm over every dim but 0) on
  each forward; here that product is computed once with ``torch._weight_norm`` (the same kernel
  ``torch.nn.utils.weight_norm`` calls) and stored as the conv's ``weight``;
* the model is a plain ``nn.Module`` (no ``ConfigMixin``); the config values are the checkpoint's
  ``vocoder/config.json`` (``latent_channels 128``, ``decoder_input_dim 1024``, ``decoder_hidden_dim 1536``,
  ``upsampling_ratios (8, 8, 4, 2)``, ``sampling_rate 44100``).

Stage 06 runs this on the host in fp32 (a DAC-style decoder: Snake activations, dilated ``Conv1d`` k=7,
``ConvTranspose1d`` upsampling 8 x 8 x 4 x 2 = 512 samples per latent). Stage 07 ports it to TTNN.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

LATENT_HOP_LENGTH = 512  # waveform samples per Flow-VAE latent frame (8 * 8 * 4 * 2)
SAMPLING_RATE = 44100
# decoders.py: every window after the first drops its leading 86 latent frames and every window before
# the last drops its trailing 344 - 86 latent frames, so the kept spans tile the full song.
CROP_LEFT_LATENT = 86
CROP_RIGHT_LATENT = 344 - 86


class MiniMaxMusic3Snake1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        hidden_states = hidden_states.reshape(shape[0], shape[1], -1)
        hidden_states = hidden_states + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * hidden_states).pow(2)
        return hidden_states.reshape(shape)


class MiniMaxMusic3VocoderResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int):
        super().__init__()
        pad = (7 - 1) * dilation // 2
        self.snake1 = MiniMaxMusic3Snake1d(dim)
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad)
        self.snake2 = MiniMaxMusic3Snake1d(dim)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = self.conv2(self.snake2(self.conv1(self.snake1(hidden_states))))
        return hidden_states + residual


class MiniMaxMusic3VocoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int):
        super().__init__()
        self.snake1 = MiniMaxMusic3Snake1d(input_dim)
        self.conv_t1 = nn.ConvTranspose1d(
            input_dim, output_dim, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2)
        )
        self.res_unit1 = MiniMaxMusic3VocoderResidualUnit(output_dim, dilation=1)
        self.res_unit2 = MiniMaxMusic3VocoderResidualUnit(output_dim, dilation=3)
        self.res_unit3 = MiniMaxMusic3VocoderResidualUnit(output_dim, dilation=9)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_t1(self.snake1(hidden_states))
        hidden_states = self.res_unit1(hidden_states)
        hidden_states = self.res_unit2(hidden_states)
        return self.res_unit3(hidden_states)


class MiniMaxMusic3Vocoder(nn.Module):
    """The Flow-VAE waveform decoder: ``(batch, 128, length)`` latents -> ``(batch, 2, 512 * length)`` stereo.

    The two audio channels are decoded as two folded 64-channel latent streams (``reshape(batch * 2, 64, L)``)."""

    def __init__(
        self,
        latent_channels: int = 128,
        decoder_input_dim: int = 1024,
        decoder_hidden_dim: int = 1536,
        upsampling_ratios: Sequence[int] = (8, 8, 4, 2),
        sampling_rate: int = SAMPLING_RATE,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.sampling_rate = sampling_rate
        self.hop_length = int(math.prod(upsampling_ratios))
        self.dec_in_proj = nn.Conv1d(latent_channels // 2, decoder_input_dim, kernel_size=1)
        self.conv_in = nn.Conv1d(decoder_input_dim, decoder_hidden_dim, kernel_size=7, padding=3)
        blocks = []
        output_dim = decoder_hidden_dim
        for index, stride in enumerate(upsampling_ratios):
            input_dim = decoder_hidden_dim // (2**index)
            output_dim = decoder_hidden_dim // (2 ** (index + 1))
            blocks.append(MiniMaxMusic3VocoderBlock(input_dim, output_dim, stride))
        self.blocks = nn.ModuleList(blocks)
        self.snake_out = MiniMaxMusic3Snake1d(output_dim)
        self.conv_out = nn.Conv1d(output_dim, 1, kernel_size=7, padding=3)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        batch_size, _, length = latents.shape
        hidden_states = latents.reshape(batch_size * 2, self.latent_channels // 2, length)
        hidden_states = self.conv_in(self.dec_in_proj(hidden_states))
        for block in self.blocks:
            hidden_states = block(hidden_states)
        waveform = torch.tanh(self.conv_out(self.snake_out(hidden_states)))
        return waveform.reshape(batch_size, 2, -1)


# ----------------------------------------------------------------------------- loading
def fold_weight_norm(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Replace every ``<conv>.weight_g`` / ``<conv>.weight_v`` pair by ``<conv>.weight`` (fp32, exactly the
    tensor ``torch.nn.utils.weight_norm`` recomputes on each forward)."""
    out: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.endswith(".weight_g"):
            base = key[: -len(".weight_g")]
            g, v = value.float(), state_dict[base + ".weight_v"].float()
            out[base + ".weight"] = torch._weight_norm(v, g, 0)
        elif key.endswith(".weight_v"):
            continue
        else:
            out[key] = value
    return out


def weights_root(weights_dir: Optional[Path] = None) -> Path:
    if weights_dir is not None:
        return Path(weights_dir)
    from models.autoports.minimaxai_minimax_music3.reference.hf_llm import weights_dir as _wd

    return _wd()


def load_vocoder_state_dict(weights_dir: Optional[Path] = None) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    path = weights_root(weights_dir) / "vocoder" / "diffusion_pytorch_model.safetensors"
    return fold_weight_norm(load_file(str(path)))


def load_vocoder(weights_dir: Optional[Path] = None, dtype: torch.dtype = torch.float32) -> MiniMaxMusic3Vocoder:
    """The vocoder with the checkpoint's config and folded weights, in eval mode."""
    root = weights_root(weights_dir)
    cfg = json.loads((root / "vocoder" / "config.json").read_text())
    model = MiniMaxMusic3Vocoder(
        latent_channels=cfg["latent_channels"],
        decoder_input_dim=cfg["decoder_input_dim"],
        decoder_hidden_dim=cfg["decoder_hidden_dim"],
        upsampling_ratios=tuple(cfg["upsampling_ratios"]),
        sampling_rate=int(cfg["sampling_rate"]),
    )
    missing, unexpected = model.load_state_dict(load_vocoder_state_dict(root), strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    return model.to(dtype).eval().requires_grad_(False)


# ----------------------------------------------------------------------------- decoders.py
def crop_bounds(chunk_index: int, num_chunks: int, num_samples: int, hop_length: int = LATENT_HOP_LENGTH):
    """``[left, right)`` sample span kept from window ``chunk_index`` of ``num_chunks`` (``decoders.py``)."""
    left = 0 if chunk_index == 0 else CROP_LEFT_LATENT * hop_length
    right = 0 if chunk_index == num_chunks - 1 else CROP_RIGHT_LATENT * hop_length
    return left, num_samples - right


def stitch_waveforms(waveforms: Sequence[torch.Tensor], hop_length: int = LATENT_HOP_LENGTH) -> torch.Tensor:
    """Crop each window's decoded ``[B, 2, samples]`` and concatenate, clamped to ``[-1, 1]`` (fp32)."""
    num_chunks = len(waveforms)
    parts: List[torch.Tensor] = []
    for k, waveform in enumerate(waveforms):
        left, right = crop_bounds(k, num_chunks, waveform.shape[-1], hop_length)
        parts.append(waveform[..., left:right])
    return torch.cat(parts, dim=-1).float().clamp(-1.0, 1.0)


@torch.no_grad()
def decode_latent_chunks(vocoder: MiniMaxMusic3Vocoder, latent_chunks: Sequence[torch.Tensor]) -> torch.Tensor:
    """``MiniMaxMusic3VocoderDecodeStep``: vocode every uncropped ``[1, 128, L_k]`` window and stitch -> ``[1, 2, S]``."""
    dtype = next(vocoder.parameters()).dtype
    waveforms = [vocoder(latents.to(dtype)) for latents in latent_chunks]
    return stitch_waveforms(waveforms, vocoder.hop_length)
