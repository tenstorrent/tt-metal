# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Reference (pure-PyTorch) XTTS-v2 ``HifiDecoder`` — latent pre-upsampling + generator.

``HifiDecoder.forward`` linearly upsamples the GPT latents along time in two steps
before the HiFi-GAN generator:

    z = F.interpolate(latents.transpose(1,2), scale=ar_mel_length_compression/output_hop_length, "linear")
    z = F.interpolate(z, scale=output_sample_rate/input_sample_rate, "linear")   # if rates differ
    o = waveform_decoder(z, g)

The speaker encoder that produces ``g`` is a later phase; here ``g`` is supplied.

Also provides :func:`build_linear_interp_matrix`, the exact matrix form of a 1D
``F.interpolate(mode="linear", align_corners=False)`` — used by the TTNN port to do
the upsample as a single on-device matmul (ttnn has no linear-1D interpolate).
"""

import math

import torch
import torch.nn.functional as F

from models.experimental.xtts.reference.xtts_hifigan import build_reference_waveform_decoder

# HifiDecoder hyper-parameters (coqui/XTTS-v2).
AR_MEL_LENGTH_COMPRESSION = 1024
OUTPUT_HOP_LENGTH = 256
INPUT_SAMPLE_RATE = 22050
OUTPUT_SAMPLE_RATE = 24000
LATENT_SCALE = AR_MEL_LENGTH_COMPRESSION / OUTPUT_HOP_LENGTH  # 4.0
SR_SCALE = OUTPUT_SAMPLE_RATE / INPUT_SAMPLE_RATE  # 160/147 ≈ 1.08844


def build_linear_interp_matrix(length_in: int, scale_factor: float) -> torch.Tensor:
    """``[L_out, L_in]`` matrix M such that ``einsum('oi,...i->...o', M, x)`` equals
    ``F.interpolate(x, scale_factor=[scale_factor], mode="linear")`` (align_corners=False).

    Uses torch's default source-index scale ``1/scale_factor`` (recompute_scale_factor
    disabled). Exact at all lengths that occur in XTTS (the fractional step always
    sees >= 4*T samples).
    """
    length_out = int(math.floor(length_in * scale_factor))
    src_scale = 1.0 / scale_factor
    matrix = torch.zeros(length_out, length_in)
    for dst in range(length_out):
        src = max(0.0, src_scale * (dst + 0.5) - 0.5)
        lo = int(math.floor(src))
        hi = min(lo + 1, length_in - 1)
        w = src - lo
        matrix[dst, lo] += 1.0 - w
        matrix[dst, hi] += w
    return matrix


class XttsHifiDecoderReference(torch.nn.Module):
    """Latent linear-upsample + HiFi-GAN generator (speaker embedding ``g`` supplied)."""

    def __init__(self, state_dict):
        super().__init__()
        self.waveform_decoder = build_reference_waveform_decoder(state_dict)

    def forward(self, latents, g):
        # latents [B, T, 1024] -> [B, 1024, T]; g [B, 512, 1].
        z = F.interpolate(latents.transpose(1, 2), scale_factor=[LATENT_SCALE], mode="linear")
        if OUTPUT_SAMPLE_RATE != INPUT_SAMPLE_RATE:
            z = F.interpolate(z, scale_factor=[SR_SCALE], mode="linear")
        return self.waveform_decoder(z, g)
