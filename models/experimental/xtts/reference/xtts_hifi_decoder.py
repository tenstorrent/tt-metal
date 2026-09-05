# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import math

import torch
import torch.nn.functional as F

from models.experimental.xtts.reference.xtts_hifigan import build_reference_waveform_decoder
from models.experimental.xtts.reference.xtts_mel import build_reference_mel_frontend
from models.experimental.xtts.reference.xtts_speaker_encoder import build_reference_speaker_encoder

from models.experimental.xtts.config import (  # noqa: F401
    AR_MEL_LENGTH_COMPRESSION,
    INPUT_SAMPLE_RATE,
    LATENT_SCALE,
    OUTPUT_HOP_LENGTH,
    OUTPUT_SAMPLE_RATE,
    SR_SCALE,
)


def build_linear_interp_matrix(length_in: int, scale_factor: float) -> torch.Tensor:
    """Build a dense linear-interpolation resampling matrix."""
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
    def __init__(self, state_dict):
        """Load the HiFi-GAN waveform decoder from checkpoint weights."""
        super().__init__()
        self.waveform_decoder = build_reference_waveform_decoder(state_dict)

    def forward(self, latents, g):
        """Upsample latents and decode waveform conditioned on speaker emb."""
        z = F.interpolate(latents.transpose(1, 2), scale_factor=[LATENT_SCALE], mode="linear")
        if OUTPUT_SAMPLE_RATE != INPUT_SAMPLE_RATE:
            z = F.interpolate(z, scale_factor=[SR_SCALE], mode="linear")
        return self.waveform_decoder(z, g)


class XttsHifiDecoderFull(torch.nn.Module):
    def __init__(self, state_dict):
        """Build mel frontend, speaker encoder, and waveform decoder."""
        super().__init__()
        self.mel_frontend = build_reference_mel_frontend(state_dict)
        self.speaker_encoder = build_reference_speaker_encoder(state_dict)
        self.decoder = XttsHifiDecoderReference(state_dict)

    @torch.no_grad()
    def speaker_embedding(self, ref_wav):
        """Compute speaker embedding from a reference waveform."""
        mel = self.mel_frontend(ref_wav)
        return self.speaker_encoder(mel).unsqueeze(-1)

    @torch.no_grad()
    def forward(self, latents, ref_wav):
        """Decode waveform from GPT latents and reference speaker audio."""
        return self.decoder(latents, self.speaker_embedding(ref_wav))
