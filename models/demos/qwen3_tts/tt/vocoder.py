# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3-TTS Vocoder (Code2Wav): converts 16-codebook tokens to 24kHz waveform.

Architecture:
    16 codebook tokens [B, T, 16] → SplitResidualVQ dequant → [B, 512, T]
    → CausalConv1d(512→1024, k=3) → 8-layer Transformer (SW=72) → [B, T, 1024]
    → 2x CausalTransConv(1024, factor=2) + ConvNeXtBlock → 4x upsample
    → Conv(1024→1536, k=7) → 4x DecoderBlock (rate=8,5,4,3, SnakeBeta) → Conv(96→1, k=7)
    → 24kHz waveform

Total upsample: 2*2*8*5*4*3 = 1920x (12.5 Hz → 24 kHz)

Implementation strategy: the Vocoder runs on the host (CPU) using the PyTorch
reference model. It's 114M params and runs once per utterance (non-autoregressive).
The 8-layer Transformer (hidden=512) could be moved to TT device for acceleration,
but the ConvNet upsampler (majority of compute) is hard to port efficiently.
"""

import numpy as np
import torch
from loguru import logger

from models.demos.qwen3_tts.reference.vocoder_ref import VocoderReference


class Vocoder:
    """
    Vocoder wrapper: runs Code2Wav decoder on host, returns waveform.

    Usage:
        vocoder = Vocoder.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
        waveform = vocoder.decode(all_codebooks)
        # waveform is numpy array [num_samples] at 24kHz
    """

    def __init__(self, model: VocoderReference, dtype=torch.bfloat16):
        self.model = model
        self.model.eval()
        self.dtype = dtype
        self.sample_rate = 24000
        self.total_upsample = model.total_upsample

    @classmethod
    def from_pretrained(cls, model_name_or_path, dtype=torch.bfloat16):
        model = VocoderReference.from_pretrained(model_name_or_path)
        model = model.to(dtype)
        return cls(model, dtype)

    @torch.inference_mode()
    def decode(self, codebook_tokens: torch.Tensor, chunk_decode: bool = True) -> np.ndarray:
        """Convert codebook tokens to waveform.

        Args:
            codebook_tokens: [B, num_frames, 16] all codebook indices (int64)
            chunk_decode: if True, use chunked decoding for memory efficiency

        Returns:
            waveform: numpy array [num_samples] at 24kHz (first batch element)
        """
        codes = codebook_tokens.long()
        num_q = 16  # default; check model if available
        if hasattr(self.model, 'quantizer') and hasattr(self.model.quantizer, 'rvq_first'):
            num_q = self.model.quantizer.rvq_first.vq.layers.__len__()
            num_q += self.model.quantizer.rvq_rest.vq.layers.__len__()
        # If last dim matches num_quantizers, input is [B, T, num_q] → transpose to [B, num_q, T]
        if codes.dim() == 3 and codes.shape[-1] == num_q:
            codes = codes.transpose(1, 2)

        codes = codes.to(next(self.model.parameters()).device)

        logger.info(f"Vocoder decode: {codes.shape[2]} frames → ~{codes.shape[2] * self.total_upsample} samples")

        if chunk_decode and codes.shape[-1] > 300:
            wav = self.model.chunked_decode(codes)
        else:
            wav = self.model(codes)

        wav = wav.squeeze(1).float()  # [B, num_samples]
        waveform = wav[0].cpu().numpy()

        duration = len(waveform) / self.sample_rate
        logger.info(f"Vocoder output: {len(waveform)} samples ({duration:.2f}s)")

        return waveform

    @torch.inference_mode()
    def decode_batch(self, codebook_tokens: torch.Tensor) -> torch.Tensor:
        """Convert codebook tokens to waveform tensor (batched).

        Args:
            codebook_tokens: [B, num_frames, 16] or [B, 16, num_frames]

        Returns:
            waveform: torch.Tensor [B, num_samples]
        """
        codes = codebook_tokens.long()
        if codes.dim() == 3 and codes.shape[-1] == 16:
            codes = codes.transpose(1, 2)

        codes = codes.to(next(self.model.parameters()).device)
        wav = self.model.chunked_decode(codes)
        return wav.squeeze(1).float()
