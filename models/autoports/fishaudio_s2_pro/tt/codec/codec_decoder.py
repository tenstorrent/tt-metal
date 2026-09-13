"""Codec front-ends. CPUCodec runs the vendored torch DAC (encoder for reference audio, decoder for output).
The TTNN decoder (Phase C) will implement the same `decode`/`decode_chunked` interface."""
from __future__ import annotations

from typing import Iterator

import numpy as np
import torch

from models.autoports.fishaudio_s2_pro.config import SAMPLE_RATE, SAMPLES_PER_FRAME


class CPUCodec:
    def __init__(self, snapshot, device: str = "cpu", dtype=torch.float32):
        from models.autoports.fishaudio_s2_pro.reference.codec_config import load_codec

        self.model = load_codec(snapshot, device=device, dtype=dtype)
        self.device, self.dtype = device, dtype
        self.sample_rate = SAMPLE_RATE

    @torch.inference_mode()
    def decode(self, codes: torch.Tensor) -> np.ndarray:
        """codes (num_codebooks, T) int -> float32 waveform (T * 2048,) in [-1, 1]."""
        if codes.shape[1] == 0:
            return np.zeros(0, dtype=np.float32)
        c = codes.to(torch.int64).clone().to(self.device)[None]  # upstream clamps IN PLACE: clone
        wav = self.model.from_indices(c)[0, 0].float().cpu().numpy()
        return np.clip(wav, -1.0, 1.0).astype(np.float32)

    @torch.inference_mode()
    def _decode_with_offset(self, codes: torch.Tensor, offset: int) -> np.ndarray:
        """decode() but with the post_module's RoPE positions starting at `offset` (the codes' absolute frame index),
        so a suffix decode reproduces the whole-clip numerics. The window-limited transformer has 8 layers x
        window 128, so a suffix also needs ~512 frames of left context for the tail to converge (<= ~1e-4)."""
        q = self.model.quantizer
        pm = q.post_module
        idx = codes.to(torch.int64).clone().to(self.device)[None]
        idx[:, 0].clamp_(max=q.semantic_quantizer.codebook_size - 1)
        idx[:, 1:].clamp_(max=q.quantizer.codebook_size - 1)
        z = q.semantic_quantizer.from_codes(idx[:, :1])[0] + q.quantizer.from_codes(idx[:, 1:])[0]
        x = z.transpose(1, 2)
        T = x.shape[1]
        mask = pm.make_window_limited_mask(T, None).to(x.device)
        y = super(type(pm), pm).forward(
            pm.look_ahead_conv(pm.input_proj(x)), torch.arange(offset, offset + T, device=x.device), mask
        )
        y = pm.output_proj(y).transpose(1, 2)
        wav = self.model.decoder(q.upsample(y))[0, 0].float().cpu().numpy()
        return np.clip(wav, -1.0, 1.0).astype(np.float32)

    @torch.inference_mode()
    def decode_chunked(
        self, codes: torch.Tensor, chunk_frames: int = 32, ctx_frames: int = 512
    ) -> Iterator[np.ndarray]:
        """Streaming decode: each chunk is decoded with ctx_frames of left context at its ABSOLUTE positions and
        cropped. With ctx_frames=512 the concatenation matches decode() to ~1e-4 (validated in tests); exact
        streaming (per-layer KV cache + conv state) is the Phase C TTNN decoder's job."""
        T = codes.shape[1]
        for s in range(0, T, chunk_frames):
            e = min(T, s + chunk_frames)
            c0 = max(0, s - ctx_frames)
            wav = self._decode_with_offset(codes[:, c0:e], c0)
            yield wav[(s - c0) * SAMPLES_PER_FRAME :]

    @torch.inference_mode()
    def encode(self, wav: np.ndarray) -> torch.Tensor:
        """float32 mono 44.1 kHz waveform -> codes (num_codebooks, T) int64 (reference audio for cloning)."""
        audio = torch.from_numpy(np.asarray(wav, dtype=np.float32)).to(self.device, self.dtype).view(1, 1, -1)
        lengths = torch.tensor([audio.shape[-1]], dtype=torch.long, device=self.device)
        indices, lens = self.model.encode(audio, lengths)
        return indices[0, :, : int(lens[0])].to(torch.int64).cpu()
