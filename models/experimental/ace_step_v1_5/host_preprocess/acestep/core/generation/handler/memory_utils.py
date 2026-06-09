# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Memory helpers for TTNN host_preprocess (CPU-only)."""

import os
from typing import Optional

import torch


class MemoryUtilsMixin:
    """Mixin containing memory sizing helpers for CPU preprocessing."""

    def is_silence(self, audio: torch.Tensor) -> bool:
        """Return True when audio is effectively silent."""
        return bool(torch.all(audio.abs() < 1e-6))

    VAE_DECODE_MAX_CHUNK_SIZE = 512

    def _get_auto_decode_chunk_size(self) -> int:
        """Choose a conservative VAE decode chunk size."""
        override = os.environ.get("ACESTEP_VAE_DECODE_CHUNK_SIZE")
        if override:
            try:
                value = int(override)
                if value > 0:
                    return value
            except ValueError:
                # Ignore malformed override and use the conservative default.
                return min(256, self.VAE_DECODE_MAX_CHUNK_SIZE)
        return min(256, self.VAE_DECODE_MAX_CHUNK_SIZE)

    def _should_offload_wav_to_cpu(self) -> bool:
        """CPU preprocess always keeps decoded wavs on host."""
        return True

    def _vram_guard_reduce_batch(
        self,
        batch_size: int,
        audio_duration: Optional[float] = None,
        use_lm: bool = False,
    ) -> int:
        """No VRAM batch clamping in TTNN host_preprocess."""
        _ = audio_duration, use_lm
        return batch_size

    def _get_vae_dtype(self, device: Optional[str] = None) -> torch.dtype:
        """VAE runs in float32 on CPU for preprocessing."""
        _ = device
        return torch.float32
