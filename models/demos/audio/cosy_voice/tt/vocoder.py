from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CosyVoiceVocoderConfig:
    mel_channels: int = 80
    sample_rate: int = 22050


def maybe_resample_mel_for_speed(mel: torch.Tensor, speed: float) -> torch.Tensor:
    if speed == 1.0:
        return mel
    return F.interpolate(mel, size=int(mel.shape[-1] / speed), mode="linear")


def audio_seconds(num_samples: int, sample_rate: int) -> float:
    return float(num_samples) / float(sample_rate) if sample_rate > 0 else 0.0


def prepare_tt_vocoder_input(mel: torch.Tensor, mesh_device, *, dtype=None, memory_config=None):
    import ttnn  # noqa: PLC0415

    return ttnn.from_torch(
        mel,
        device=mesh_device,
        dtype=dtype or ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
    )
