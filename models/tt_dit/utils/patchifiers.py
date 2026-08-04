# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Self-contained LTX-2 latent shape types and patchifier grid helpers."""

from __future__ import annotations

from typing import NamedTuple

import torch


class VideoPixelShape(NamedTuple):
    """Shape of a video in pixel space: (batch, frames, height, width, fps)."""

    batch: int
    frames: int
    height: int
    width: int
    fps: float


class VideoLatentShape(NamedTuple):
    """Shape of a video in VAE latent space: (batch, channels, frames, height, width)."""

    batch: int
    channels: int
    frames: int
    height: int
    width: int


class AudioLatentShape(NamedTuple):
    """Shape of audio in VAE latent space: (batch, channels, frames, mel_bins)."""

    batch: int
    channels: int
    frames: int
    mel_bins: int

    @staticmethod
    def from_duration(
        batch: int,
        duration: float,
        channels: int = 8,
        mel_bins: int = 16,
        sample_rate: int = 16000,
        hop_length: int = 160,
        audio_latent_downsample_factor: int = 4,
    ) -> AudioLatentShape:
        latents_per_second = float(sample_rate) / float(hop_length) / float(audio_latent_downsample_factor)
        return AudioLatentShape(
            batch=batch,
            channels=channels,
            frames=round(duration * latents_per_second),
            mel_bins=mel_bins,
        )

    @staticmethod
    def from_video_pixel_shape(
        shape: VideoPixelShape,
        channels: int = 8,
        mel_bins: int = 16,
        sample_rate: int = 16000,
        hop_length: int = 160,
        audio_latent_downsample_factor: int = 4,
    ) -> AudioLatentShape:
        return AudioLatentShape.from_duration(
            batch=shape.batch,
            duration=float(shape.frames) / float(shape.fps),
            channels=channels,
            mel_bins=mel_bins,
            sample_rate=sample_rate,
            hop_length=hop_length,
            audio_latent_downsample_factor=audio_latent_downsample_factor,
        )


def video_get_patch_grid_bounds(
    shape: VideoLatentShape,
    patch_size: tuple[int, int, int] = (1, 1, 1),
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Per-patch [start, end) grid bounds for video latent tokens.

    Returns (batch, 3, num_patches, 2) where axis 1 is (frame, height, width)
    and axis 3 is [start, end).
    """
    grid_coords = torch.meshgrid(
        torch.arange(start=0, end=shape.frames, step=patch_size[0], device=device),
        torch.arange(start=0, end=shape.height, step=patch_size[1], device=device),
        torch.arange(start=0, end=shape.width, step=patch_size[2], device=device),
        indexing="ij",
    )
    patch_starts = torch.stack(grid_coords, dim=0)  # (3, F, H, W)
    patch_size_delta = torch.tensor(patch_size, device=patch_starts.device, dtype=patch_starts.dtype).view(3, 1, 1, 1)
    patch_ends = patch_starts + patch_size_delta
    latent_coords = torch.stack((patch_starts, patch_ends), dim=-1)  # (3, F, H, W, 2)

    F, H, W = latent_coords.shape[1], latent_coords.shape[2], latent_coords.shape[3]
    latent_coords = latent_coords.reshape(3, F * H * W, 2)
    latent_coords = latent_coords.unsqueeze(0).expand(shape.batch, -1, -1, -1)
    return latent_coords


def audio_get_patch_grid_bounds(
    shape: AudioLatentShape,
    sample_rate: int = 16000,
    hop_length: int = 160,
    audio_latent_downsample_factor: int = 4,
    is_causal: bool = True,
    shift: int = 0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Per-patch temporal bounds (seconds) for audio latent tokens, shape (batch, 1, num_frames, 2)."""

    def _latent_to_seconds(start_idx: int, end_idx: int) -> torch.Tensor:
        frame = torch.arange(start_idx, end_idx, dtype=torch.float32, device=device)
        mel_frame = frame * audio_latent_downsample_factor
        if is_causal:
            mel_frame = (mel_frame + 1 - audio_latent_downsample_factor).clamp(min=0)
        return mel_frame * hop_length / sample_rate

    num_steps = shape.frames
    start_timings = _latent_to_seconds(shift, num_steps + shift)
    start_timings = start_timings.unsqueeze(0).expand(shape.batch, -1).unsqueeze(1)  # (B, 1, N)

    end_timings = _latent_to_seconds(shift + 1, num_steps + shift + 1)
    end_timings = end_timings.unsqueeze(0).expand(shape.batch, -1).unsqueeze(1)  # (B, 1, N)

    return torch.stack([start_timings, end_timings], dim=-1)  # (B, 1, N, 2)


def get_pixel_coords(
    latent_coords: torch.Tensor,
    scale_factors: tuple[int, int, int] = (8, 32, 32),
    causal_fix: bool = False,
) -> torch.Tensor:
    """Scale latent-space [start, end) coordinates to pixel space per axis.

    scale_factors is (temporal, height, width); causal_fix offsets the first
    temporal frame for causal encoding.
    """
    broadcast_shape = [1] * latent_coords.ndim
    broadcast_shape[1] = -1
    scale_tensor = torch.tensor(scale_factors, device=latent_coords.device).view(*broadcast_shape)

    pixel_coords = latent_coords * scale_tensor

    if causal_fix:
        pixel_coords[:, 0, ...] = (pixel_coords[:, 0, ...] + 1 - scale_factors[0]).clamp(min=0)

    return pixel_coords
