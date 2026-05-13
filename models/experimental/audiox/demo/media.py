# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import torch
import torch.nn.functional as F


def _resample_video_frames(frames_tchw: torch.Tensor, target_frames: int) -> torch.Tensor:
    """Return exactly ``target_frames`` frames by nearest-neighbor sampling."""
    if frames_tchw.ndim != 4:
        raise ValueError(f"expected [T, C, H, W] frames, got shape {tuple(frames_tchw.shape)}")
    if frames_tchw.shape[0] == 0:
        raise ValueError("video contains no frames")
    if frames_tchw.shape[0] == target_frames:
        return frames_tchw.float()
    if frames_tchw.shape[0] == 1:
        return frames_tchw.expand(target_frames, -1, -1, -1).float().clone()

    idx = torch.linspace(0, frames_tchw.shape[0] - 1, steps=target_frames)
    idx = idx.round().to(dtype=torch.long)
    return frames_tchw.index_select(0, idx).float()


def _resize_video_frames(frames_tchw: torch.Tensor, size: int = 224) -> torch.Tensor:
    """Resize each frame to CLIP's expected square spatial size."""
    return F.interpolate(frames_tchw.float(), size=(size, size), mode="bilinear", align_corners=False)


def prepare_video_prompt(frames_tchw: torch.Tensor, *, target_frames: int, image_size: int = 224) -> torch.Tensor:
    """Convert raw frames into AudioX CLIP conditioner input [1, T, C, H, W]."""
    frames_tchw = _resample_video_frames(frames_tchw, target_frames=target_frames)
    if frames_tchw.shape[1] != 3:
        raise ValueError(f"expected RGB frames, got shape {tuple(frames_tchw.shape)}")
    frames_tchw = _resize_video_frames(frames_tchw, size=image_size)
    return frames_tchw.unsqueeze(0).contiguous()


def prepare_audio_prompt(
    waveform_ct: torch.Tensor,
    *,
    target_samples: int,
    target_channels: int = 2,
) -> torch.Tensor:
    """Convert raw waveform into AudioX audio conditioner input [1, C, T]."""
    if waveform_ct.ndim != 2:
        raise ValueError(f"expected [C, T] audio, got shape {tuple(waveform_ct.shape)}")
    if waveform_ct.shape[0] == 0:
        raise ValueError("audio contains no channels")

    waveform_ct = waveform_ct.float()
    if waveform_ct.shape[0] == 1 and target_channels == 2:
        waveform_ct = waveform_ct.expand(2, -1).clone()
    elif waveform_ct.shape[0] != target_channels:
        waveform_ct = waveform_ct[:target_channels]
        if waveform_ct.shape[0] < target_channels:
            pad_c = target_channels - waveform_ct.shape[0]
            waveform_ct = torch.cat([waveform_ct, waveform_ct[-1:].expand(pad_c, -1)], dim=0)

    if waveform_ct.shape[1] < target_samples:
        waveform_ct = F.pad(waveform_ct, (0, target_samples - waveform_ct.shape[1]))
    else:
        waveform_ct = waveform_ct[:, :target_samples]
    return waveform_ct.unsqueeze(0).contiguous()


def resample_output_audio(audio_bct: torch.Tensor, *, input_sample_rate: int, output_sample_rate: int) -> torch.Tensor:
    """Resample generated audio [B, C, T] onto the requested output rate."""
    if audio_bct.ndim != 3:
        raise ValueError(f"expected [B, C, T] audio, got shape {tuple(audio_bct.shape)}")
    if input_sample_rate == output_sample_rate:
        return audio_bct

    import torchaudio

    batch, channels, samples = audio_bct.shape
    flat = audio_bct.reshape(batch * channels, samples)
    flat = torchaudio.functional.resample(flat, input_sample_rate, output_sample_rate)
    return flat.reshape(batch, channels, flat.shape[-1])


def load_video_prompt(path: Path, *, target_frames: int, image_size: int = 224) -> torch.Tensor:
    """Load a video file into AudioX's CLIP conditioner input format."""
    from torchvision.io import read_video

    frames, _, _ = read_video(str(path), pts_unit="sec")
    if frames.ndim != 4:
        raise ValueError(f"expected 4D video tensor from {path}, got shape {tuple(frames.shape)}")

    if frames.shape[-1] != 3:
        raise ValueError(f"expected RGB video from {path}, got shape {tuple(frames.shape)}")
    frames_tchw = frames.permute(0, 3, 1, 2)
    return prepare_video_prompt(frames_tchw, target_frames=target_frames, image_size=image_size)


def load_image_prompt(path: Path, *, target_frames: int, image_size: int = 224) -> torch.Tensor:
    """Load a single image and repeat it across the visual prompt timeline."""
    from torchvision.io import read_image

    image_chw = read_image(str(path))
    if image_chw.ndim != 3 or image_chw.shape[0] != 3:
        raise ValueError(f"expected RGB image from {path}, got shape {tuple(image_chw.shape)}")

    frames_tchw = image_chw.unsqueeze(0)
    return prepare_video_prompt(frames_tchw, target_frames=target_frames, image_size=image_size)


def load_audio_prompt(path: Path, *, target_sample_rate: int, target_samples: int, target_channels: int = 2) -> torch.Tensor:
    """Load an audio file into AudioX's audio conditioner input format."""
    import torchaudio

    waveform, sample_rate = torchaudio.load(str(path))
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
    return prepare_audio_prompt(waveform, target_samples=target_samples, target_channels=target_channels)
