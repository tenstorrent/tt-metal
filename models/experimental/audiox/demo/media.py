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
    frames_tchw = _resize_video_frames(frames_tchw, size=image_size)
    return frames_tchw.unsqueeze(0).contiguous()


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
