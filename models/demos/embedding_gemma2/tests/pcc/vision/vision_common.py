# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Vision test inputs (real images / a real video from the repo) and activation helpers.

Every image case comes out of the HF processor with padding patches (position -1): the processor
fits the aspect ratio into at most 2520 patches with both sides multiples of 3 patches, so N_valid
< 2520 for all of them (landscape 2376, portrait 2340, square 2304, small 2430, tiny 2304).
The video case is 8 frames (1 fps) of a screen recording: 1260 patches per frame (the 140-token
video budget), 1080 of them valid (45 x 24), so 120 soft tokens per frame.
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

import ttnn

_ROOT = Path(__file__).resolve().parents[6]  # tt-metal checkout

IMAGE_FILES = {
    "landscape": "models/sample_data/house_in_field_1080p.jpg",  # 1600 x 900
    "portrait": "models/demos/vision/classification/resnet50/ttnn_resnet/demo/images/"
    "ILSVRC2012_val_00040007_n03877472.JPEG",  # 332 x 500
    "square": "models/experimental/retinanet/resources/dog_800x800.jpg",  # 800 x 800
    "small": "models/sample_data/iam_ocr_image.jpg",  # 248 x 76 (wide text strip)
    "tiny": "models/sample_data/torchvision_mnist_digit_7.jpg",  # 28 x 28 grayscale
}
VIDEO_FILE = "tech_reports/ttnn/images/ttnn-dashboard.mp4"  # 3456 x 1954, 60 fps screen recording
VIDEO_FRAMES = 8

# case -> list of image names, or "video"
CASES = {
    "landscape": ["landscape"],
    "portrait": ["portrait"],
    "square": ["square"],
    "small": ["small"],
    "tiny": ["tiny"],
    "b2_square_small": ["square", "small"],
    "video8": "video",
}
ALL_CASES = tuple(CASES)
# per-layer teacher-forced tests: one image of each patch bucket plus the batched case
LAYER_CASES = ("landscape", "b2_square_small", "video8")


@functools.lru_cache(maxsize=1)
def processor(snapshot: str):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(snapshot)


def video_frames(n: int = VIDEO_FRAMES):
    """`n` RGB frames at 1 fps (every 60th frame) from the repo screen recording, [n, H, W, 3] uint8."""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(_ROOT / VIDEO_FILE))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(i * fps)))
        ok, f = cap.read()
        if not ok:
            raise RuntimeError(f"cannot read frame {i} of {VIDEO_FILE}")
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)


@functools.lru_cache(maxsize=8)
def load_case(snapshot: str, case: str) -> tuple[torch.Tensor, torch.Tensor]:
    """(pixel_values [B, N, 768] fp32, position_ids [B, N, 2] int64) from the HF processor."""
    from PIL import Image

    p = processor(snapshot)
    spec = CASES[case]
    if spec == "video":
        out = p.video_processor(videos=[video_frames()], return_tensors="pt")
        return out["pixel_values_videos"].float(), out["video_position_ids"].long()
    images = [Image.open(_ROOT / IMAGE_FILES[name]).convert("RGB") for name in spec]
    out = p.image_processor(images=images, return_tensors="pt")
    return out["pixel_values"].float(), out["image_position_ids"].long()


def upload_rows(device, x: torch.Tensor, S: int, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    """[B, N, D] torch -> [B, 1, S, D] TILE device tensor, zero padded to S rows."""
    B, N, D = x.shape
    xp = torch.zeros((B, 1, S, D), dtype=torch.float32)
    xp[:, 0, :N] = x.float()
    return ttnn.from_torch(xp, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def download_rows(t: ttnn.Tensor, N: int) -> torch.Tensor:
    """[B, 1, S, D] device -> [B, N, D] float32."""
    x = ttnn.to_torch(t).float()
    return x.reshape(x.shape[0], x.shape[-2], x.shape[-1])[:, :N]
