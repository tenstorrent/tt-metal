# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-side condition-image preprocessing, mirroring QwenImage21Pipeline + VaeImageProcessor:
resize to ~output_resolution^2 pixels (multiples of 32), RGBA for the VAE, RGB-over-white for the vision encoder."""
from __future__ import annotations

import math

import numpy as np
import torch
from PIL import Image

from .config import IMAGE_PAD_TOKEN_ID, SYS_PROMPT

PROMPT_TEMPLATE_TI2I = (
    f"<|im_start|>system\n{SYS_PROMPT}<|im_end|>\n"
    "<|im_start|>user\n<image1><|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def calculate_dimensions(target_area: int, ratio: float):
    """diffusers' calculate_dimensions: (width, height) rounded to multiples of 32."""
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    size = round(width / 32) * 32, round(height / 32) * 32
    if min(size) == 0:
        raise ValueError("condition image aspect ratio rounds to a zero dimension")
    return size


def edit_prompt_text(prompt: str, n_images: int) -> str:
    """The ti2i chat template with one vision placeholder per condition image."""
    replace = "<image1><|vision_start|><|image_pad|><|vision_end|>"
    for i in range(2, n_images + 1):
        replace += f" <image{i}><|vision_start|><|image_pad|><|vision_end|>"
    template = PROMPT_TEMPLATE_TI2I.replace("<image1><|vision_start|><|image_pad|><|vision_end|>", replace)
    return template.format(prompt if prompt else " ")


def condition_size(img: Image.Image, output_resolution: int = 1024):
    w, h = img.size
    return calculate_dimensions(output_resolution * output_resolution, w / h)


def resize_pil(img: Image.Image, width: int, height: int) -> Image.Image:
    """VaeImageProcessor.resize for PIL input (LANCZOS)."""
    return img.resize((width, height), resample=Image.LANCZOS)


def prepare_condition_image(img: Image.Image, output_resolution: int = 1024):
    """-> (vision_rgb PIL (alpha composited over white), vae_tensor [1, 4, 1, H, W] float in [-1, 1], (W, H))."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    width, height = condition_size(img, output_resolution)
    resized = resize_pil(img, width, height)
    white = Image.new("RGB", resized.size, (255, 255, 255))
    white.paste(resized, mask=resized.getchannel("A"))
    arr = np.asarray(resized).astype(np.float32) / 255.0  # [H, W, 4]
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1, 4, H, W]
    t = 2.0 * t - 1.0
    return white, t.unsqueeze(2), (width, height)


def image_pad_mask_from_ids(input_ids: torch.Tensor, drop_idx: int) -> torch.Tensor:
    """[L - drop_idx] bool, True at <|image_pad|> positions after the system tokens are dropped."""
    return input_ids.reshape(-1)[drop_idx:] == IMAGE_PAD_TOKEN_ID
