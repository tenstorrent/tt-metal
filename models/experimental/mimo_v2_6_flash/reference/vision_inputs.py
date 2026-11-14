# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ViT inputs exactly as served: vLLM 0.31.0's ``MiMoV2OmniMultiModalProcessor`` run on the host.

The processor is built the way the server builds it (``ModelConfig`` of the checkpoint with the omni architecture,
``dtype=bfloat16``, so ``pixel_values`` arrive in bf16 like in serving) and called on PIL images or decoded video
frames (``np.ndarray`` [T, H, W, 3] uint8, what vLLM's video loader hands the processor; the processor treats the
frames as 1 fps). It needs a vLLM install (``vllm-env`` on bhg2); importing this module does not import vLLM.

vLLM platform plugins (vllm-tt-plugin) are disabled while the processor is built unless ``VLLM_PLUGINS`` is set:
building a processor needs no device.
"""

import os
from functools import lru_cache

import numpy as np
import torch

IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
VIDEO_PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>"
OMNI_ARCH = "MiMoV2OmniForCausalLM"


@lru_cache(maxsize=None)
def vllm_processor(model_path, max_model_len=262144):
    os.environ.setdefault("VLLM_PLUGINS", "")
    from vllm.config import ModelConfig
    from vllm.multimodal import MULTIMODAL_REGISTRY

    model_config = ModelConfig(
        model=model_path,
        trust_remote_code=True,
        hf_overrides={"architectures": [OMNI_ARCH]},
        max_model_len=max_model_len,
        dtype="bfloat16",
    )
    return MULTIMODAL_REGISTRY.create_processor(model_config)


def _run(model_path, prompt, mm_data):
    proc = vllm_processor(model_path)
    items = proc.info.parse_mm_data(mm_data)
    return proc(prompt, items, hf_processor_mm_kwargs={})


def _field(item, key):
    return item[key].data


def process_images(model_path, images):
    """PIL images -> {pixel_values [sum patches, 1536], image_grid_thw [n, 3] int64, num_tokens [n]}.

    ``num_tokens`` are the placeholder lengths vLLM gives each image in the prompt."""
    out = _run(model_path, IMAGE_PLACEHOLDER * len(images), {"image": list(images)})
    items = out["mm_kwargs"]["image"]
    return {
        "pixel_values": torch.cat([_field(it, "pixel_values") for it in items], dim=0),
        "image_grid_thw": torch.stack([_field(it, "image_grid_thw") for it in items]).to(torch.int64),
        "num_tokens": [ph.length for ph in out["mm_placeholders"]["image"]],
    }


def process_video(model_path, frames):
    """Decoded frames [T, H, W, 3] uint8 -> {pixel_values_videos [patches, 1536], video_grid_thw [1, 3] int64,
    num_tokens [1]}. ``num_tokens`` counts the placeholder rows that take embeddings (``is_embed``): the video's
    placeholder range also holds timestamp and vision start / end tokens."""
    assert isinstance(frames, np.ndarray) and frames.ndim == 4 and frames.shape[-1] == 3, frames.shape
    out = _run(model_path, VIDEO_PLACEHOLDER, {"video": [frames]})
    (item,) = out["mm_kwargs"]["video"]
    (ph,) = out["mm_placeholders"]["video"]
    n_embed = ph.length if ph.is_embed is None else int(ph.is_embed.sum())
    return {
        "pixel_values_videos": _field(item, "pixel_values_videos"),
        "video_grid_thw": _field(item, "video_grid_thw").reshape(1, 3).to(torch.int64),
        "num_tokens": [n_embed],
    }


# ---------------------------------------------------------------------------------------------------------------
# Test media (repo sample images, so tests need no downloads)
# ---------------------------------------------------------------------------------------------------------------

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "sample_data")


def sample_image(name="huggingface_cat_image.jpg", size=None):
    """A repo sample image as RGB PIL, resized to ``size`` (w, h) when given."""
    from PIL import Image

    img = Image.open(os.path.join(SAMPLE_DIR, name)).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.BICUBIC)
    return img


def sample_video(num_frames=32, size=(640, 360), name="house_in_field_1080p.jpg"):
    """[num_frames, h, w, 3] uint8: a ``size`` window panning and zooming across a sample image (natural content
    that changes from frame to frame)."""
    from PIL import Image

    src = Image.open(os.path.join(SAMPLE_DIR, name)).convert("RGB")
    sw, sh = src.size
    w, h = size
    frames = []
    for f in range(num_frames):
        a = f / max(num_frames - 1, 1)
        crop_w = int(sw * (0.6 - 0.2 * a))
        crop_h = crop_w * h // w
        x0 = int((sw - crop_w) * a)
        y0 = int((sh - crop_h) * (0.5 + 0.3 * np.sin(np.pi * a)) / 1.3)
        crop = src.crop((x0, y0, x0 + crop_w, y0 + crop_h)).resize((w, h), Image.BICUBIC)
        frames.append(np.asarray(crop))
    return np.stack(frames)
