# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Input construction for the Qwen-Image-Edit image_edit task.

Everything here is input ENCODING: the HF image processor / Qwen2VLProcessor, the prompt template, the
initial noise and the scheduler's sigma schedule. It runs on host BEFORE the TT forward, exactly as
QwenImageEditPipeline.__call__ prepares them, and is shared by the TT pipeline and the HF golden so both
see identical inputs.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image

MODEL_ID = "Qwen/Qwen-Image-Edit"

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
SAMPLE_IMAGES = [
    "models/sample_data/demo.jpeg",
    "models/sample_data/house_in_field_1080p.jpg",
    "models/sample_data/huggingface_cat_image.jpg",
    "models/sample_data/ILSVRC2012_val_00048736.JPEG",
]

# 32 distinct edit instructions (one per sample).
PROMPTS = [
    "Change the sky to a vivid orange sunset.",
    "Make the whole scene look like a watercolor painting.",
    "Turn the image into black and white.",
    "Add falling snow everywhere.",
    "Make it look like it is night time with a full moon.",
    "Convert the image into a pencil sketch.",
    "Add a red hot air balloon in the sky.",
    "Make the colors warm and golden like autumn.",
    "Turn the grass into sand dunes.",
    "Add thick fog to the background.",
    "Make the scene look like a van Gogh painting.",
    "Replace the season with spring blossoms.",
    "Add a rainbow across the sky.",
    "Make the picture look like an old sepia photograph.",
    "Turn it into a pixel art style image.",
    "Make everything look like it is made of glass.",
    "Give the cat a small blue wizard hat.",
    "Change the cat's fur color to white.",
    "Make the background a cozy library.",
    "Add sunglasses to the animal.",
    "Turn the image into a cartoon style drawing.",
    "Make the lighting dramatic with strong shadows.",
    "Add colorful confetti in the air.",
    "Make it look like an oil painting by Rembrandt.",
    "Turn the scene into a snowy winter landscape.",
    "Add neon lights and a cyberpunk mood.",
    "Make the image look underwater with bubbles.",
    "Change the colors to shades of purple.",
    "Add a small wooden boat in the foreground.",
    "Make it look like a mosaic of tiles.",
    "Turn the photo into a comic book panel.",
    "Add soft morning sunlight rays.",
]

NEGATIVE_PROMPT = " "
SEED_BASE = 1000
PROMPT_TEMPLATE = (
    "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, "
    "background), then explain how the user's text instruction should alter or modify the image. Generate a new "
    "image that meets the user's requirements while maintaining consistency with the original input where "
    "appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
PROMPT_TEMPLATE_DROP = 64  # QwenImageEditPipeline.prompt_template_encode_start_idx


def calculate_dimensions(target_area, ratio):
    """QwenImageEditPipeline's calculate_dimensions (multiples of 32)."""
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    return round(width / 32) * 32, round(height / 32) * 32


def sample_images(n: int):
    """n distinct real condition images: 8 square crops (4 positions x 2 scales) of each sample photo."""
    out = []
    per = 8
    for path in SAMPLE_IMAGES:
        im = Image.open(os.path.join(_REPO, path)).convert("RGB")
        w, h = im.size
        for scale in (0.9, 0.65):
            side = int(min(w, h) * scale)
            for fx, fy in ((0.5, 0.5), (0.0, 0.0), (1.0, 1.0), (0.0, 1.0)):
                x0 = int((w - side) * fx)
                y0 = int((h - side) * fy)
                out.append(im.crop((x0, y0, x0 + side, y0 + side)))
        if len(out) >= n:
            break
    while len(out) < n:  # more than 32 samples: cycle with a mirrored copy (still distinct pixels)
        out.append(out[len(out) % (per * len(SAMPLE_IMAGES))].transpose(Image.FLIP_LEFT_RIGHT))
    return out[:n]


def sample_prompts(n: int):
    return [
        PROMPTS[i % len(PROMPTS)] + ("" if i < len(PROMPTS) else f" Variation {i // len(PROMPTS)}.") for i in range(n)
    ]


def sample_seeds(n: int):
    return [SEED_BASE + i for i in range(n)]


@dataclass
class EditConfig:
    batch: int = 32
    area: int = 256 * 256
    num_inference_steps: int = 50
    true_cfg_scale: float = 4.0
    negative_prompt: str = NEGATIVE_PROMPT


@dataclass
class EncodedInputs:
    """Host-side encoded inputs for one batched image_edit call (identical for TT and HF)."""

    cfg: EditConfig
    images: list  # original PIL images
    prompts: list
    seeds: list
    width: int
    height: int
    prompt_images: list  # resized PIL images fed to the VL processor
    vae_image: torch.Tensor  # [B, 3, 1, H, W] in [-1, 1]
    cond: dict  # processor outputs for the prompts
    uncond: dict  # processor outputs for the negative prompts
    latents: torch.Tensor  # [B, S_lat, 64] packed initial noise
    img_shapes: list  # [(1, h, w), (1, h, w)] per sample
    timesteps: torch.Tensor  # [N] scheduler timesteps (0..1000)
    sigmas: torch.Tensor  # [N + 1]
    extra: dict = field(default_factory=dict)


_PROC_CACHE = {}


def load_processors(model_id: str = MODEL_ID):
    """Qwen2VLProcessor + VaeImageProcessor + FlowMatch scheduler exactly as the pipeline builds them."""
    if model_id in _PROC_CACHE:
        return _PROC_CACHE[model_id]
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.image_processor import VaeImageProcessor
    from transformers import Qwen2VLProcessor

    processor = Qwen2VLProcessor.from_pretrained(model_id, subfolder="processor")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    vae_scale_factor = 8  # 2 ** len(vae.temperal_downsample)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2)
    _PROC_CACHE[model_id] = (processor, image_processor, scheduler, vae_scale_factor)
    return _PROC_CACHE[model_id]


def _pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)


def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.15):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def encode_inputs(cfg: EditConfig, images=None, prompts=None, seeds=None, model_id: str = MODEL_ID) -> EncodedInputs:
    """QwenImageEditPipeline.__call__ steps 3-5 (preprocess, processor, prepare_latents, timesteps), on host."""
    from diffusers.utils.torch_utils import randn_tensor

    processor, image_processor, scheduler, vsf = load_processors(model_id)
    B = cfg.batch
    images = images if images is not None else sample_images(B)
    prompts = prompts if prompts is not None else sample_prompts(B)
    seeds = seeds if seeds is not None else sample_seeds(B)
    assert len(images) == len(prompts) == len(seeds) == B

    ratios = {round(im.size[0] / im.size[1], 6) for im in images}
    assert len(ratios) == 1, "a batch must share one aspect ratio (one calculated size)"
    width, height = calculate_dimensions(cfg.area, images[0].size[0] / images[0].size[1])
    multiple_of = vsf * 2
    width, height = width // multiple_of * multiple_of, height // multiple_of * multiple_of

    # QwenImageEditPipeline resizes a single PIL image to the calculated size for BOTH the VL processor and
    # the VAE; with a list of images its resize() is a no-op and the VL processor would see full-size
    # images. The inputs are therefore resized here, once, so the batched call matches the per-image one.
    images = [image_processor.resize(im, height, width) for im in images]
    prompt_images = images
    vae_image = image_processor.preprocess(prompt_images, height, width).unsqueeze(2).to(torch.float32)

    def _proc(texts):
        out = processor(
            text=[PROMPT_TEMPLATE.format(t) for t in texts], images=prompt_images, padding=True, return_tensors="pt"
        )
        return {k: v for k, v in out.items()}

    cond = _proc(prompts)
    uncond = _proc([cfg.negative_prompt] * B)

    num_channels_latents = 16
    lh, lw = 2 * (height // (vsf * 2)), 2 * (width // (vsf * 2))
    generators = [torch.Generator(device="cpu").manual_seed(s) for s in seeds]
    noise = randn_tensor((B, 1, num_channels_latents, lh, lw), generator=generators, dtype=torch.float32)
    latents = _pack_latents(noise, B, num_channels_latents, lh, lw)

    img_shapes = [(1, height // vsf // 2, width // vsf // 2), (1, height // vsf // 2, width // vsf // 2)]

    n = cfg.num_inference_steps
    sigmas = np.linspace(1.0, 1 / n, n)
    mu = calculate_shift(
        latents.shape[1],
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
    )
    scheduler.set_timesteps(n, sigmas=sigmas, mu=mu)
    return EncodedInputs(
        cfg=cfg,
        images=images,
        prompts=prompts,
        seeds=seeds,
        width=width,
        height=height,
        prompt_images=prompt_images,
        vae_image=vae_image,
        cond=cond,
        uncond=uncond,
        latents=latents,
        img_shapes=img_shapes,
        timesteps=scheduler.timesteps.clone().to(torch.float32),
        sigmas=scheduler.sigmas.clone().to(torch.float32),
    )
