# SPDX-FileCopyrightText: © 2026 Qwen Image 2.1 contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU-only, unlearned metadata for standalone single-image generation.

The pinned Diffusers scheduler and rotary module construct these tensors from
configuration and request dimensions. No weights or captured activations are read.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Rope
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift


def build_metadata(checkpoint: Path, prefix_tokens: int, height: int, width: int, steps: int) -> dict:
    """Build batch-one text-to-image metadata with no condition images or padding."""
    for name, value in (("prefix_tokens", prefix_tokens), ("height", height), ("width", width), ("steps", steps)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if height < 32 or width < 32 or height % 32 or width % 32:
        raise ValueError("height and width must be positive multiples of 32")
    if steps < 2:
        raise ValueError(
            "the pinned terminal-shift scheduler requires at least two total steps; truncate a longer schedule for a one-step diagnostic"
        )
    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint / "transformer/config.json").read_text())
    if config.get("_class_name") != "QwenImage21Transformer2DModel":
        raise ValueError("checkpoint must contain QwenImage21Transformer2DModel")
    if config.get("patch_size", 1) != 1 or config.get("in_channels", 64) != 64:
        raise ValueError("only the unpatched 64-channel Qwen-Image 2.1 layout is supported")
    axes = config.get("axes_dims_rope", [16, 56, 56])
    if axes != [16, 56, 56]:
        raise ValueError("checkpoint rotary axes differ from the supported 128-channel heads")
    latent_height, latent_width = height // 16, width // 16
    if prefix_tokens >= 8192 or max(latent_height, latent_width) > 2048:
        raise ValueError("request exceeds the positional embedding table")
    latent_tokens = latent_height * latent_width
    target_mask = torch.cat((torch.zeros(prefix_tokens, dtype=torch.bool), torch.ones(latent_tokens, dtype=torch.bool)))
    # This module contains only deterministic frequency tables, no learned parameters.
    rope = QwenImage21Rope(theta=10000, axes_dim=axes)(
        [(1, latent_height, latent_width)], target_mask, device=torch.device("cpu")
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        checkpoint, subfolder="scheduler", local_files_only=True
    )
    mu = calculate_shift(
        latent_tokens,
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
    )
    scheduler.set_timesteps(steps, device="cpu", sigmas=np.linspace(1.0, 1.0 / steps, steps), mu=mu)
    if len(scheduler.sigmas) != steps + 1:
        raise ValueError("scheduler must provide one terminal sigma")
    if not torch.isfinite(scheduler.sigmas).all():
        raise ValueError("scheduler returned nonfinite sigmas")
    # Match the pipeline's two BF16 rounding operations exactly.
    timesteps = (scheduler.timesteps.to(torch.bfloat16) / 1000).tolist()
    return {
        "schedule": {
            "steps": steps,
            "latent_tokens": latent_tokens,
            "mu": mu,
            "sigmas": scheduler.sigmas.tolist(),
            "timesteps": timesteps,
        },
        "target_mask": target_mask,
        "rope": rope,
        "segments": [(0, prefix_tokens, True)],
        "key_valid": None,
    }
