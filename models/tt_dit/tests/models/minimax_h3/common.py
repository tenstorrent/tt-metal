# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the MiniMax-H3 bringup tests."""

import json
import math
import os

import numpy as np
import pytest
import torch
from PIL import Image

# The VAE's fixed work units: the encoder always runs (17, 256, 256) tiles and the decoder
# always (7, 16, 16) latent chunks, so every gate that builds one uses these shapes.
TILE = 256
CLIP_FRAMES = 17
LATENT_TILE = 16
DECODE_LATENT_FRAMES = 7


def weights_subdir(subfolder: str) -> str | None:
    base = os.environ.get("MINIMAX_H3_DIFFUSERS_DIR", "/data/cglagovich/MiniMax-H3-diffusers")
    candidate = os.path.join(base, subfolder)
    return candidate if os.path.isfile(os.path.join(candidate, "config.json")) else None


def load_config(weights_dir: str) -> dict:
    return {
        k: v
        for k, v in json.loads(open(os.path.join(weights_dir, "config.json")).read()).items()
        if not k.startswith("_")
    }


def _reference_class(name: str):
    pytest.importorskip("diffusers", reason="pinned diffusers reference not installed")
    from diffusers.models.autoencoders import autoencoder_kl_minimax_h3 as ref

    return getattr(ref, name)


def random_encoder_state(config: dict) -> dict:
    """State dict from a randomly-initialised reference encoder -- fast, and enough for timing."""
    cls = _reference_class("MiniMaxH3VideoEncoder3d")
    module = cls(
        in_channels=3,
        out_channels=2 * config["latent_channels"],
        block_out_channels=tuple(config["block_out_channels"]),
        layers_per_block=config["layers_per_block"],
        spatial_downsample_factors=tuple(config["spatial_downsample_factors"]),
        temporal_downsample_factors=tuple(config["temporal_downsample_factors"]),
        norm_num_groups=config["norm_num_groups"],
        norm_eps=config["norm_eps"],
        spatial_padding_mode=config["spatial_padding_mode"],
    )
    return dict(module.state_dict())


def random_decoder_state(config: dict, *, num_layers: int | None = None) -> dict:
    """Likewise for the 36-layer decoder: 2.4 B random parameters beat a 10.4 GB read.

    ``num_layers`` overrides the config depth, for gates that only need the ops exercised
    rather than the full 2.4 B parameters materialised.
    """
    cls = _reference_class("MiniMaxH3VideoViTDecoder3d")
    module = cls(
        in_channels=config["latent_channels"],
        out_channels=config["out_channels"],
        patch_size=16,
        patch_size_t=4,
        num_layers=config["decoder_num_layers"] if num_layers is None else num_layers,
        num_attention_heads=config["decoder_num_attention_heads"],
        attention_head_dim=config["decoder_attention_head_dim"],
        num_register_tokens=config["decoder_num_register_tokens"],
        ffn_mult=config["decoder_ffn_mult"],
        rope_theta=config["decoder_rope_theta"],
        rope_dim_ratio=config["decoder_rope_dim_ratio"],
        norm_eps=config["decoder_norm_eps"],
    )
    return dict(module.state_dict())


def psnr(reference: torch.Tensor, test: torch.Tensor) -> float:
    """Peak signal-to-noise ratio in dB, with the peak taken from the reference's own range.

    The roundtrip quality gates use this rather than PCC alone: PCC per component says the
    port matches the reference, but a faint vignette or a dull high end sails through a
    0.99 PCC and shows up as a PSNR drop.
    """
    mse = torch.mean((reference.float() - test.float()) ** 2).item()
    if mse == 0.0:
        return float("inf")
    peak = reference.abs().max().item()
    return float("inf") if peak == 0.0 else 20.0 * math.log10(peak) - 10.0 * math.log10(mse)


def create_fractal_image(width: int, height: int) -> Image.Image:
    """A Mandelbrot escape-time image, the repo's existing convention for an I2V seed.

    Copied from `tests/models/wan2_2/test_pipeline_wan_i2v.py`, and it is the right tool for the
    *discriminating* case for one reason: a fractal is content the model would never generate for this
    prompt, so "decoded frame 0 resembles the keyframe" cannot be satisfied by a pipeline that ignores
    the keyframe. See `test_fl2va_follows_the_keyframe`.
    """
    c = np.linspace(-2.0, 1.0, width)[None, :] + 1j * np.linspace(-1.5, 1.5, height)[:, None]
    z = np.zeros_like(c)
    img = np.zeros(c.shape, dtype=np.uint8)
    for i in range(32):
        z = z * z + c
        img[(img == 0) & (np.abs(z) > 2)] = 255 - 8 * i
    return Image.fromarray(np.dstack((img, np.roll(img, width // 10, 1), np.roll(img, height // 10, 0))), "RGB")


def randomize_norm_weights(module: torch.nn.Module, *, scale: float = 0.5) -> torch.nn.Module:
    """Give every `nn.RMSNorm` in `module` a non-trivial affine weight, in place.

    `nn.RMSNorm` initialises `weight` to all ones, so a reference model built with random weights
    (rather than loaded from the checkpoint) has an *identity* affine in every norm. That makes the
    norm weights invisible to a PCC comparison: a port that loaded the wrong norm weight, swapped two
    of them, or never loaded them at all would still match the reference exactly.

    MiniMax-H3 is full of RMSNorms -- `norm1`, `norm2`, the per-head `norm_q`/`norm_k`, the refiner's
    `final_norm` -- so this blind spot covers most of the model's non-matmul parameters. Measured on
    the token refiner at real dims, randomizing the norms moves "norm weights never loaded" from PCC
    1.000000 (undetectable) to 0.887, and "norm1/norm2 swapped" from 1.000000 to 0.986.

    Call this on the torch reference *before* taking its `state_dict`, so the TT module under test
    loads the same non-trivial values.
    """
    for submodule in module.modules():
        if isinstance(submodule, torch.nn.RMSNorm) and submodule.weight is not None:
            submodule.weight.data = 1.0 + scale * torch.randn_like(submodule.weight.data)
    return module
