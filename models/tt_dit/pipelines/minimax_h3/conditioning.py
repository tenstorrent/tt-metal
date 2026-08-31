# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3 FL2VA keyframe conditioning.

A keyframe is consumed twice by H3: once by the VLM, which turns it into
semantic tokens inside the text stream, and once here by the video VAE, which
turns it into pixel-accurate conditioning rows anchored to the first or last
latent frame. This module is the second path.

Four details are part of the released model's numerical contract and reproduce
nothing without them:

- pixels are normalized with **ImageNet** statistics, not to ``[-1, 1]``;
- the VAE posterior is **sampled**, under its own generator seeded to 42,
  independent of the request seed;
- the sampled latent is **rounded through float16** before normalization, which
  discards roughly half the mantissa of every conditioning latent;
- the conditioning noise is the **first** draw off the request generator, ahead
  of the video and then the audio noise.

The VAE forward is injected rather than imported so this stays host-testable and
so a device encoder can supply the sampled latent directly.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch

from .packing import MINIMAX_H3_KEYFRAME_ENCODE_SEED, patchify_video_latents

# ImageNet statistics. H3 conditions on VLM-style normalized pixels, not on the
# [-1, 1] range most video VAEs take.
MINIMAX_H3_PIXEL_MEAN = (0.485, 0.456, 0.406)
MINIMAX_H3_PIXEL_STD = (0.229, 0.224, 0.225)


def normalize_keyframe_pixels(image, device: torch.device | str | None = None) -> torch.Tensor:
    """PIL keyframe to ``(1, 3, 1, H, W)`` ImageNet-normalized fp32 pixels."""
    pixel_mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device=device).view(1, -1, 1, 1, 1)
    pixel_std = torch.tensor(MINIMAX_H3_PIXEL_STD, device=device).view(1, -1, 1, 1, 1)
    pixels = torch.from_numpy(np.array(image)).to(device).permute(2, 0, 1)[None, :, None]
    return (pixels.to(torch.float32).div(255.0) - pixel_mean) / pixel_std


def raw_keyframe_pixels(image, device: torch.device | str | None = None) -> torch.Tensor:
    """PIL keyframe to ``(1, 3, 1, H, W)`` raw **uint8** pixels.

    For an encoder built with ``pixel_norm`` -- the normalization above is folded into its
    ``conv_in``, so the bytes cross PCIe at a quarter of fp32 and the host runs no float pass.
    """
    return torch.from_numpy(np.array(image)).to(device).permute(2, 0, 1)[None, :, None]


def sample_posterior(moments: torch.Tensor, seed: int = MINIMAX_H3_KEYFRAME_ENCODE_SEED) -> torch.Tensor:
    """Sample a DiagonalGaussian from concatenated ``[mean, logvar]`` moments.

    Seeded independently of the request generator, so a keyframe encodes to the
    same latent regardless of the request seed. Taking the mean instead of
    sampling does not reproduce the released model.
    """
    mean, logvar = moments.chunk(2, dim=1)
    logvar = logvar.clamp(-30.0, 20.0)
    generator = torch.Generator().manual_seed(int(seed))
    noise = torch.randn(mean.shape, generator=generator, dtype=torch.float32).to(mean.device)
    return mean + (0.5 * logvar).exp() * noise.to(mean.dtype)


def keyframe_condition_rows(
    latents: torch.Tensor,
    latents_mean: Sequence[float],
    latents_std: Sequence[float],
    patch_size: tuple[int, int, int] = (1, 2, 2),
) -> torch.Tensor:
    """Sampled keyframe latent to packed conditioning rows, ``[n, C * prod(patch)]`` fp32.

    The float16 round trip is deliberate: the reference rounds the sampled latent
    to fp16 before normalizing, keeping about 11 bits of each value.
    """
    channels = latents.shape[1]
    mean = torch.tensor(tuple(latents_mean), dtype=torch.float32).view(1, channels, 1, 1, 1)
    std = torch.tensor(tuple(latents_std), dtype=torch.float32).view(1, channels, 1, 1, 1)
    latents = latents.to(torch.float16).float().cpu()
    return patchify_video_latents((latents - mean) / std, patch_size)


def encode_keyframes(
    images: Sequence,
    encode_clip: Callable[[torch.Tensor], torch.Tensor],
    latents_mean: Sequence[float],
    latents_std: Sequence[float],
    patch_size: tuple[int, int, int] = (1, 2, 2),
    device: torch.device | str | None = None,
    seed: int = MINIMAX_H3_KEYFRAME_ENCODE_SEED,
    raw_pixels: bool = False,
) -> torch.Tensor:
    """Encode prepared keyframes into packed conditioning rows, in packed order.

    ``encode_clip`` maps ``(1, 3, 1, H, W)`` normalized pixels to concatenated
    ``[mean, logvar]`` moments. A keyframe is a single frame, so the VAE's
    17-frame temporal chunking never applies -- the spatial encoder alone is what
    the released model conditions on.

    ``raw_pixels`` hands ``encode_clip`` raw uint8 instead: only for a device VAE
    built with ``pixel_norm``, whose conv_in carries the normalization itself.
    """
    to_pixels = raw_keyframe_pixels if raw_pixels else normalize_keyframe_pixels
    rows = [
        keyframe_condition_rows(
            sample_posterior(encode_clip(to_pixels(image, device=device)), seed=seed),
            latents_mean,
            latents_std,
            patch_size,
        )
        for image in images
    ]
    return torch.cat(rows)


def keyframe_condition_noise(
    condition_latent_shapes: Sequence[tuple[int, int, int]],
    latent_channels: int,
    patch_size: tuple[int, int, int] = (1, 2, 2),
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Draw the noise the conditioning rows are mixed with, one draw per condition.

    These are the **first** draws off the request generator, before the video and
    audio noise; the order is part of what a generator reproduces.

    This follows the diffusers contract, which draws at each condition's own
    latent shape. sglang instead re-seeds per condition and draws at
    ``target_latent_t + cond_frames`` before slicing, which is a different
    stream; diffusers wins as the HF/MiniMax-authored path.
    """
    rows = []
    for num_latent_frames, latent_height, latent_width in condition_latent_shapes:
        noise = torch.randn(
            (1, latent_channels, num_latent_frames, latent_height, latent_width),
            generator=generator,
            dtype=dtype,
        )
        rows.append(patchify_video_latents(noise, patch_size))
    return torch.cat(rows)


# Noise augmentation is not implemented here. It is the rectified-flow
# forward process at t = noise_aug, which is exactly MiniMaxH3Scheduler.scale_noise,
# and the reference calls that same method -- so callers must use it:
#
#     scheduler.scale_noise(condition_rows, MINIMAX_H3_KEYFRAME_NOISE_AUG, noise)
#
# A second copy of `t*x0 + (1-t)*noise` computing `1 - t` in Python double instead
# of the scheduler's float32 drifts by 2.4e-7.
