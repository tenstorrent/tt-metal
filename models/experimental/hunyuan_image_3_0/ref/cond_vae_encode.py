# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Conditional VAE encode glue for I2I (host PyTorch).
#
# Mirrors upstream:
#   HunyuanImage-3.0/hunyuan_image_3/modeling_hunyuan_image_3.py
#     vae_encode()           (~2426)
#     _encode_cond_image()   (~2456)
#
# References
# ----------
#   ref/vae/encoder.py
#   ref/tokenizer/image_info.py — CondImage
#   ref/image_gen/input_instantiate.py — scatter into inputs_embeds

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .vae.encoder import Encoder, Z_CHANNELS

FFACTOR_TEMPORAL = 4


def _randn_tensor(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    if generator is not None:
        return torch.randn(shape, generator=generator, device=device, dtype=dtype)
    return torch.randn(shape, device=device, dtype=dtype)


class DiagonalGaussianDistribution:
    """Minimal posterior sampler (upstream autoencoder_kl_3d.py)."""

    def __init__(self, parameters: Tensor, *, deterministic: bool = False) -> None:
        if parameters.ndim == 3:
            dim = 2
        elif parameters.ndim in (4, 5):
            dim = 1
        else:
            raise NotImplementedError(f"Unsupported parameters.ndim={parameters.ndim}")
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=dim)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        if deterministic:
            self.std = torch.zeros_like(self.mean)

    def sample(self, generator: torch.Generator | None = None) -> Tensor:
        noise = _randn_tensor(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        return self.mean + self.std * noise


def prepare_vae_encode_input(image: Tensor, *, ffactor_temporal: int = FFACTOR_TEMPORAL) -> Tensor:
    """``[C,H,W]`` or ``[B,C,H,W]`` -> ``[B,C,T,H,W]`` (matches upstream ``vae.encode``)."""
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim == 4:
        image = image.unsqueeze(2)
    if image.shape[2] == 1:
        image = image.expand(-1, -1, ffactor_temporal, -1, -1)
    return image


def apply_vae_latent_scaling(
    latents: Tensor,
    *,
    scaling_factor: float | None = None,
    shift_factor: float | None = None,
) -> Tensor:
    if shift_factor:
        latents = latents.sub(shift_factor)
    if scaling_factor:
        latents = latents.mul(scaling_factor)
    return latents


def vae_encode_image(
    encoder: Encoder,
    image: Tensor,
    *,
    scaling_factor: float | None = None,
    shift_factor: float | None = None,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """Encode one conditional image -> ``(timesteps, latents)`` with ``t=0``.

    Returns:
        timesteps: ``[B]`` zeros
        latents:   ``[B, C, H, W]`` scaled VAE latent
    """
    x = prepare_vae_encode_input(image).to(dtype=dtype)
    with torch.no_grad():
        h = encoder(x.float())
        latents = DiagonalGaussianDistribution(h).sample(generator)
        if latents.shape[2] == 1:
            latents = latents.squeeze(2)
        latents = apply_vae_latent_scaling(
            latents,
            scaling_factor=scaling_factor,
            shift_factor=shift_factor,
        )
    t = torch.zeros(latents.shape[0], device=latents.device, dtype=latents.dtype)
    return t, latents


def _cond_vae_tensor(cond_image: Any) -> Tensor:
    from .tokenizer.image_info import CondImage

    if isinstance(cond_image, CondImage):
        return cond_image.vae_image
    return cond_image


def _first_section_type(batch_cond_images: list[list[Any]]) -> str:
    from .tokenizer.image_info import CondImage

    first = batch_cond_images[0][0]
    if isinstance(first, CondImage):
        return first.section_type
    return getattr(first, "section_type", "cond_vae_image")


@dataclass
class CondVaeEncodeOutput:
    """Encoded conditional VAE latents per upstream ``_encode_cond_image``."""

    cond_vae_images: Tensor | list[Tensor] | list[list[Tensor]] | None
    cond_timesteps: Tensor | list[Tensor] | None


def encode_cond_images(
    batch_cond_images: list[list[Any]] | None,
    encoder: Encoder,
    *,
    cfg_factor: int = 1,
    scaling_factor: float | None = None,
    shift_factor: float | None = None,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float32,
) -> CondVaeEncodeOutput:
    """Mirror ``HunyuanImage3ForCausalMM._encode_cond_image`` (VAE branch only)."""
    if batch_cond_images is None or len(batch_cond_images) == 0 or len(batch_cond_images[0]) == 0:
        return CondVaeEncodeOutput(cond_vae_images=None, cond_timesteps=None)

    section_type = _first_section_type(batch_cond_images)
    if section_type not in ("cond_vae_image", "cond_joint_image"):
        return CondVaeEncodeOutput(cond_vae_images=None, cond_timesteps=None)

    batch_cond_vae_images: list[list[Tensor]] = []
    batch_cond_t: list[list[Tensor]] = []
    for cond_images in batch_cond_images:
        cond_vae_image_list: list[Tensor] = []
        cond_t_list: list[Tensor] = []
        for cond_image in cond_images:
            vae_image = _cond_vae_tensor(cond_image)
            cond_t_i, cond_vae_i = vae_encode_image(
                encoder,
                vae_image,
                scaling_factor=scaling_factor,
                shift_factor=shift_factor,
                generator=generator,
                dtype=dtype,
            )
            cond_vae_image_list.append(cond_vae_i.squeeze(0))
            cond_t_list.append(cond_t_i)
        batch_cond_vae_images.append(cond_vae_image_list)
        batch_cond_t.append(cond_t_list)

    if all(len(items) == 1 for items in batch_cond_vae_images) and all(
        items[0].shape == batch_cond_vae_images[0][0].shape for items in batch_cond_vae_images
    ):
        cond_vae_images: Tensor | list = torch.stack([items[0] for items in batch_cond_vae_images], dim=0)
        cond_t: Tensor | list = torch.cat([items[0] for items in batch_cond_t], dim=0)
        if cfg_factor > 1:
            cond_t = cond_t.repeat(cfg_factor)
            cond_vae_images = cond_vae_images.repeat(cfg_factor, 1, 1, 1)
    else:
        cond_t = [torch.cat(item, dim=0) for item in batch_cond_t]
        cond_vae_images = []
        for items in batch_cond_vae_images:
            if all(items[0].shape == item.shape for item in items):
                cond_vae_images.append(torch.stack(items, dim=0))
            else:
                cond_vae_images.append(items)
        if cfg_factor > 1:
            cond_t = cond_t * cfg_factor
            cond_vae_images = cond_vae_images * cfg_factor

    return CondVaeEncodeOutput(cond_vae_images=cond_vae_images, cond_timesteps=cond_t)


def latent_channels() -> int:
    return Z_CHANNELS
