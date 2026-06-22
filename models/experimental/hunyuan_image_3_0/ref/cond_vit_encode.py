# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Conditional SigLIP2 + aligner pack for I2I (host PyTorch).
#
# Mirrors upstream:
#   HunyuanImage-3.0/hunyuan_image_3/modeling_hunyuan_image_3.py
#     _encode_cond_image() ViT branch      (~2514)
#     _prepare_vit_image_kwargs()          (~2539)

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


def _first_section_type(batch_cond_images: list[list[Any]]) -> str:
    from .tokenizer.image_info import CondImage

    first = batch_cond_images[0][0]
    if isinstance(first, CondImage):
        return first.section_type
    return getattr(first, "section_type", "cond_vit_image")


def _vit_tensor(cond_image: Any) -> Tensor:
    from .tokenizer.image_info import CondImage

    if isinstance(cond_image, CondImage):
        if cond_image.section_type == "cond_joint_image":
            return cond_image.vit_image
        return cond_image.vit_image
    return cond_image


def _cond_vit_tensor(cond_image: Any) -> Tensor:
    return _vit_tensor(cond_image)


@dataclass
class CondVitEncodeOutput:
    """Packed conditional ViT pixel tensors + encoder kwargs."""

    cond_vit_images: list[Tensor] | None
    cond_vit_image_kwargs: dict[str, list[Tensor]] | None


def prepare_cond_vit_images(
    batch_cond_images: list[list[Any]] | None,
    *,
    cfg_factor: int = 1,
    dtype: torch.dtype = torch.float32,
) -> list[Tensor] | None:
    """Mirror ``_encode_cond_image`` ViT branch (stack per batch row, no forward)."""
    if batch_cond_images is None or len(batch_cond_images) == 0 or len(batch_cond_images[0]) == 0:
        return None

    section_type = _first_section_type(batch_cond_images)
    if section_type not in ("cond_vit_image", "cond_joint_image"):
        return None

    cond_vit_images: list[Tensor] = []
    for cond_images in batch_cond_images:
        cond_vit_image_list = [_cond_vit_tensor(cond_image) for cond_image in cond_images]
        cond_vit_images.append(torch.stack(cond_vit_image_list, dim=0).to(dtype=dtype))

    if cfg_factor > 1:
        cond_vit_images = cond_vit_images * cfg_factor
    return cond_vit_images


def prepare_cond_vit_image_kwargs(
    batch_cond_images: list[list[Any]] | None,
    *,
    cfg_factor: int = 1,
) -> dict[str, list[Tensor]] | None:
    """Mirror ``_prepare_vit_image_kwargs`` (spatial_shapes + attention_mask lists)."""
    if batch_cond_images is None or len(batch_cond_images) == 0 or len(batch_cond_images[0]) == 0:
        return None

    first_vit = _vit_tensor(batch_cond_images[0][0])
    if not hasattr(first_vit, "vision_encoder_kwargs") or not first_vit.vision_encoder_kwargs:
        return None

    cond_vit_image_kwargs: dict[str, list[Tensor]] = {"spatial_shapes": [], "attention_mask": []}
    for cond_images in batch_cond_images:
        cond_vit_image_kwargs["spatial_shapes"].append(
            torch.stack([_vit_tensor(cond_image).vision_encoder_kwargs["spatial_shapes"] for cond_image in cond_images])
        )
        cond_vit_image_kwargs["attention_mask"].append(
            torch.stack(
                [_vit_tensor(cond_image).vision_encoder_kwargs["pixel_attention_mask"] for cond_image in cond_images]
            )
        )
    if cfg_factor > 1:
        cond_vit_image_kwargs["spatial_shapes"] = cond_vit_image_kwargs["spatial_shapes"] * cfg_factor
        cond_vit_image_kwargs["attention_mask"] = cond_vit_image_kwargs["attention_mask"] * cfg_factor
    return cond_vit_image_kwargs


def encode_cond_vit_images(
    batch_cond_images: list[list[Any]] | None,
    *,
    cfg_factor: int = 1,
    dtype: torch.dtype = torch.float32,
) -> CondVitEncodeOutput:
    """Pack cond ViT tensors and kwargs for ``instantiate_vit_image_tokens``."""
    return CondVitEncodeOutput(
        cond_vit_images=prepare_cond_vit_images(batch_cond_images, cfg_factor=cfg_factor, dtype=dtype),
        cond_vit_image_kwargs=prepare_cond_vit_image_kwargs(batch_cond_images, cfg_factor=cfg_factor),
    )
