# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Assemble multi-image vision inputs for TtMolmo2Model.forward_prefill.

The HF processor emits ``image_token_pooling`` indices relative to *each image's own crops*;
HF ``Molmo2Model.build_batched_images`` adds a per-image patch offset before pooling.
``forward_prefill`` expects indices into the concatenated crops of the whole request, so the
same offsets must be applied here — otherwise every image after the first pools patches of
image 0.
"""

from typing import List, Optional

import torch


def offset_image_token_pooling(
    image_token_pooling: torch.Tensor,  # [sum_i N_pooled_i, k_pool], per-image-relative indices
    image_num_crops: torch.Tensor,  # [num_images]
    image_grids: torch.Tensor,  # [num_images, 4]
    n_patches: int,  # patches per crop (pixel_values.shape[1])
) -> torch.Tensor:
    """Single-example port of the offset step of HF ``build_batched_images``."""
    pooled_per_image = image_grids[:, :2].prod(dim=1) + image_grids[:, 2:].prod(dim=1)
    assert int(pooled_per_image.sum()) == image_token_pooling.shape[0], (
        f"image_grids describe {int(pooled_per_image.sum())} pooled patches, "
        f"image_token_pooling has {image_token_pooling.shape[0]}"
    )
    out = image_token_pooling.clone()
    row, patch_offset = 0, 0
    for n_pooled, n_crops in zip(pooled_per_image.tolist(), image_num_crops.tolist()):
        rows = out[row : row + n_pooled]
        out[row : row + n_pooled] = torch.where(rows >= 0, rows + patch_offset, rows)
        row += n_pooled
        patch_offset += n_crops * n_patches
    return out


def concat_image_inputs(pixel_values: List[torch.Tensor], token_pooling: List[torch.Tensor]):
    """Concatenate per-image (pixel_values [n_crops, n_patches, D], pooling [N_pooled, k]) pairs,
    shifting each image's pooling indices past the crops of the preceding images."""
    assert len(pixel_values) == len(token_pooling), f"{len(pixel_values)} images vs {len(token_pooling)} poolings"
    pooled, patch_offset = [], 0
    for pv, pool in zip(pixel_values, token_pooling):
        pooled.append(torch.where(pool >= 0, pool + patch_offset, pool))
        patch_offset += pv.shape[0] * pv.shape[1]
    return torch.cat(pixel_values, dim=0), torch.cat(pooled, dim=0)


def vision_inputs_from_processor(proc_out) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """(pixel_values [1, n_crops, n_patches, D], pooled_patches_idx [1, N_pooled, k]) for one example."""
    pv = proc_out.get("pixel_values")
    if pv is not None:
        pool = offset_image_token_pooling(
            proc_out["image_token_pooling"], proc_out["image_num_crops"], proc_out["image_grids"], pv.shape[1]
        )
        return pv.float().unsqueeze(0), pool.unsqueeze(0)
    pv = proc_out.get("pixel_values_videos")
    if pv is not None:
        return pv.float().unsqueeze(0), proc_out["video_token_pooling"].unsqueeze(0)
    return None, None
