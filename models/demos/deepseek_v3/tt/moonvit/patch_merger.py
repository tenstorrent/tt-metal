# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
2x2 spatial patch merger.

Mirrors the HF `patch_merger` free function in modeling_kimi_k25.py.
For each image with grid (H, W):
    seq    : (H*W, D)
    view   : (H/kh, kh, W/kw, kw, D)
    permute: (H/kh, W/kw, kh, kw, D)
    view   : (H/kh * W/kw, kh*kw, D)

HF returns a Python list of per-image tensors (3D). For the v1
TT-Metal pipeline we concatenate the per-image outputs and optionally
flatten the (kh*kw, D) tail into a single `merge_dim` (= kh*kw*D)
that the projector consumes directly.

The merger has no compute — pure shape ops. Doing it on host keeps
v1 simple; a device-side variant via ttnn.reshape/ttnn.permute is a
Phase 4 productionization concern.

References:
  - `patch_merger` in modeling_kimi_k25.py.
"""
from __future__ import annotations

from typing import List, Tuple

import torch

from models.demos.deepseek_v3.tt.moonvit.pos_emb import GridHws, _grid_hws_to_list


def patch_merger_per_image(
    x: torch.Tensor,
    grid_hws: GridHws,
    merge_kernel_size: Tuple[int, int] = (2, 2),
) -> List[torch.Tensor]:
    """Match HF patch_merger exactly: return one 3D tensor per image.

    Args:
        x: (L, D) packed tokens, where L = sum(H_i * W_i).
        grid_hws: per-image (H, W) shapes.
        merge_kernel_size: spatial merge factor (kh, kw). MoonViT default (2, 2).

    Returns:
        list of tensors, one per image, each (H_i/kh * W_i/kw, kh*kw, D).
    """
    if x.ndim != 2:
        raise ValueError(f"x must be 2D (L, D); got shape {tuple(x.shape)}")
    shapes = _grid_hws_to_list(grid_hws)
    kh, kw = merge_kernel_size
    d_model = x.shape[-1]

    outputs: List[torch.Tensor] = []
    cursor = 0
    for h, w in shapes:
        if h % kh != 0 or w % kw != 0:
            raise ValueError(f"grid ({h}, {w}) is not divisible by merge_kernel_size {(kh, kw)}")
        seq = x[cursor : cursor + h * w]
        new_h, new_w = h // kh, w // kw
        reshaped = seq.view(new_h, kh, new_w, kw, d_model)
        reshaped = reshaped.permute(0, 2, 1, 3, 4).contiguous()
        merged = reshaped.view(new_h * new_w, kh * kw, d_model)
        outputs.append(merged)
        cursor += h * w
    return outputs


def patch_merger(
    x: torch.Tensor,
    grid_hws: GridHws,
    merge_kernel_size: Tuple[int, int] = (2, 2),
    flatten: bool = True,
) -> torch.Tensor:
    """Same as `patch_merger_per_image` but returns a single concatenated tensor.

    Args:
        x: (L, D) packed tokens.
        grid_hws: per-image (H, W) shapes.
        merge_kernel_size: spatial merge factor.
        flatten: if True (default), flatten the kh*kw axis into the channel
            axis so the output is (L_new_total, kh*kw*D). This is the format
            the projector's first Linear consumes (input dim = 4608 for
            MoonViT defaults). If False, output is (L_new_total, kh*kw, D).

    Returns:
        single 2D or 3D tensor depending on `flatten`.
    """
    per_image = patch_merger_per_image(x, grid_hws, merge_kernel_size)
    out = torch.cat(per_image, dim=0)
    if flatten:
        l_new, k, d = out.shape
        out = out.reshape(l_new, k * d)
    return out
