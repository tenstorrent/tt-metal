# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

##########################################################################
# Adapted from BEVFormer (https://github.com/fundamentalvision/BEVFormer).
# Original work Copyright (c) OpenMMLab.
# Modified by Zhiqi Li.
# Licensed under the Apache License, Version 2.0.
##########################################################################

import numpy as np
import torch
from typing import Union
from torch import Tensor


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def multi_scale_deformable_attn_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """CPU version of multi-scale deformable attention (mmcv compatible).

    Args:
        value: (bs, num_keys, num_heads, embed_dims//num_heads)
        value_spatial_shapes: (num_levels, 2) - spatial shapes [H, W] for each level
        sampling_locations: (bs, num_queries, num_heads, num_levels, num_points, 2)
        attention_weights: (bs, num_queries, num_heads, num_levels, num_points)

    Returns:
        output: (bs, num_queries, embed_dims)
    """
    import torch.nn.functional as F

    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        H_ = int(H_.item())
        W_ = int(W_.item())
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, H_, W_)
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(bs, num_heads * embed_dims, num_queries)
    )
    return output.transpose(1, 2).contiguous()


def limit_period(
    val: Union[np.ndarray, Tensor], offset: float = 0.5, period: float = np.pi
) -> Union[np.ndarray, Tensor]:
    limited_val = val - torch.floor(val / period + offset) * period
    return limited_val
