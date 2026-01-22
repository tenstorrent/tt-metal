# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from typing import List


def multi_scale_deformable_attn(value, value_spatial_shapes, sampling_locations, attention_weights, device):
    """Pure TTNN implementation of multi-scale deformable attention.

    Args:
        value: Value tensor with shape (bs, num_value, num_heads, embed_dims)
        value_spatial_shapes: Spatial shapes tensor with shape (num_levels, 2)
        sampling_locations: Sampling locations with shape (bs, num_query, num_heads, num_levels, num_points, 2)
        attention_weights: Attention weights with shape (bs, num_query, num_heads, num_levels, num_points)
        device: TTNN device

    Returns:
        Output tensor with shape (bs, num_query, num_heads * embed_dims)
    """
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = []
    value_list.append(value)
    sampling_locations = ttnn.to_layout(sampling_locations, layout=ttnn.ROW_MAJOR_LAYOUT)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []

    for level, (H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = value_list[level]
        value_l_ = ttnn.reshape(value_l_, [value_l_.shape[0], value_l_.shape[1], value_l_.shape[2] * value_l_.shape[3]])
        value_l_ = ttnn.permute(value_l_, (0, 2, 1))

        value_l_ = ttnn.reshape(value_l_, [bs * num_heads, embed_dims, int(H_.item()), int(W_.item())])

        sampling_grid_l_ = sampling_grids[:, :, :, level]
        sampling_grid_l_ = ttnn.permute(sampling_grid_l_, (0, 2, 1, 3, 4))

        sampling_grid_l_ = ttnn.reshape(
            sampling_grid_l_,
            [
                sampling_grid_l_.shape[0] * sampling_grid_l_.shape[1],
                sampling_grid_l_.shape[2],
                sampling_grid_l_.shape[3],
                sampling_grid_l_.shape[4],
            ],
        )

        value_l_ = ttnn.permute(value_l_, (0, 2, 3, 1))
        value_l_ = ttnn.to_layout(value_l_, layout=ttnn.ROW_MAJOR_LAYOUT)
        sampling_value_l_ = ttnn.grid_sample(value_l_, sampling_grid_l_)
        ttnn.deallocate(value_l_)
        ttnn.deallocate(sampling_grid_l_)
        sampling_value_l_ = ttnn.permute(sampling_value_l_, (0, 3, 1, 2))

        sampling_value_list.append(sampling_value_l_)

    # Reshape attention_weights AFTER the loop (matching reference implementation)
    attention_weights = ttnn.permute(attention_weights, (0, 2, 1, 3, 4))
    attention_weights = ttnn.reshape(attention_weights, [bs * num_heads, 1, num_queries, num_levels * num_points])

    output = ttnn.stack(sampling_value_list, -2)
    ttnn.deallocate(sampling_grids)
    output = ttnn.reshape(
        output, [output.shape[0], output.shape[1], output.shape[2], output.shape[3] * output.shape[4]]
    )

    output = output * attention_weights
    output = ttnn.reallocate(output)
    for val in sampling_value_list:
        ttnn.deallocate(val)
    output = ttnn.sum(output, 3)
    output = ttnn.reallocate(output)
    output = ttnn.reshape(output, [bs, num_heads * embed_dims, num_queries])
    output = ttnn.reallocate(output)
    output = ttnn.permute(output, (0, 2, 1))

    ttnn.deallocate(attention_weights)
    ttnn.deallocate(sampling_value_l_)
    ttnn.deallocate(value)

    return output


def inverse_sigmoid(x: ttnn.Tensor, eps: float = 1e-5) -> ttnn.Tensor:
    """Compute inverse sigmoid using TTNN operations.

    inverse_sigmoid(x) = log(x / (1 - x))

    Args:
        x: Input tensor with values in [0, 1].
        eps: Small epsilon for numerical stability.

    Returns:
        Inverse sigmoid of input tensor.
    """
    x = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT)
    x = ttnn.clamp(x, min=0, max=1)
    x1 = ttnn.clamp(x, min=eps)

    # Create ones tensor of same shape for (1 - x) computation
    if len(x.shape) == 3:
        x_temp = ttnn.ones(shape=[x.shape[0], x.shape[1], x.shape[2]], layout=ttnn.TILE_LAYOUT, device=x.device())
    else:
        x_temp = ttnn.ones(
            shape=[x.shape[0], x.shape[1], x.shape[2], x.shape[3]], layout=ttnn.TILE_LAYOUT, device=x.device()
        )

    x_temp = x_temp - x
    x2 = ttnn.clamp(x_temp, min=eps)

    return ttnn.log(ttnn.div(x1, x2))


def bbox_xyxy_to_cxcywh(bbox: ttnn.Tensor) -> ttnn.Tensor:
    """Convert bounding boxes from xyxy to cxcywh format using TTNN operations.

    Args:
        bbox: Bounding boxes in xyxy format (xmin, ymin, xmax, ymax).
              Shape: (..., 4)

    Returns:
        Bounding boxes in cxcywh format (cx, cy, w, h).
        Shape: (..., 4)
    """
    x0 = bbox[..., 0:1]
    y0 = bbox[..., 1:2]
    x1 = bbox[..., 2:3]
    y1 = bbox[..., 3:4]

    cx = ttnn.div(ttnn.add(x0, x1), 2.0)
    cy = ttnn.div(ttnn.add(y0, y1), 2.0)
    w = ttnn.subtract(x1, x0)
    h = ttnn.subtract(y1, y0)

    result = ttnn.concat([cx, cy, w, h], dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
    return result


def denormalize_2d_bbox(bboxes: torch.Tensor, pc_range: List[float]) -> torch.Tensor:
    """Denormalize bounding boxes from [0, 1] to real-world coordinates."""
    new_bboxes = bboxes.clone()
    new_bboxes[..., 0::2] = bboxes[..., 0::2] * (pc_range[3] - pc_range[0]) + pc_range[0]
    new_bboxes[..., 1::2] = bboxes[..., 1::2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    return new_bboxes


def denormalize_2d_pts(pts: torch.Tensor, pc_range: List[float]) -> torch.Tensor:
    """Denormalize points from [0, 1] to real-world coordinates."""
    new_pts = pts.clone()
    new_pts[..., 0] = pts[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
    new_pts[..., 1] = pts[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
    return new_pts
