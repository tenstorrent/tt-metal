# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import numpy as np
from loguru import logger


def inverse_sigmoid(x, eps: float = 1e-7):
    device = x.device()

    # Convert to torch for safe operations
    x_torch = ttnn.to_torch(x).to(torch.float32)

    # Clamp to valid range with safety margin
    x_torch = torch.clamp(x_torch, min=eps, max=1.0 - eps)

    # Compute inverse sigmoid: log(x / (1-x))
    one_minus_x = 1.0 - x_torch

    # Additional safety: ensure denominator is not too small
    one_minus_x = torch.clamp(one_minus_x, min=eps)
    x_torch = torch.clamp(x_torch, min=eps)

    result = torch.log(x_torch / one_minus_x)

    # Check for NaN/Inf and handle
    if torch.isnan(result).any() or torch.isinf(result).any():
        logger.warning(f"NaN/Inf in inverse_sigmoid! Clamping output")
        result = torch.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0)

    # Convert back to ttnn
    result_ttnn = ttnn.from_torch(result, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    result_ttnn = ttnn.to_device(result_ttnn, device)

    return result_ttnn


def limit_period(val, offset: float = 0.5, period: float = np.pi):
    tmp_val = ttnn.add(ttnn.div(val, period), offset)
    tmp_val = ttnn.floor(tmp_val)
    tmp_val = ttnn.mul(tmp_val, period)

    limited_val = ttnn.sub(val, tmp_val)
    return limited_val


def denormalize_bbox(normalized_bboxes, pc_range):
    # rotation
    rot_sine = normalized_bboxes[..., 6:7]

    rot_cosine = normalized_bboxes[..., 7:8]
    rot_sine = ttnn.to_layout(rot_sine, layout=ttnn.TILE_LAYOUT)
    rot_cosine = ttnn.to_layout(rot_cosine, layout=ttnn.TILE_LAYOUT)
    rot = ttnn.atan2(rot_sine, rot_cosine)

    rot = ttnn.mul(rot, -1)
    rot = ttnn.sub(rot, np.pi / 2)

    rot = limit_period(rot, period=np.pi * 2)

    # center in the bev
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]

    # size
    length = normalized_bboxes[..., 2:3]
    width = normalized_bboxes[..., 3:4]
    height = normalized_bboxes[..., 5:6]

    width = ttnn.to_layout(width, layout=ttnn.TILE_LAYOUT)
    length = ttnn.to_layout(length, layout=ttnn.TILE_LAYOUT)
    height = ttnn.to_layout(height, layout=ttnn.TILE_LAYOUT)

    width = ttnn.exp(width)
    length = ttnn.exp(length)
    height = ttnn.exp(height)
    if normalized_bboxes.shape[-1] > 8:
        # velocity
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        cx = ttnn.to_layout(cx, layout=ttnn.TILE_LAYOUT)
        cy = ttnn.to_layout(cy, layout=ttnn.TILE_LAYOUT)
        cz = ttnn.to_layout(cz, layout=ttnn.TILE_LAYOUT)
        vx = ttnn.to_layout(vx, layout=ttnn.TILE_LAYOUT)
        vy = ttnn.to_layout(vy, layout=ttnn.TILE_LAYOUT)
        denormalized_bboxes = ttnn.concat([cx, cy, cz, length, width, height, rot, vx, vy], dim=-1)
    else:
        cx = ttnn.to_layout(cx, layout=ttnn.TILE_LAYOUT)
        cy = ttnn.to_layout(cy, layout=ttnn.TILE_LAYOUT)
        cz = ttnn.to_layout(cz, layout=ttnn.TILE_LAYOUT)
        denormalized_bboxes = torch.concat([cx, cy, cz, length, width, height, rot], dim=-1)

    return denormalized_bboxes
