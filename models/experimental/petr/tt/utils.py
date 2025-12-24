# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import numpy as np


def inverse_sigmoid(x, eps: float = 1e-7):
    x = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT)
    x_clamped = ttnn.clip(x, eps, 1.0 - eps)
    x_neg = ttnn.neg(x_clamped)
    one_minus_x = ttnn.add(x_neg, 1.0)
    one_minus_x = ttnn.clip(one_minus_x, eps, float("inf"))

    ratio = ttnn.div(x_clamped, one_minus_x)
    result = ttnn.log(ratio)

    return result


def limit_period(val, offset: float = 0.5, period: float = np.pi):
    tmp_val = ttnn.add(ttnn.div(val, period), offset)
    tmp_val = ttnn.floor(tmp_val)
    tmp_val = ttnn.mul(tmp_val, period)

    limited_val = ttnn.sub(val, tmp_val)
    return limited_val


def denormalize_bbox(normalized_bboxes, pc_range):
    rot_sine = normalized_bboxes[..., 6:7]
    rot_cosine = normalized_bboxes[..., 7:8]
    rot_sine = ttnn.to_layout(rot_sine, layout=ttnn.TILE_LAYOUT)
    rot_cosine = ttnn.to_layout(rot_cosine, layout=ttnn.TILE_LAYOUT)
    rot = ttnn.atan2(rot_sine, rot_cosine)

    rot = ttnn.mul(rot, -1)
    rot = ttnn.sub(rot, np.pi / 2)

    rot = limit_period(rot, period=np.pi * 2)

    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]

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
