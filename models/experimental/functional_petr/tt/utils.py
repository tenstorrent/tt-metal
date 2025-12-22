# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import numpy as np


def inverse_sigmoid(x, eps: float = 1e-5):
    x = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT)
    x = ttnn.clamp(x, min=0, max=1)
    x1 = ttnn.clamp(x, min=eps)
    if len(x.shape) == 3:
        x_temp = ttnn.ones(shape=[x.shape[0], x.shape[1], x.shape[2]], layout=ttnn.TILE_LAYOUT, device=x.device())
    else:
        x_temp = ttnn.ones(
            shape=[x.shape[0], x.shape[1], x.shape[2], x.shape[3]], layout=ttnn.TILE_LAYOUT, device=x.device()
        )
    x_temp = x_temp - x
    x2 = ttnn.clamp(x_temp, min=eps)
    return ttnn.log(ttnn.div(x1, x2))


def limit_period(val, offset: float = 0.5, period: float = np.pi):
    # limited_val = val - torch.floor(val / period + offset) * period
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

    print("np.pi / 2", np.pi / 2)

    # rot = -rot - np.pi / 2
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
        print("ifffff")
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
