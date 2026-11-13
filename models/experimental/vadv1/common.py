# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
import torch
from torch import nn

CAN_BUS_DIMS = 18
# Each camera's yaw in the order the model takes the cameras: front, front right, front left, back,
# back left, back right.
CAMERA_YAWS_DEG = (0.0, -55.0, 55.0, 180.0, 110.0, -110.0)
# Focal length of the synthetic cameras at a 640-pixel-wide image.
FOCAL_PX_AT_640 = 500.0

# The last BatchNorm of each bottleneck draws its gamma from this range, so that each residual branch
# stays about as large as its identity path and the backbone output does not grow block after block.
RESIDUAL_BN_GAMMA = (0.1, 0.3)


def _ring_offsets(attention):
    """mmcv's initial sampling offsets: each head points along its own direction of a ring, the i-th
    point i + 1 steps out, at every level (and every BEV in the temporal queue)."""
    heads, points = attention.num_heads, attention.num_points
    groups = attention.num_levels * getattr(attention, "num_bev_queue", 1)
    thetas = torch.arange(heads, dtype=torch.float32) * (2.0 * math.pi / heads)
    grid = torch.stack([thetas.cos(), thetas.sin()], -1)
    grid = (grid / grid.abs().max(-1, keepdim=True)[0]).view(heads, 1, 1, 2).repeat(1, groups, points, 1)
    for i in range(points):
        grid[:, :, i, :] *= i + 1
    attention.sampling_offsets.bias.copy_(grid.flatten())


def init_dummy_weights(torch_model, seed=0):
    """Fill every parameter and BatchNorm buffer of a reference module with seeded random values.

    PyTorch's default init covers the Linears, embeddings and LayerNorms. On top of it the convs get
    He-normal weights, the BatchNorms random running statistics and affine parameters, which the
    ResNet's own init leaves at identity, the raw level and camera embeddings, which the reference
    allocates uninitialized, N(0, 1), and the deformable attentions mmcv's ring of initial sampling
    offsets.
    """
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    def normal_(tensor, std, mean=0.0):
        tensor.copy_(torch.randn(tensor.shape, generator=generator) * std + mean)

    def uniform_(tensor, low, high):
        tensor.copy_(torch.rand(tensor.shape, generator=generator) * (high - low) + low)

    with torch.no_grad():
        for module in torch_model.modules():
            if isinstance(module, nn.Conv2d):
                fan_out = module.weight.shape[0] * module.weight.shape[2] * module.weight.shape[3]
                normal_(module.weight, math.sqrt(2.0 / fan_out))
                if module.bias is not None:
                    module.bias.zero_()
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                uniform_(module.weight, 0.5, 1.5)
                normal_(module.bias, 0.1)
                normal_(module.running_mean, 0.1)
                uniform_(module.running_var, 0.5, 1.5)
            elif hasattr(module, "reset_parameters"):
                module.reset_parameters()

        for module in torch_model.modules():
            if isinstance(getattr(module, "bn3", None), nn.modules.batchnorm._BatchNorm):
                uniform_(module.bn3.weight, *RESIDUAL_BN_GAMMA)
            if isinstance(getattr(module, "sampling_offsets", None), nn.Linear):
                _ring_offsets(module)
            for name in ("level_embeds", "cams_embeds"):
                if isinstance(getattr(module, name, None), nn.Parameter):
                    normal_(getattr(module, name), 1.0)

    torch_model.eval()
    return torch_model


def synthetic_rig(width=640, height=360):
    """Six pinhole cameras around the ego, as lidar2img matrices for images of (width, height)
    content pixels. The lidar frame has x to the right, y forward and z up; each camera sits at the
    origin and looks along its yaw."""
    focal = FOCAL_PX_AT_640 * width / 640
    intrinsic = np.array([[focal, 0, width / 2, 0], [0, focal, height / 2, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    matrices = []
    for yaw_deg in CAMERA_YAWS_DEG:
        yaw = math.radians(yaw_deg)
        forward = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        right = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        down = np.array([0.0, 0.0, -1.0])
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = np.stack([right, down, forward])
        matrices.append(intrinsic @ extrinsic)
    return matrices


def synthetic_can_bus(x, y, yaw, generator=None):
    """An absolute CAN bus for an ego at (x, y) metres heading yaw radians. With a generator, the
    motion fields between the orientation and the heading are N(0, 1) draws, else zero."""
    can_bus = np.zeros(CAN_BUS_DIMS)
    can_bus[:3] = (x, y, 0.0)
    can_bus[3:7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    if generator is not None:
        can_bus[7:-2] = torch.randn(CAN_BUS_DIMS - 9, generator=generator, dtype=torch.float64).numpy()
    angle = math.degrees(yaw) % 360
    can_bus[-2], can_bus[-1] = math.radians(angle), angle
    return can_bus


def synthetic_agents(num_agents, fut_ts, step_s, seed=0):
    """Random agents within 20 m of the ego for the planning metric, which is all the ground truth
    the forward reads: LiDAR boxes (x, y, z, w, l, h, yaw, vx, vy), class labels, and attribute rows
    laid out as the metric reads them, with constant-velocity future offsets, all valid, and the
    class at column 3 * fut_ts + 9."""
    generator = torch.Generator().manual_seed(seed)

    def uniform(low, high, *shape):
        return torch.rand(shape, generator=generator) * (high - low) + low

    xy = uniform(-20.0, 20.0, num_agents, 2)
    z = torch.full((num_agents, 1), -1.7)
    size = torch.cat(
        [uniform(1.6, 2.0, num_agents, 1), uniform(3.8, 5.2, num_agents, 1), uniform(1.5, 1.7, num_agents, 1)], 1
    )
    yaw = uniform(-math.pi, math.pi, num_agents, 1)
    velocity = torch.randn(num_agents, 2, generator=generator) * 2.0
    boxes = torch.cat([xy, z, size, yaw, velocity], 1)
    labels = torch.randint(0, 10, (num_agents,), generator=generator)

    attrs = torch.zeros(num_agents, 4 * fut_ts + 10)
    attrs[:, : 2 * fut_ts] = (velocity * step_s).repeat(1, fut_ts)
    attrs[:, 2 * fut_ts : 3 * fut_ts] = 1.0
    attrs[:, 3 * fut_ts + 9] = labels.float()
    return boxes, labels, attrs
