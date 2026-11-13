# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch import nn

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
