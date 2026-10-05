# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Seeded random weights for the ResNet101-DCN backbone and FPN neck tests."""

import math

import torch
import torch.nn as nn

from models.experimental.bevformer.reference.resnet import ModulatedDeformConv2dPack

# Tuned so the random backbone matches the trained BEVFormer-base one in output std at C3-C5
# (order 1), DCN offsets (under a pixel on average, so samples fall between pixels)
# and DCN masks (near 0.9).
DCN_OFFSET_STD = 0.04
DCN_MASK_BIAS = 2.6
RESIDUAL_BN_GAMMA = (0.1, 0.3)


def init_dummy_backbone_weights(torch_model, seed=0, input_std=1.0):
    """Fill every parameter and BatchNorm buffer with seeded random values.

    The stem conv is divided by ``input_std``, the std of the images the model will see, so
    everything after the stem sees the unit scale the values above were tuned for.

    The reference model's own initialization cannot be used. ``ModulatedDeformConv2dPack``
    allocates its weight uninitialized, and the ResNet stores ``init_cfg`` without applying
    it, so its BatchNorms keep identity statistics that no trained network has.
    """
    generator = torch.Generator().manual_seed(seed)

    def normal_(tensor, std, mean=0.0):
        tensor.copy_(torch.randn(tensor.shape, generator=generator) * std + mean)

    def uniform_(tensor, low, high):
        tensor.copy_(torch.rand(tensor.shape, generator=generator) * (high - low) + low)

    offset_convs = {
        id(module.conv_offset) for module in torch_model.modules() if isinstance(module, ModulatedDeformConv2dPack)
    }

    with torch.no_grad():
        for module in torch_model.modules():
            if isinstance(module, ModulatedDeformConv2dPack):
                fan_out = module.weight.shape[0] * module.weight.shape[2] * module.weight.shape[3]
                normal_(module.weight, math.sqrt(2.0 / fan_out))
                if module.bias is not None:
                    module.bias.zero_()
                normal_(module.conv_offset.weight, DCN_OFFSET_STD)
                # conv_offset emits 3*K*K channels: two thirds of (y, x)-interleaved offsets, then the
                # mask logits, which go through sigmoid.
                module.conv_offset.bias.zero_()
                module.conv_offset.bias[2 * module.conv_offset.bias.numel() // 3 :] = DCN_MASK_BIAS
            elif isinstance(module, nn.Conv2d):
                if id(module) in offset_convs:
                    continue
                fan_out = module.weight.shape[0] * module.weight.shape[2] * module.weight.shape[3]
                normal_(module.weight, math.sqrt(2.0 / fan_out))
                if module.bias is not None:
                    module.bias.zero_()
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                uniform_(module.weight, 0.5, 1.5)
                normal_(module.bias, 0.1)
                normal_(module.running_mean, 0.1)
                uniform_(module.running_var, 0.5, 1.5)

        # With the BatchNorm ranges above, the last BatchNorm of each branch at gamma ~1
        # makes the branch outgrow its identity path block after block, and the output
        # grows by orders of magnitude over 33 blocks. This range keeps each branch's std
        # at about its identity's, as in the trained backbone (ratio 0.6-1.1).
        for module in torch_model.modules():
            if hasattr(module, "bn3") and isinstance(module.bn3, nn.modules.batchnorm._BatchNorm):
                uniform_(module.bn3.weight, *RESIDUAL_BN_GAMMA)

        if input_std != 1.0:
            torch_model.conv1.weight /= input_std

    torch_model.eval()
    return torch_model


def init_dummy_fpn_weights(torch_model, seed=0):
    """Fill the FPN's convs with seeded random weights and biases."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for module in torch_model.modules():
            if isinstance(module, nn.Conv2d):
                fan_out = module.weight.shape[0] * module.weight.shape[2] * module.weight.shape[3]
                module.weight.copy_(torch.randn(module.weight.shape, generator=generator) * math.sqrt(2.0 / fan_out))
                if module.bias is not None:
                    module.bias.copy_(torch.randn(module.bias.shape, generator=generator) * 0.1)
    torch_model.eval()
    return torch_model
