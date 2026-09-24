# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Weights for the ResNet101-DCN backbone tests.

``BEVFORMER_BACKBONE_WEIGHTS`` selects the source:

* ``dummy`` (default): seeded random weights, see ``init_dummy_backbone_weights``.
* ``uniad``: the ``img_backbone`` of UniAD's checkpoint, which has the same
  architecture (ResNet101, caffe style, DCNv2 in stages 3 and 4).
"""

import math
import os
import subprocess
from collections import OrderedDict

import torch
import torch.nn as nn

from models.experimental.bevformer.reference.resnet import ModulatedDeformConv2dPack

UNIAD_CHECKPOINT = "models/experimental/uniad/uniad_base_e2e.pth"
UNIAD_DOWNLOAD_SCRIPT = "models/experimental/uniad/weights_download.sh"
BACKBONE_PREFIX = "img_backbone."

# Tuned so the random backbone matches the trained UniAD one in output std at C3-C5
# (order 1), DCN offsets (under a pixel on average, so samples fall between pixels)
# and DCN masks (near 0.9). It does not reproduce the trained backbone's outlier
# channels, which bfloat8_b handles worst, so PCC here reads higher than with
# trained weights.
DCN_OFFSET_STD = 0.04
DCN_MASK_BIAS = 2.6
RESIDUAL_BN_GAMMA = (0.1, 0.3)


def init_dummy_backbone_weights(torch_model, seed=0):
    """Fill every parameter and BatchNorm buffer with seeded random values.

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
                # conv_offset emits (offset_y|offset_x|mask) thirds; the mask goes through sigmoid.
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
        # at about its identity's, as in the trained UniAD backbone (ratio 0.6-1.1).
        for module in torch_model.modules():
            if hasattr(module, "bn3") and isinstance(module.bn3, nn.modules.batchnorm._BatchNorm):
                uniform_(module.bn3.weight, *RESIDUAL_BN_GAMMA)

    torch_model.eval()
    return torch_model


def load_uniad_backbone_weights(torch_model):
    if not os.path.exists(UNIAD_CHECKPOINT):
        subprocess.run(["bash", UNIAD_DOWNLOAD_SCRIPT], check=True)

    checkpoint = torch.load(UNIAD_CHECKPOINT, map_location=torch.device("cpu"))
    state_dict = checkpoint.get("state_dict", checkpoint)
    backbone_state = OrderedDict(
        (key[len(BACKBONE_PREFIX) :], value) for key, value in state_dict.items() if key.startswith(BACKBONE_PREFIX)
    )
    torch_model.load_state_dict(backbone_state)
    torch_model.eval()
    return torch_model


def load_backbone_weights(torch_model):
    source = os.environ.get("BEVFORMER_BACKBONE_WEIGHTS", "dummy")
    if source == "dummy":
        return init_dummy_backbone_weights(torch_model)
    if source == "uniad":
        return load_uniad_backbone_weights(torch_model)
    raise ValueError(f"BEVFORMER_BACKBONE_WEIGHTS must be 'dummy' or 'uniad', got {source!r}")
