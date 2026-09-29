# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Weights for the ResNet101-DCN backbone and FPN neck tests.

``BEVFORMER_BACKBONE_WEIGHTS`` selects the source for both:

* ``dummy`` (default): seeded random weights, see ``init_dummy_backbone_weights`` and
  ``init_dummy_fpn_weights``.
* ``uniad``: the ``img_backbone`` / ``img_neck`` of UniAD's checkpoint, which have the
  same architecture (ResNet101, caffe style, DCNv2 in layer3 and layer4; FPN 512/1024/2048
  -> 4 x 256).
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
NECK_PREFIX = "img_neck."

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
        # at about its identity's, as in the trained UniAD backbone (ratio 0.6-1.1).
        for module in torch_model.modules():
            if hasattr(module, "bn3") and isinstance(module.bn3, nn.modules.batchnorm._BatchNorm):
                uniform_(module.bn3.weight, *RESIDUAL_BN_GAMMA)

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


def load_uniad_weights(torch_model, prefix):
    if not os.path.exists(UNIAD_CHECKPOINT):
        subprocess.run(["bash", UNIAD_DOWNLOAD_SCRIPT], check=True)

    checkpoint = torch.load(UNIAD_CHECKPOINT, map_location=torch.device("cpu"))
    state_dict = checkpoint.get("state_dict", checkpoint)
    module_state = OrderedDict(
        (key[len(prefix) :], value) for key, value in state_dict.items() if key.startswith(prefix)
    )
    torch_model.load_state_dict(module_state)
    torch_model.eval()
    return torch_model


def _weights_source():
    source = os.environ.get("BEVFORMER_BACKBONE_WEIGHTS", "dummy")
    if source not in ("dummy", "uniad"):
        raise ValueError(f"BEVFORMER_BACKBONE_WEIGHTS must be 'dummy' or 'uniad', got {source!r}")
    return source


def full_backbone_pcc():
    """PCC the full backbone and the backbone+FPN tests assert for the selected weights.

    The dummy weights are tuned so every layer output stays above 0.99. The trained UniAD weights
    measure 0.985 / 0.980 / 0.961 at C3 / C4 / C5 and >= 0.969 at every FPN level on
    Blackhole; their threshold is a regression floor below that, not an accuracy target.
    """
    return 0.99 if _weights_source() == "dummy" else 0.95


def load_backbone_weights(torch_model):
    if _weights_source() == "dummy":
        return init_dummy_backbone_weights(torch_model)
    return load_uniad_weights(torch_model, BACKBONE_PREFIX)


def load_fpn_weights(torch_model):
    if _weights_source() == "dummy":
        return init_dummy_fpn_weights(torch_model)
    return load_uniad_weights(torch_model, NECK_PREFIX)
