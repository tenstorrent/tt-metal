# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Camera images through the ResNet101-DCN backbone and the FPN neck, end to end.

The TT backbone's outputs feed the TT FPN as they are, so this checks the hand-off
between the two (layout, dtype, memory) as well as each half.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.bevformer.reference.fpn import FPN
from models.experimental.bevformer.reference.resnet import ResNet
from models.experimental.bevformer.tests.backbone_weights import load_backbone_weights, load_fpn_weights
from models.experimental.bevformer.tt.model_preprocessing_backbone import (
    create_fpn_parameters,
    create_resnet_parameters,
)
from models.experimental.bevformer.tt.tt_fpn import TtFPN
from models.experimental.bevformer.tt.tt_resnet import TtResNet
from tests.ttnn.utils_for_testing import assert_with_pcc

NUM_CAMS = 6

# BEVFormer-base's camera images are 1600x900, padded to 1600x928 so the height is a
# multiple of 32 before the backbone.
IMAGE_HEIGHT = 928
IMAGE_WIDTH = 1600

# Backbone stages and FPN levels whose convs keep their activations in DRAM because
# they do not fit in L1.
DRAM_ACTIVATION_STAGES = (0, 1, 3)
DRAM_ACTIVATION_LEVELS = (0, 1)

RESNET_KWARGS = dict(
    depth=101,
    in_channels=3,
    stem_channels=None,
    base_channels=64,
    num_stages=4,
    strides=(1, 2, 2, 2),
    dilations=(1, 1, 1, 1),
    out_indices=(1, 2, 3),
    style="caffe",
    deep_stem=False,
    avg_down=False,
    frozen_stages=4,
    conv_cfg=None,
    stage_with_dcn=(False, False, True, True),
    pretrained=None,
    init_cfg=None,
)


def _assert_pcc(expected, actual, pcc):
    passed, message = assert_with_pcc(expected, actual, pcc)
    logger.info(f"PCC {message} (threshold {pcc})")
    return passed, message


@torch.no_grad()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
def test_backbone_fpn(device, reset_seeds):
    torch_backbone = ResNet(
        **RESNET_KWARGS,
        norm_cfg={"type": "BN2d", "requires_grad": False},
        norm_eval=True,
        dcn={"type": "DCNv2", "deform_groups": 1, "fallback_on_stride": False},
        plugins=None,
        with_cp=False,
        zero_init_residual=True,
    )
    torch_backbone = load_backbone_weights(torch_backbone)
    torch_fpn = FPN(
        in_channels=[512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs="on_output",
        num_outs=4,
        relu_before_extra_convs=True,
    )
    torch_fpn = load_fpn_weights(torch_fpn)

    torch_input = torch.randn(NUM_CAMS, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
    torch_features = torch_backbone(torch_input)
    torch_outputs = torch_fpn(list(torch_features))

    backbone_parameters = create_resnet_parameters(torch_backbone, torch_input)
    fpn_parameters = create_fpn_parameters(torch_fpn, torch_features)

    tt_backbone = TtResNet(
        backbone_parameters.conv_args,
        backbone_parameters["res_model"],
        device,
        **RESNET_KWARGS,
        dcn=True,
        dram_activation_stages=DRAM_ACTIVATION_STAGES,
    )
    tt_fpn = TtFPN(
        conv_args=fpn_parameters.model_args,
        conv_pth=fpn_parameters,
        device=device,
        dram_activation_levels=DRAM_ACTIVATION_LEVELS,
    )

    nhwc = torch_input.permute(0, 2, 3, 1)
    tt_input = ttnn.from_torch(
        nhwc.reshape(1, 1, NUM_CAMS * IMAGE_HEIGHT * IMAGE_WIDTH, 3), device=device, dtype=ttnn.bfloat16
    )
    tt_outputs = tt_fpn(list(tt_backbone(tt_input)))

    for torch_output, tt_output in zip(torch_outputs, tt_outputs):
        n, c, h, w = torch_output.shape
        tt_output = ttnn.to_torch(tt_output).reshape(n, h, w, c).permute(0, 3, 1, 2)
        _assert_pcc(torch_output, tt_output, 0.99)
