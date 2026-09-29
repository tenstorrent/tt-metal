# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Camera images through the ResNet101-DCN backbone and the FPN neck, end to end.

The TT backbone's outputs feed the TT FPN as they are, so this checks the hand-off
between the two (layout, dtype, memory) as well as each half.
"""

import pytest
import torch

import ttnn
from models.experimental.bevformer.tests.backbone_common import (
    BACKBONE_OUTPUT_DTYPES,
    assert_pcc,
    build_reference_backbone,
    build_reference_fpn,
    from_conv_layout,
    random_image_batch,
    to_conv_layout,
    tt_fpn_kwargs,
    tt_resnet_kwargs,
)
from models.experimental.bevformer.tests.backbone_weights import full_backbone_pcc
from models.experimental.bevformer.tt.model_preprocessing_backbone import (
    create_fpn_parameters,
    create_resnet_parameters,
)
from models.experimental.bevformer.tt.tt_fpn import TtFPN
from models.experimental.bevformer.tt.tt_resnet import TtResNet


@torch.no_grad()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
def test_backbone_fpn(device, reset_seeds):
    torch_backbone = build_reference_backbone()
    torch_fpn = build_reference_fpn()

    torch_input = random_image_batch()
    torch_features = torch_backbone(torch_input)
    torch_outputs = torch_fpn(list(torch_features))

    backbone_parameters = create_resnet_parameters(torch_backbone, torch_input)
    fpn_parameters = create_fpn_parameters(torch_fpn, torch_features)

    tt_backbone = TtResNet(
        backbone_parameters.conv_args, backbone_parameters["res_model"], device, **tt_resnet_kwargs()
    )
    # test_fpn feeds the FPN these dtypes on their own.
    assert tt_backbone.output_dtypes == BACKBONE_OUTPUT_DTYPES
    tt_fpn = TtFPN(
        conv_args=fpn_parameters.conv_args,
        conv_pth=fpn_parameters,
        device=device,
        input_dtypes=tt_backbone.output_dtypes,
        **tt_fpn_kwargs(),
    )

    tt_input = to_conv_layout(torch_input, device, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_outputs = tt_fpn(list(tt_backbone(tt_input)))

    for torch_output, tt_output in zip(torch_outputs, tt_outputs, strict=True):
        assert_pcc(torch_output, from_conv_layout(tt_output, torch_output.shape), full_backbone_pcc())
