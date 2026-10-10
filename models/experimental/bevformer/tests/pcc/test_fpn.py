# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from models.experimental.bevformer.tests.backbone_common import (
    BACKBONE_OUTPUT_DTYPES,
    FPN_KWARGS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    NUM_CAMS,
    assert_pcc,
    build_reference_fpn,
    from_conv_layout,
    to_conv_layout,
    tt_fpn_kwargs,
)
from models.experimental.bevformer.tt.model_preprocessing_backbone import create_fpn_parameters
from models.experimental.bevformer.tt.tt_fpn import TtFPN


def _backbone_output_shapes(height, width):
    """(H, W) of C3, C4, C5: stride 8, then each stride-2 layer rounds up."""
    shapes = [(height // 8, width // 8)]
    for _ in FPN_KWARGS["in_channels"][1:]:
        h, w = shapes[-1]
        shapes.append(((h + 1) // 2, (w + 1) // 2))
    return shapes


@pytest.mark.parametrize("device_params", [{"l1_small_size": 10 * 1024}], indirect=True)
def test_fpn(device, reset_seeds):
    torch_model = build_reference_fpn()

    input_tensors = [
        torch.randn(NUM_CAMS, channels, h, w)
        for channels, (h, w) in zip(FPN_KWARGS["in_channels"], _backbone_output_shapes(IMAGE_HEIGHT, IMAGE_WIDTH))
    ]
    parameters = create_fpn_parameters(torch_model, input_tensors)
    torch_outputs = torch_model(input_tensors)

    tt_model = TtFPN(
        conv_args=parameters.conv_args,
        conv_pth=parameters,
        device=device,
        input_dtypes=BACKBONE_OUTPUT_DTYPES,
        **tt_fpn_kwargs(),
    )
    tt_outputs = tt_model(
        [to_conv_layout(tensor, device, dtype) for tensor, dtype in zip(input_tensors, BACKBONE_OUTPUT_DTYPES)]
    )

    for torch_output, tt_output in zip(torch_outputs, tt_outputs, strict=True):
        assert_pcc(torch_output, from_conv_layout(tt_output, torch_output.shape), 0.99)
