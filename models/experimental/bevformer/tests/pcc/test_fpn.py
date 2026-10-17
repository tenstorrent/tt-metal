# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from models.experimental.bevformer.model_config import FPN_KWARGS, NUM_CAMS, SPATIAL_SHAPES
from models.experimental.bevformer.tests.common import (
    BACKBONE_OUTPUT_DTYPES,
    assert_pcc,
    build_reference_fpn,
    from_conv_layout,
    to_conv_layout,
)
from models.experimental.bevformer.tt.model_preprocessing import create_fpn_parameters
from models.experimental.bevformer.tt.tt_fpn import TTFPN


@pytest.mark.parametrize("device_params", [{"l1_small_size": 10 * 1024}], indirect=True)
def test_fpn(device, reset_seeds):
    torch_model = build_reference_fpn()

    # C3-C5 have the sizes of the FPN's first three levels; its extra conv makes the fourth.
    input_tensors = [
        torch.randn(NUM_CAMS, channels, h, w)
        for channels, (h, w) in zip(FPN_KWARGS["in_channels"], SPATIAL_SHAPES, strict=False)
    ]
    parameters = create_fpn_parameters(torch_model, input_tensors)
    torch_outputs = torch_model(input_tensors)

    tt_model = TTFPN(parameters, device, input_dtypes=BACKBONE_OUTPUT_DTYPES)
    tt_outputs = tt_model(
        [to_conv_layout(tensor, device, dtype) for tensor, dtype in zip(input_tensors, BACKBONE_OUTPUT_DTYPES)]
    )

    for torch_output, tt_output in zip(torch_outputs, tt_outputs, strict=True):
        assert_pcc(torch_output, from_conv_layout(tt_output, torch_output.shape), 0.99)
