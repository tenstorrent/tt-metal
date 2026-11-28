# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from models.experimental.bevformer.model_config import FPN_KWARGS, NUM_CAMS, SPATIAL_SHAPES, tt_fpn_kwargs
from models.experimental.bevformer.tests.common import (
    BACKBONE_OUTPUT_DTYPES,
    assert_pcc,
    backbone_output_layouts,
    build_reference_fpn,
    from_conv_layout,
    to_conv_layout,
)
from models.experimental.bevformer.tt.model_preprocessing import create_fpn_parameters
from models.experimental.bevformer.tt.tt_common import reduced_grid
from models.experimental.bevformer.tt.tt_fpn import TTFPN


@pytest.mark.parametrize("device_params", [{"l1_small_size": 10 * 1024}], indirect=True)
def test_fpn(device, reset_seeds):
    torch_model = build_reference_fpn()

    # Each backbone output has the size of one of the FPN's first levels; the FPN's extra convs, if
    # any, make the rest.
    input_tensors = [
        torch.randn(NUM_CAMS, channels, h, w)
        for channels, (h, w) in zip(FPN_KWARGS["in_channels"], SPATIAL_SHAPES, strict=False)
    ]
    on_reduced_grid = reduced_grid(device)
    parameters = create_fpn_parameters(
        torch_model, input_tensors, tt_fpn_kwargs(len(torch_model.fpn_convs), on_reduced_grid)
    )
    torch_outputs = torch_model(input_tensors)

    tt_model = TTFPN(parameters, device, input_dtypes=BACKBONE_OUTPUT_DTYPES)
    tt_outputs = tt_model(
        [
            to_conv_layout(tensor, device, dtype, layout)
            for tensor, dtype, layout in zip(
                input_tensors, BACKBONE_OUTPUT_DTYPES, backbone_output_layouts(on_reduced_grid)
            )
        ]
    )

    for torch_output, tt_output in zip(torch_outputs, tt_outputs, strict=True):
        assert_pcc(torch_output, from_conv_layout(tt_output, torch_output.shape), 0.99)
