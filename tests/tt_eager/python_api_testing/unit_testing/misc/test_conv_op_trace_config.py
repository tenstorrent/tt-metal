# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy
from loguru import logger
from tests.tt_eager.python_api_testing.conv.conv_op_trace_config import (
    trace_conv_to_generate_data_top_left_indices_and_pad_metadata,
    traced_conv_reference,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_allclose_and_pcc


# conv params - output_channels, input_channels, filter_h, filter_w, stride_h, stride_w, pad_h, pad_w, dilation, groups
@pytest.mark.parametrize(
    "conv_params, input_nchw_shape",
    (
        ((1, 1, 2, 2, 1, 1, 0, 0, 1, 1), (8, 1, 8, 8)),
        ((1, 1, 2, 2, 1, 1, 1, 1, 1, 1), (8, 1, 8, 8)),
        ((1, 1, 4, 4, 1, 1, 0, 0, 1, 1), (8, 1, 115, 115)),
    ),
)
def test_run_op_trace_config(conv_params, input_nchw_shape):
    pad_metadata, data_top_left_indices = trace_conv_to_generate_data_top_left_indices_and_pad_metadata(
        conv_params, input_nchw_shape
    )
    logger.trace(f"Data top left indices - {data_top_left_indices}")
    logger.trace(f"Pad meta data - {pad_metadata}")
    # run trace conv reference
    traced_conv_reference(pad_metadata, data_top_left_indices, conv_params, input_nchw_shape)
