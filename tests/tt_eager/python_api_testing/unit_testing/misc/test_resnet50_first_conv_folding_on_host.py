# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger

import numpy as np

from tt_lib.utils import (
    tilize_to_list,
    tilize,
    untilize,
    _nearest_32,
    _nearest_y,
    convert_weights_2d_matrix,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose_and_pcc,
    comp_pcc,
)
from models.utility_functions import (
    pad_and_fold_conv_activation_for_unity_stride,
    pad_and_fold_conv_filters_for_unity_stride,
)
from tests.tt_eager.python_api_testing.conv.conv_unit_test_utils import (
    create_conv_act_tensor,
    create_conv_act_tensor_special,
    create_conv_weight_tensor,
    create_conv_weight_tensor_special_special,
    create_conv_bias_tensor,
)
import torch


@pytest.mark.parametrize("has_bias", (True,))
@pytest.mark.parametrize("fuse_relu", (True,))
@pytest.mark.parametrize(
    "N",
    (8,),
)
def test_resnet50_first_conv(
    device,
    use_program_cache,
    N,
    has_bias,
    fuse_relu,
):
    compute_grid_size = device.compute_with_storage_grid_size()
    is_e75_grid_size = (compute_grid_size.x * compute_grid_size.y) == 88
    if N == 8 and is_e75_grid_size:
        pytest.skip(
            f"Skipping batch 8 on E75 because expected grid size is 12x9 but E75 grid size is {compute_grid_size}"
        )
    if N != 8:
        pytest.skip("Skipping non-batch 8 tests due to potential non-determinism")

    (K, C, padded_C, H, W, R, S, padded_S, stride_h, stride_w, pad_h, pad_w) = (
        64,
        3,
        4,
        224,
        224,
        7,
        7,
        8,
        2,
        2,
        3,
        3,
    )

    torch.manual_seed(0)
    a_activation_shape = [N, C, H, W]
    A_pyt = torch.randn(a_activation_shape, dtype=torch.bfloat16).float()
    b_weights_shape = [K, C, R, S]
    B_pyt = torch.randn(b_weights_shape, dtype=torch.bfloat16).float()
    bias_shape = [K]
    bias_pyt = torch.randn(bias_shape)

    # Calculate conv result with golden result. Run Pytorch conv
    out_golden = torch.nn.functional.conv2d(
        A_pyt, B_pyt, bias=bias_pyt, stride=(stride_h, stride_w), padding=(pad_h, pad_w)
    )
    if fuse_relu:
        out_golden = torch.nn.ReLU()(out_golden)
    A_pyt_padded_folded = pad_and_fold_conv_activation_for_unity_stride(A_pyt, pad_h, pad_w, stride_h, stride_w)
    B_pyt_padded_folded = pad_and_fold_conv_filters_for_unity_stride(B_pyt, stride_h, stride_w)

    # Calculate conv result with folded conv. Run Pytorch conv with unity stride and no padding.
    out_result = torch.nn.functional.conv2d(A_pyt_padded_folded, B_pyt_padded_folded, bias=bias_pyt)
    if fuse_relu:
        out_result = torch.nn.ReLU()(out_result)

    # Compare against golden
    golden_pcc = 0.9999999999999847

    passing_pcc, output_pcc = comp_pcc(out_golden, out_result, golden_pcc)
    logger.debug(f"Passing={passing_pcc}")
    logger.debug(f"Output pcc={output_pcc}")
    assert passing_pcc
