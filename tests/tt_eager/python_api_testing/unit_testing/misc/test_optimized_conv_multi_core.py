# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from pathlib import Path
import sys
from loguru import logger

import numpy as np
import ttnn
from tt_lib.utils import (
    tilize_to_list,
    tilize,
    untilize,
    _nearest_32,
    _nearest_y,
    convert_weights_2d_matrix,
)
from models.utility_functions import print_diff_argmax, is_close, comp_pcc, comp_allclose_and_pcc
from tests.tt_eager.python_api_testing.conv.conv_unit_test_utils import (
    create_conv_act_tensor,
    create_conv_weight_tensor,
    create_conv_bias_tensor,
    create_conv_weight_tensor_special_padding,
)
import torch


@pytest.mark.parametrize("untilize_out", (False,))
@pytest.mark.parametrize("has_bias", (False,))
@pytest.mark.parametrize("fuse_relu", (False,))
@pytest.mark.parametrize("N", (1,))
@pytest.mark.parametrize(
    "K, C, H, W, R, S, stride_h, stride_w, pad_h, pad_w, act_block_h, num_cores_x, num_cores_y, per_core_out_matrix_h_ntiles, per_core_weight_matrix_w_ntiles",
    (
        # (32, 32, 2, 2, 1, 1, 1, 1, 0, 0),
        # (32, 3, 100, 100, 3, 3, 1, 1, 0, 0),
        # (64, 32, 2, 2, 1, 1, 1, 1, 0, 0),
        # # channels = 3 padding
        # (32, 3, 5, 5, 1, 1, 1, 1, 0, 0),
        # # w/ conv padding
        # (32, 32, 5, 5, 1, 1, 1, 1, 1, 1),
        # # Hat = 1, Wat = 1, Wbt = 1
        # (32, 32, 5, 5, 1, 1, 1, 1, 0, 0),
        # # Hat = 2, Wat = 1, Wbt = 1
        # (32, 32, 8, 8, 1, 1, 1, 1, 0, 0),
        # # Hat = 1, Wat = 2, Wbt = 1
        # (32, 64, 5, 5, 1, 1, 1, 1, 0, 0),
        # # Hat = 2, Wat = 2, Wbt = 1
        # (32, 64, 8, 8, 1, 1, 1, 1, 0, 0),
        # # Hat = 1, Wat = 1, Wbt = 2
        # (64, 32, 5, 5, 1, 1, 1, 1, 0, 0),
        # # Hat = 1, Wat = 2, Wbt = 2
        # (64, 64, 5, 5, 1, 1, 1, 1, 0, 0),
        # # Hat = 2, Wat = 1, Wbt = 2
        # (64, 32, 8, 8, 1, 1, 1, 1, 0, 0),
        # # Hat = 2, Wat = 2, Wbt = 2
        # (64, 64, 8, 8, 1, 1, 1, 1, 0, 0),
        # # Hat = 8, Wat = 8, Wbt = 8
        # (8 * 32, 8 * 32, 16, 16, 1, 1, 1, 1, 0, 0),
        # # resnet50 first conv
        # (64, 3, 224, 224, 7, 7, 2, 2, 3, 3),
        # # num blocks weight w = 4, num blocks act h = 4, num blocks act w = 3
        # (16 * 32, 32, 24, 24, 3, 3, 1, 1, 0, 0),
        # (32, 32, 16, 16, 1, 1, 1, 1, 0, 0, 2, 1, 4, 2),
        (64, 32, 16, 16, 1, 1, 1, 1, 0, 0, 1, 1, 4, 2, 2),
    ),
)
def test_run_optimized_conv(
    device,
    use_program_cache,
    N,
    K,
    C,
    H,
    W,
    R,
    S,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    act_block_h,
    num_cores_x,
    num_cores_y,
    per_core_out_matrix_h_ntiles,
    per_core_weight_matrix_w_ntiles,
    untilize_out,
    has_bias,
    fuse_relu,
):
    if has_bias and untilize_out:
        ## bias is only supported without untilize out
        pytest.skip()

    num_iterations = 1  # run twice to test op caching flow for conv op
    for i in range(num_iterations):
        # torch.set_printoptions(threshold=10000)
        torch.manual_seed(0)
        a_activation_shape = [N, C, H, W]
        # A_pyt = torch.randn(a_activation_shape)
        A_pyt = torch.normal(mean=0, std=0.1, size=a_activation_shape)
        # A_pyt = torch.ones(a_activation_shape, dtype=torch.bfloat16).float()
        b_weights_shape = [K, C, R, S]
        # B_pyt = torch.randn(b_weights_shape)
        B_pyt = torch.normal(mean=0, std=0.1, size=b_weights_shape)
        # B_pyt = torch.ones(b_weights_shape, dtype=torch.bfloat16).float()
        bias_shape = [1, 1, 1, K]
        # bias_pyt = torch.randn(bias_shape)
        bias_pyt = torch.normal(mean=0, std=0.1, size=bias_shape)
        # bias_pyt = torch.zeros(bias_shape, dtype=torch.bfloat16).float() * 3.
        # bias_pyt = torch.range(start=0, end=(K - 1), dtype=torch.bfloat16).float()

        # Parameters to define block dims
        act_block_w = (int)((_nearest_32(_nearest_y(C, 16) * S)) / 32)
        weight_block_h = act_block_w
        weight_block_w = 1
        out_subblock_h = 1
        out_subblock_w = 1
        out_block_h = act_block_h * 2

        OH = ((int)((H - R + 2 * pad_h) / stride_h)) + 1
        OW = ((int)((W - S + 2 * pad_w) / stride_w)) + 1
        conv_output_shape = [N, OH, OW, K]
        act_matrix_height_ntiles = (int)(_nearest_y(N * OH * OW, act_block_h * 32) / 32)
        # Prepare activations
        A_cl_host = create_conv_act_tensor(A_pyt, N, C, H, W)
        A = A_cl_host.to(device)

        # Prepare weights
        B_tiled_host = create_conv_weight_tensor_special_padding(B_pyt, K, C, R, S, weight_block_h, weight_block_w)
        B_tiled = B_tiled_host.to(device)

        # Bias
        bias_cl_host = create_conv_bias_tensor(bias_pyt, 1, K, _nearest_y(K, weight_block_w * 32), pad=0)
        bias_device = bias_cl_host.to(device)

        if has_bias:
            bias = torch.flatten(bias_pyt)
        else:
            bias = None

        # Calculate conv result with golden result. Run Pytorch conv
        out_golden = torch.nn.functional.conv2d(
            A_pyt, B_pyt, bias=bias, stride=(stride_h, stride_w), padding=(pad_h, pad_w)
        )
        if fuse_relu:
            out_golden = torch.nn.ReLU()(out_golden)

        # Run TT metal OP
        if not has_bias:
            bias_device = None
        out = ttnn.operations.conv2d.optimized_conv(
            A,
            B_tiled,
            bias=bias_device,
            conv_reader_indices=None,
            conv_params=[R, S, stride_h, stride_w, pad_h, pad_w],
            output_channels=K,
            untilize_out=untilize_out,
            has_bias=has_bias,
            fuse_relu=fuse_relu,
            math_fidelity=ttnn.MathFidelity.HiFi4,
            parallelization_config=ttnn.operations.conv2d.OptimizedConvParallelizationConfig(
                grid_size=(num_cores_x, num_cores_y),
                num_cores_nhw=num_cores_y,
                per_core_out_matrix_height_ntiles=per_core_out_matrix_h_ntiles,
                per_core_out_matrix_width_ntiles=per_core_weight_matrix_w_ntiles,
            ),
            block_config=ttnn.operations.conv2d.OptimizedConvBlockConfig(
                act_block_h_ntiles=act_block_h,
                act_block_w_ntiles=act_block_w,
                out_subblock_h_ntiles=out_subblock_h,
                out_subblock_w_ntiles=out_subblock_w,
            ),
        )
        if not untilize_out:
            out_unpadded_shape = [1, 1, N * OH * OW, K]
            assert out_unpadded_shape == list(out.shape_without_padding())
            out = ttnn.format_output_tensor(out, out.shape_without_padding(), device, ttnn.ROW_MAJOR_LAYOUT)
            out = out.reshape(conv_output_shape[0], conv_output_shape[1], conv_output_shape[2], conv_output_shape[3])
        out = out.cpu()
        assert list(out.get_legacy_shape()) == conv_output_shape
        assert out.get_layout() == ttnn.ROW_MAJOR_LAYOUT

        # Copy output to host and convert tt tensor to pytorch tensor
        out_result = out.to_torch().float()
        out_result = torch.transpose(out_result, 2, 3)
        out_result = torch.transpose(out_result, 1, 2)

        torch.set_printoptions(precision=3, sci_mode=False, linewidth=500, threshold=10000, edgeitems=32)

        # print(f'OUT: {out_result}')
        # print(f'GLD: {out_golden}')

        # Compare against golden
        assert out_result.shape == out_golden.shape
        [output_N, output_C, output_H, output_W] = out_result.shape
        # print("Golden - ")
        # print(out_golden.flatten())
        # print("Result - ")
        # print(out_result.flatten())
        # for n in range(output_N):
        #     for c in range(output_C):
        #         for h in range(output_H):
        #             for w in range(output_W):
        #                 calculated = torch.tensor(out_result[n][c][h][w])
        #                 golden = torch.tensor(out_golden[n][c][h][w])
        #                 atol_delta = torch.abs(golden - calculated).item()
        #                 rtol_delta = torch.abs(golden - calculated) / torch.abs(calculated)
        #                 if atol_delta > 0.1 or rtol_delta > 0.1:
        #                     print(f"Bad value at {n},{c},{h},{w} with ATOL={atol_delta} and RTOL={rtol_delta}")
        #                     print(f"    result={calculated}, golden={golden}")

        passing_allclose_and_pcc, output_info = comp_allclose_and_pcc(
            out_golden, out_result, rtol=1e-1, atol=1e-3, pcc=0.9999
        )  # For LowFi we need 0.99976
        logger.debug(f"Passing={passing_allclose_and_pcc}")
        logger.debug(f"Output info={output_info}")
        passing_pcc, _ = comp_pcc(out_golden, out_result, pcc=0.9998)  # For LowFi we need 0.99976
        assert passing_pcc
        # assert passing_allclose_and_pcc
