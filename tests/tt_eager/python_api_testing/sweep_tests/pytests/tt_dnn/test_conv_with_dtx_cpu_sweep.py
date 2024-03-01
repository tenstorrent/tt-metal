# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import numpy as np
import tt_lib as ttl
from tt_lib.utils import blocked_mm_with_conv_act, _nearest_32, _nearest_y
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.tt_eager.python_api_testing.conv.pytorch_conv_tb import (
    TestLevel,
    generate_conv_tb_with_pytorch_golden,
    generate_conv_tb,
)
from tests.tt_eager.python_api_testing.conv.conv_utils import (
    create_conv_act_tensor,
    create_conv_weight_tensor,
)
import torch


def run_conv_as_large_matmul_dtx_cpu(conv_op_test_params, pytorch_inputs_and_golden):
    print("Testing convolution with following parameters - ")
    conv_op_test_params.print("   ")
    ctp = conv_op_test_params.conv_params
    N = ctp.act_shape[0]
    C = ctp.act_shape[1]
    H = ctp.act_shape[2]
    W = ctp.act_shape[3]
    K = ctp.weight_shape[0]
    assert ctp.weight_shape[1] == C
    R = ctp.weight_shape[2]
    S = ctp.weight_shape[3]
    stride_h = ctp.stride_h
    stride_w = ctp.stride_w
    pad_h = ctp.pad_h
    pad_w = ctp.pad_w

    A_pyt = pytorch_inputs_and_golden[0]
    B_pyt = pytorch_inputs_and_golden[1]

    # Parameters defining block dims
    act_block_h = 4
    act_block_w = 4
    weight_block_h = act_block_w
    weight_block_w = 4
    OH = ((int)((H - R + 2 * pad_h) / stride_h)) + 1
    OW = ((int)((W - S + 2 * pad_w) / stride_w)) + 1
    mm_output_shape = [
        1,
        1,
        _nearest_y(OH * OW, 32 * act_block_h),
        _nearest_y(K, 32 * weight_block_w),
    ]

    # Prepare activations
    A_cl = create_conv_act_tensor(A_pyt, 1, C, H, W)
    # Prepare weights
    B_tiled = create_conv_weight_tensor(B_pyt, K, C, R, S, weight_block_h, weight_block_w)

    if conv_op_test_params.test_level == TestLevel.INPUT_TENSOR_CREATE:
        return True
    assert conv_op_test_params.test_level == TestLevel.OP_FULL_COMPUTE

    # Call DTX pass to transform A
    act_block_width_datums = act_block_w * 32
    act_block_height_datums = act_block_h * 32
    weight_block_width_datums = weight_block_w * 32
    matrix_activation_h_tiles = (int)(_nearest_y(OH * OW, act_block_height_datums) / 32)
    matrix_weight_w_tiles = (int)(_nearest_y(K, weight_block_width_datums) / 32)
    matrix_activation_w_tiles = (int)(_nearest_y(_nearest_y(C, 16) * R * S, act_block_width_datums) / 32)

    num_blocks_act_w = (int)(matrix_activation_w_tiles / act_block_w)
    num_blocks_act_h = (int)(matrix_activation_h_tiles / act_block_h)
    num_blocks_weight_w = (int)(matrix_weight_w_tiles / weight_block_w)
    (act_address_map, weight_address_map) = ttl.dtx.conv_transform(
        [_nearest_y(C, 16), H, W],
        [_nearest_y(K, weight_block_width_datums), _nearest_y(C, 16), R, S],
        [R, S, stride_h, stride_w, pad_h, pad_w],
        act_block_height_datums,
        act_block_width_datums,
        weight_block_width_datums,
        num_blocks_act_h,
        num_blocks_weight_w,
        1,
        False,
    )

    # Run host side CPU function
    out_pytorch = blocked_mm_with_conv_act(
        A_cl.buffer(),
        B_tiled.buffer(),
        act_address_map,
        weight_address_map,
        num_blocks_act_h,
        num_blocks_act_w,
        num_blocks_weight_w,
        act_block_h,
        act_block_w,
        weight_block_w,
    )
    assert list(out_pytorch.shape) == mm_output_shape
    out_pytorch = out_pytorch[:, :, 0 : (OH * OW), 0:K]

    # Convert matmul output layout to conv output layout
    out_tr = torch.transpose(out_pytorch, 2, 3)
    assert list(out_tr.shape) == [1, 1, K, (OH * OW)]
    out_result = out_tr.reshape([1, K, OH, OW])

    # Compare against pytorch golden result
    out_golden = pytorch_inputs_and_golden[2]
    assert out_result.shape == out_golden.shape
    passing_pcc, output_pcc = comp_pcc(out_golden, out_result, 0.99)
    logger.debug(f"Passing={passing_pcc}")
    logger.debug(f"Output pcc={output_pcc}")
    return passing_pcc


def test_sweep_conv_with_dtx_cpu():
    test_bench = generate_conv_tb()
    pytorch_conv_golden_tb = generate_conv_tb_with_pytorch_golden(test_bench)
    passing = True
    full_op_compute_passing_tests = []
    input_tensor_only_passing_tests = []
    failing_tests = []
    i = 0
    for (
        conv_op_test_params,
        pytorch_inputs_and_golden,
    ) in pytorch_conv_golden_tb.items():
        passing_ = run_conv_as_large_matmul_dtx_cpu(conv_op_test_params, pytorch_inputs_and_golden)
        if passing_:
            if conv_op_test_params.test_level == TestLevel.INPUT_TENSOR_CREATE:
                input_tensor_only_passing_tests.append(conv_op_test_params)
            else:
                full_op_compute_passing_tests.append(conv_op_test_params)
        else:
            failing_tests.append(conv_op_test_params)
            print("Failed test - ")
            conv_op_test_params.print("   ")
            assert False
        passing &= passing_
        i += 1
        if i == 10:
            # only running first 10 tests from the sweep test
            # we run the full sweep on hardware
            break
    print("Following tests that create only input tensors passed - ")
    for conv_op_test_params in input_tensor_only_passing_tests:
        conv_op_test_params.print("   ")
    print("Following tests that rull full op compute passed - ")
    for conv_op_test_params in full_op_compute_passing_tests:
        conv_op_test_params.print("   ")
    print("Following tests failed - ")
    for conv_op_test_params in failing_tests:
        conv_op_test_params.print("   ")
    print(str(len(input_tensor_only_passing_tests)) + ' "INPUT TENSORS CREATION" tests PASSED.')
    print(str(len(full_op_compute_passing_tests)) + ' "FULL OP COMPUTE" tests PASSED.')
    print(str(len(failing_tests)) + ' "FULL OP COMPUTE" tests FAILED.')
    # assert passing
