import pytest
from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")

import numpy as np
import tt_lib as ttl
from tt_lib.utils import _nearest_32, _nearest_y
from python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from python_api_testing.conv.pytorch_conv_tb import (
    TestLevel,
    generate_conv_tb_with_pytorch_golden,
    generate_conv_tb,
)
from tests.python_api_testing.conv.conv_utils import (
    create_conv_act_tensor,
    create_conv_weight_tensor,
)

import torch
from time import sleep


def run_conv_as_large_matmul(conv_op_test_params, pytorch_inputs_and_golden):
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

    # torch.manual_seed(0)
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    ttl.device.SetDefaultDevice(device)
    host = ttl.device.GetHost()

    A_pyt = pytorch_inputs_and_golden[0]
    B_pyt = pytorch_inputs_and_golden[1]

    # Parameters to define block dims
    act_block_h = 4
    act_block_w = 4
    weight_block_h = act_block_w
    weight_block_w = 4
    out_subblock_h = 4
    out_subblock_w = 2

    OH = ((int) ((H - R + 2 * pad_h) / stride_h)) + 1
    OW = ((int) ((W - S + 2 * pad_w) / stride_w)) + 1
    conv_output_shape = [1,K, OH, OW]

    # Prepare activations
    A_cl_host = create_conv_act_tensor(A_pyt, 1, C, H, W)
    A_cl_data = A_cl_host.data()
    A = A_cl_host.to(device)

    # Prepare weights
    B_tiled_host = create_conv_weight_tensor(
        B_pyt, K, C, R, S, weight_block_h, weight_block_w
    )
    B_tiled = B_tiled_host.to(device)

    if conv_op_test_params.test_level == TestLevel.INPUT_TENSOR_CREATE:
        print("Ran test till tensor creation only. Did not run full op compute.")
        return True
    assert conv_op_test_params.test_level == TestLevel.OP_FULL_COMPUTE

    # Run TT metal OP
    out = ttl.tensor.conv(
        A,
        B_tiled,
        [R, S, stride_h, stride_w, pad_h, pad_w],
        act_block_h,
        act_block_w,
        weight_block_w,
        out_subblock_h,
        out_subblock_w,
        K
    )
    out = out.to(host)
    assert(out.shape() == conv_output_shape)
    assert(out.layout() == ttl.tensor.Layout.CHANNELS_LAST)

    # Copy output to host and convert tt tensor to pytorch tensor
    conv_output_shape_cl = [1,OH,OW,K]
    out_result = torch.tensor(out.data()).reshape(conv_output_shape_cl)
    out_result = torch.transpose(out_result, 2, 3)
    out_result = torch.transpose(out_result, 1, 2)

    # Compare against pytorch golden result
    out_golden = pytorch_inputs_and_golden[2]
    assert out_result.shape == out_golden.shape
    passing_pcc, output_pcc = comp_pcc(out_golden, out_result, 0.99)
    print("Passing=", passing_pcc)
    print("Output pcc=", output_pcc)

    ttl.device.CloseDevice(device)
    return passing_pcc


def test_sweep_conv_tt():
    test_bench = generate_conv_tb()
    pytorch_conv_golden_tb = generate_conv_tb_with_pytorch_golden(test_bench)
    passing = True
    full_op_compute_passing_tests = []
    input_tensor_only_passing_tests = []
    input_tensor_only_failing_tests = []
    input_tensor_only_failing_tests_exception = []
    full_op_compute_failing_tests = []
    full_op_compute_failing_tests_with_exception = []
    input_tensor_only_tests = 0
    full_op_compute_tests = 0
    for (
        conv_op_test_params,
        pytorch_inputs_and_golden,
    ) in pytorch_conv_golden_tb.items():
        passing_tests = full_op_compute_passing_tests
        failing_tests = full_op_compute_failing_tests
        failing_tests_with_exception = full_op_compute_failing_tests_with_exception
        if conv_op_test_params.test_level == TestLevel.INPUT_TENSOR_CREATE:
            passing_tests = input_tensor_only_passing_tests
            failing_tests = input_tensor_only_failing_tests
            failing_tests_with_exception = input_tensor_only_failing_tests_exception
            input_tensor_only_tests += 1
        else:
            assert conv_op_test_params.test_level == TestLevel.OP_FULL_COMPUTE
            full_op_compute_tests += 1
        try:
            passing_ = run_conv_as_large_matmul(
                conv_op_test_params, pytorch_inputs_and_golden
            )
            if passing_:
                passing_tests.append(conv_op_test_params)
            else:
                failing_tests.append(conv_op_test_params)
                print("Failed test - ")
                conv_op_test_params.print("   ")
        except Exception as e:
            print("Exception error: " + str(e))
            failing_tests_with_exception.append(conv_op_test_params)
            passing_ = False
        passing &= passing_
    print("Following tests that create only input tensors passed - ")
    for conv_op_test_params in input_tensor_only_passing_tests:
        conv_op_test_params.print("   ")
    print(
        "Following tests that create only input tensors failed with exception/error - "
    )
    for conv_op_test_params in input_tensor_only_failing_tests_exception:
        conv_op_test_params.print("   ")
    print("Following tests that ran full op compute passed - ")
    for conv_op_test_params in full_op_compute_passing_tests:
        conv_op_test_params.print("   ")
    print("Following tests that ran full op compute failed with incorrect mismatch - ")
    for conv_op_test_params in full_op_compute_failing_tests:
        conv_op_test_params.print("   ")
    print("Following tests that ran full op compute failed with exception/error - ")
    for conv_op_test_params in full_op_compute_failing_tests_with_exception:
        conv_op_test_params.print("   ")

    print(
        str(len(input_tensor_only_passing_tests))
        + " out of "
        + str(input_tensor_only_tests)
        + ' "INPUT TENSORS CREATION" tests PASSED.'
    )
    print(
        str(len(input_tensor_only_failing_tests_exception))
        + " out of "
        + str(input_tensor_only_tests)
        + ' "INPUT TENSORS CREATION" tests FAILED with exception.'
    )

    print(
        str(len(full_op_compute_passing_tests))
        + " out of "
        + str(full_op_compute_tests)
        + ' "FULL OP COMPUTE" tests PASSED.'
    )
    print(
        str(len(full_op_compute_failing_tests))
        + " out of "
        + str(full_op_compute_tests)
        + ' "FULL OP COMPUTE" tests FAILED due to mismatch with golden output.'
    )
    print(
        str(len(full_op_compute_failing_tests_with_exception))
        + " out of "
        + str(full_op_compute_tests)
        + ' "FULL OP COMPUTE" tests FAILED with exception/error.'
    )
    assert passing
