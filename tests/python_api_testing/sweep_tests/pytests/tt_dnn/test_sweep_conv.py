import pytest
from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../../..")

import numpy as np

from libs import tt_lib as ttl
from libs.tt_lib.utils import tilize_to_list, tilize, untilize, channels_last, _nearest_32, convert_weights_2d_matrix
from python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from python_api_testing.conv.pytorch_conv_tb import TestLevel, generate_conv_tb_with_pytorch_golden, generate_conv_tb

import torch
from time import sleep

def run_conv_on_device(A, B_tiled, conv_params, untilize_out):
    return ttl.tensor.conv(A, B_tiled, conv_params, untilize_out)


def run_conv_as_large_matmul(conv_op_test_params, pytorch_inputs_and_golden):
    print("Testing convolution with following parameters - ")
    conv_op_test_params.print("   ")
    ctp = conv_op_test_params.conv_params
    N = ctp.act_shape[0]
    C = ctp.act_shape[1]
    H = ctp.act_shape[2]
    W = ctp.act_shape[3]
    K = ctp.weight_shape[0]
    assert(ctp.weight_shape[1] == C)
    R = ctp.weight_shape[2]
    S = ctp.weight_shape[3]
    stride_h = ctp.stride_h;
    stride_w = ctp.stride_w;
    pad_h = ctp.pad_h;
    pad_w = ctp.pad_w;
    # check if params are valid
    assert (H - R + 2 * pad_h) >= 1 and (W - S + 2 * pad_w) >= 1
    OH = ((int) ((H - R + 2 * pad_h) / stride_h)) + 1
    OW = ((int) ((W - S + 2 * pad_w) / stride_w)) + 1

    #torch.manual_seed(0)
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    a_activation_shape = [1,C,H,W]
    b_weights_shape = [K,C,R,S]
    mm_output_shape = [1,1,_nearest_32(OH*OW),K]

    A_pyt = pytorch_inputs_and_golden[0]
    A_ = ttl.tensor.Tensor(
        torch.flatten(A_pyt).tolist(),
        a_activation_shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR)
    A_cl = A_.to(ttl.tensor.Layout.CHANNELS_LAST)
    A = A_cl.to(device, ttl.tensor.MemoryConfig(False, 0))

    # Prepare weights
    B_pyt = pytorch_inputs_and_golden[1]
    B_ = ttl.tensor.Tensor(
        torch.flatten(B_pyt).tolist(),
        b_weights_shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR
    )
    B_tiled_ = ttl.tensor.convert_conv_weight_tensor_to_tiled_layout(B_)
    assert(B_tiled_.shape() == [1, 1, C*R*S, K])
    B_tiled = B_tiled_.to(device)
    if(conv_op_test_params.test_level == TestLevel.INPUT_TENSOR_CREATE):
        return True
    assert(conv_op_test_params.test_level == TestLevel.OP_FULL_COMPUTE)
    # Run TT metal OP

    out = run_conv_on_device(A, B_tiled, [R,S,stride_h,stride_w,pad_h,pad_w], True)
    assert(out.shape() == mm_output_shape)
    # Copy output to host and convert tt tensor to pytorch tensor
    out_pytorch = torch.tensor(out.to(host).data()).reshape(mm_output_shape)
    ttl.device.CloseDevice(device)
    # remove padding
    out_pytorch = out_pytorch[:, :, 0 : (OH * OW), :]

    # Convert matmul output layout to conv output layout
    out_tr = torch.transpose(out_pytorch, 2, 3)
    assert(list(out_tr.shape) == [1,1,K,(OH*OW)])
    out_result = out_tr.reshape([1,K,OH,OW])

    # Compare against pytorch golden result
    out_golden = pytorch_inputs_and_golden[2]
    assert(out_result.shape == out_golden.shape)
    passing_pcc, output_pcc = comp_pcc(out_golden, out_result, 0.99)
    print("Passing=", passing_pcc)
    print("Output pcc=", output_pcc)
    return passing_pcc

def test_sweep_conv():
    test_bench = generate_conv_tb()
    pytorch_conv_golden_tb = generate_conv_tb_with_pytorch_golden(test_bench)
    passing = True
    full_op_compute_passing_tests = []
    input_tensor_only_passing_tests = []
    failing_tests = []
    failing_tests_exception = []
    for conv_op_test_params, pytorch_inputs_and_golden in pytorch_conv_golden_tb.items():
        try:
            passing_ = run_conv_as_large_matmul(conv_op_test_params, pytorch_inputs_and_golden)
            if passing_:
                if conv_op_test_params.test_level == TestLevel.INPUT_TENSOR_CREATE:
                    input_tensor_only_passing_tests.append(conv_op_test_params)
                else:
                    full_op_compute_passing_tests.append(conv_op_test_params)
            else:
                failing_tests_exception.append(conv_op_test_params)
                print("Failed test - ")
                conv_op_test_params.print("   ")
                #assert(False)
        except:
            failing_tests.append(conv_op_test_params)
            passing_ = False
        passing &= passing_
    print("Following tests that create only input tensors passed - ")
    for conv_op_test_params in input_tensor_only_passing_tests:
        conv_op_test_params.print("   ")
    print("Following tests that rull full op compute passed - ")
    for conv_op_test_params in full_op_compute_passing_tests:
        conv_op_test_params.print("   ")
    print("Following tests failed with incorrect mismatch - ")
    for conv_op_test_params in failing_tests:
        conv_op_test_params.print("   ")
    print("Following tests failed with exception/error - ")
    for conv_op_test_params in failing_tests_exception:
        conv_op_test_params.print("   ")
    print(str(len(input_tensor_only_passing_tests)) + " \"INPUT TENSORS CREATION\" tests PASSED.")
    print(str(len(full_op_compute_passing_tests)) + " \"FULL OP COMPUTE\" tests PASSED.")
    print(str(len(failing_tests)) + " \"FULL OP COMPUTE\" tests FAILED.")
    #assert passing
