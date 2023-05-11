import pytest
from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../../..")

import numpy as np
import tt_lib as ttl
from tt_lib.utils import blocked_mm_with_conv_act, _nearest_32
from python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from python_api_testing.conv.pytorch_conv_tb import TestLevel, generate_conv_tb_with_pytorch_golden, generate_conv_tb
from tests.python_api_testing.conv.conv_utils import create_conv_act_tensor, create_conv_weight_tensor
import torch

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
    OH = ((int) ((H - R + 2 * pad_h) / stride_h)) + 1
    OW = ((int) ((W - S + 2 * pad_w) / stride_w)) + 1

    #torch.manual_seed(0)
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    a_activation_shape = [1,C,H,W]
    b_weights_shape = [K,C,R,S]
    mm_output_shape = [1,1,_nearest_32(OH*OW),_nearest_32(K)]

    A_pyt = pytorch_inputs_and_golden[0]
    A_cl = create_conv_act_tensor(A_pyt, 1, C, H, W)

    # Prepare weights
    B_pyt = pytorch_inputs_and_golden[1]
    B_tiled_ = create_conv_weight_tensor(B_pyt, K, C, R, S)
    B_tiled_data = B_tiled_.data()
    if(conv_op_test_params.test_level == TestLevel.INPUT_TENSOR_CREATE):
        return True
    assert(conv_op_test_params.test_level == TestLevel.OP_FULL_COMPUTE)
    A_cl_data = A_cl.data()
    # Call DTX pass to transform A
    matrix_activation_h_tiles = (int) (_nearest_32(OH*OW) / 32)
    matrix_weight_w_tiles = (int) (_nearest_32(K) / 32)
    matrix_activation_w_tiles = (int) (_nearest_32(C)*R*S/32)
    # hardcode num of blocks
    num_blocks_in0_w = matrix_activation_w_tiles
    num_blocks_in0_h = matrix_activation_h_tiles
    num_blocks_in1_w = matrix_weight_w_tiles
    in0_block_h = 1
    in0_block_w = 1
    in1_block_w = 1
    dim_order = [0,1,2]
    in0_block_width_datums = (int) (_nearest_32(C)*R*S/num_blocks_in0_w)
    in0_block_height_datums = (int) (_nearest_32(OH*OW)/num_blocks_in0_h)
    block_shape_yx = [in0_block_height_datums, in0_block_width_datums]
    address_map = ttl.dtx.conv_transform([_nearest_32(C),H,W], [R,S,stride_h,stride_w,pad_h,pad_w], (dim_order,block_shape_yx), 1)

    in1_block_h = in0_block_w
    in1_tile_stride_h = matrix_weight_w_tiles
    in1_block_stride_h = matrix_weight_w_tiles * in1_block_h
    in1_block_stride_w = in1_block_w

    # Run host side CPU function
    out_pytorch = blocked_mm_with_conv_act(A_cl_data, B_tiled_data, address_map, num_blocks_in0_h, num_blocks_in0_w,
                                    num_blocks_in1_w, in0_block_h, in0_block_w, in1_block_w, in1_tile_stride_h, in1_block_stride_h, in1_block_stride_w)
    assert(list(out_pytorch.shape) == mm_output_shape)
    out_pytorch = out_pytorch[:, :, 0 : (OH * OW), 0 : K]

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
    i = 0
    for conv_op_test_params, pytorch_inputs_and_golden in pytorch_conv_golden_tb.items():
        passing_ = run_conv_as_large_matmul(conv_op_test_params, pytorch_inputs_and_golden)
        if passing_:
            if conv_op_test_params.test_level == TestLevel.INPUT_TENSOR_CREATE:
                input_tensor_only_passing_tests.append(conv_op_test_params)
            else:
                full_op_compute_passing_tests.append(conv_op_test_params)
        else:
            failing_tests.append(conv_op_test_params)
            print("Failed test - ")
            conv_op_test_params.print("   ")
            assert(False)
        passing &= passing_
        i+=1
        if(i == 10):
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
    print(str(len(input_tensor_only_passing_tests)) + " \"INPUT TENSORS CREATION\" tests PASSED.")
    print(str(len(full_op_compute_passing_tests)) + " \"FULL OP COMPUTE\" tests PASSED.")
    print(str(len(failing_tests)) + " \"FULL OP COMPUTE\" tests FAILED.")
    #assert passing
