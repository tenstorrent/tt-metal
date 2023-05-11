import pytest
from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")

import numpy as np

from libs import tt_lib as ttl
from libs.tt_lib.utils import blocked_mm_with_conv_act, tilize_to_list, tilize, untilize, channels_last, _nearest_32, convert_weights_2d_matrix
from python_api_testing.models.utility_functions import print_diff_argmax, is_close, comp_pcc
from python_api_testing.conv.pytorch_conv_tb import TestLevel, generate_conv_tb_with_pytorch_golden, generate_conv_tb
from tests.python_api_testing.conv.conv_unit_test_utils import create_conv_act_tensor, create_conv_weight_tensor

import torch

@pytest.mark.parametrize(
    "K, C, H, W, R, S, stride_h, stride_w, pad_h, pad_w",
    (
        # channels padding
        (32, 3, 5, 5, 1, 1, 1, 1, 0, 0),
        # Hat = 1, Wat = 1, Wbt = 1
        (32, 32, 5, 5, 1, 1, 1, 1, 0, 0),
        # Hat = 2, Wat = 1, Wbt = 1
        (32, 32, 8, 8, 1, 1, 1, 1, 0, 0),
        # # Hat = 1, Wat = 2, Wbt = 1
        (32, 64, 5, 5, 1, 1, 1, 1, 0, 0),
        # # Hat = 2, Wat = 2, Wbt = 1
        (32, 64, 8, 8, 1, 1, 1, 1, 0, 0),
        # # Hat = 1, Wat = 1, Wbt = 2
        (64, 32, 5, 5, 1, 1, 1, 1, 0, 0),
        # # Hat = 1, Wat = 2, Wbt = 2
        (64, 64, 5, 5, 1, 1, 1, 1, 0, 0),
        # # Hat = 2, Wat = 1, Wbt = 2
        (64, 32, 8, 8, 1, 1, 1, 1, 0, 0),
        # # Hat = 2, Wat = 2, Wbt = 2
        (64, 64, 8, 8, 1, 1, 1, 1, 0, 0),
    ),
)
def test_run_conv_as_large_matmul_cpu(K, C, H, W, R, S, stride_h, stride_w, pad_h, pad_w):
    OH = ((int) ((H - R + 2 * pad_h) / stride_h)) + 1
    OW = ((int) ((W - S + 2 * pad_w) / stride_w)) + 1

    #torch.manual_seed(0)
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    a_activation_shape = [1,C,H,W]
    b_weights_shape = [K,C,R,S]

    mm_output_shape = [1,1,_nearest_32(OH*OW),_nearest_32(K)]
    act_shape_channel_padded = [1, _nearest_32(C), H, W]
    A_pyt = torch.randn(a_activation_shape, dtype=torch.bfloat16).float()
    A_cl = create_conv_act_tensor(A_pyt, 1, C, H, W)
    A_cl_data = A_cl.data()
    # Prepare weights
    B_pyt = torch.randn(b_weights_shape, dtype=torch.bfloat16).float()
    B_tiled_ = create_conv_weight_tensor(B_pyt, K, C, R, S)
    B_tiled_data = B_tiled_.data()
    # Call DTX pass to transform A
    matrix_activation_h_tiles = (int) (_nearest_32(OH*OW) / 32)
    matrix_weight_w_tiles = (int) (_nearest_32(K) / 32)
    matrix_activation_w_tiles = (int) (_nearest_32(C*R*S)/32)
    # hardcode num of blocks
    num_blocks_in0_w = matrix_activation_w_tiles
    num_blocks_in0_h = matrix_activation_h_tiles
    num_blocks_in1_w = matrix_weight_w_tiles
    in0_block_h = 1
    in0_block_w = 1
    in1_block_w = 1
    dim_order = [0,1,2]
    in0_block_width_datums = (int) (_nearest_32(C*R*S)/num_blocks_in0_w)
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

    # Calculate conv result with golden result. Run Pytorch conv
    out_golden = torch.nn.functional.conv2d(A_pyt, B_pyt, stride=(stride_h, stride_w), padding=(pad_h, pad_w))
    assert(out_result.shape == out_golden.shape)
    passing_pcc, output_pcc = comp_pcc(out_golden, out_result, 0.99)
    print("Passing=", passing_pcc)
    print("Output pcc=", output_pcc)
    assert passing_pcc
