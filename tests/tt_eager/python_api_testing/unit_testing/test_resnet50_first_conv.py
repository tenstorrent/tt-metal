"""
SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

SPDX-License-Identifier: Apache-2.0
"""

import pytest
from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")

import numpy as np

import tt_lib as ttl
from tt_lib.utils import (
    tilize_to_list,
    tilize,
    untilize,
    _nearest_32,
    _nearest_y,
    convert_weights_2d_matrix,
)
from models.utility_functions import print_diff_argmax, is_close, comp_pcc
from tests.tt_eager.python_api_testing.conv.conv_unit_test_utils import (
    create_conv_act_tensor,
    create_conv_act_tensor_special,
    create_conv_weight_tensor,
    create_conv_weight_tensor_special_special,
)
import torch

@pytest.mark.parametrize("untilize_out", (False,))
@pytest.mark.parametrize("N", (1,2,8))
@pytest.mark.parametrize("extra_padding_for_32B_alignment", (25,))
@pytest.mark.skip(reason="Conv disabled in main.")
def test_resnet50_first_conv(use_program_cache, N, extra_padding_for_32B_alignment, device, untilize_out):
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
        3
    )

    num_iterations = 1  # run twice to test op caching flow for conv op
    for i in range(num_iterations):
        # torch.set_printoptions(threshold=10000)
        torch.manual_seed(0)
        a_activation_shape = [N, C, H, W]
        A_pyt = torch.randn(a_activation_shape, dtype=torch.bfloat16).float()
        b_weights_shape = [K, C, R, S]
        B_pyt = torch.randn(b_weights_shape, dtype=torch.bfloat16).float()

        # Parameters to define block dims
        #[128, 32], [32, 64], [128, 64]
        assert padded_C * padded_S % 32 == 0
        act_block_w = (int)((padded_C * padded_S) / 32)
        weight_block_h = act_block_w
        weight_block_w = 2
        out_subblock_h = 1
        out_subblock_w = 2
        # pad filter from 7x7 to 7x8
        OH = ((int)((H - R + 2 * pad_h) / stride_h)) + 1
        OW = ((int)((W - padded_S + (2 * pad_w) + 1) / stride_w)) + 1
        conv_output_shape = [N, OH, OW, K]
        print(a_activation_shape)
        print(conv_output_shape)

        if(N == 1):
            act_block_h_datums = 256
            grid_size = (7,7)
            per_core_act_h_ntiles = 8
        elif(N == 2):
            act_block_h_datums = 256
            grid_size = (7,7)
            per_core_act_h_ntiles = 16
        elif(N == 8):
            act_block_h_datums = 256
            grid_size = (7,7)
            per_core_act_h_ntiles = 64
        act_block_h = (int) (act_block_h_datums / 32)

        # Prepare activations

        A_cl_host = create_conv_act_tensor_special(
            A_pyt, N, C, H, W, pad_h, pad_w, extra_pad_w_right=1 + extra_padding_for_32B_alignment
        )
        print("A_cl_host shape", A_cl_host.shape())
        memory_config = ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1)

        # save original shape (N, H, W, C)
        original_A_cl_host_shape = A_cl_host.shape()

        # re-shape to (N, H, 1, W*C)
        A_cl_host = A_cl_host.reshape(A_cl_host.shape()[0], A_cl_host.shape()[1], 1, A_cl_host.shape()[2] * A_cl_host.shape()[3])
        print("A_cl_host shape after re-shape (only for transfer)", A_cl_host.shape())
        A_cl_device = A_cl_host.to(device, memory_config)

        print(original_A_cl_host_shape)
        # re-shape back to original shape (N, H, W, C)
        A_cl_device = A_cl_device.reshape(original_A_cl_host_shape[0], original_A_cl_host_shape[1], original_A_cl_host_shape[2], original_A_cl_host_shape[3])
        print("A_cl_device shape into OP", A_cl_device.shape())

        # Prepare weights
        B_tiled_host = create_conv_weight_tensor_special_special(
            B_pyt, K, C, R, S, weight_block_h, weight_block_w, padded_S
        )
        B_tiled = B_tiled_host.to(device)
        # Calculate conv result with golden result. Run Pytorch conv
        out_golden = torch.nn.functional.conv2d(
            A_pyt, B_pyt, stride=(stride_h, stride_w), padding=(pad_h, pad_w)
        )

        # Run TT metal OP
        out = ttl.tensor.optimized_conv(
            A_cl_device,
            B_tiled,
            [R, padded_S, stride_h, stride_w, 0, 0],
            act_block_h,
            act_block_w,
            weight_block_w,
            out_subblock_h,
            out_subblock_w,
            K,
            untilize_out,
            False,
            False,
            ttl.tensor.MathFidelity.HiFi4,
            ttl.tensor.OptimizedConvParallelizationConfig(grid_size=grid_size, per_core_act_matrix_height_ntiles=per_core_act_h_ntiles),
            extra_padding_for_32B_alignment
        )
        if not untilize_out:
           out_unpadded_shape = [1, 1, N*OH*OW, K]
           assert out_unpadded_shape == out.shape_without_padding()
           out = ttl.tensor.format_output_tensor(out, out.shape_without_padding(), device, ttl.tensor.Layout.ROW_MAJOR)
           out = out.reshape(conv_output_shape[0], conv_output_shape[1], conv_output_shape[2], conv_output_shape[3])
        out = out.cpu()
        assert out.shape() == conv_output_shape
        assert out.layout() == ttl.tensor.Layout.ROW_MAJOR

        # Copy output to host and convert tt tensor to pytorch tensor
        out_result = out.to_torch()
        out_result = torch.transpose(out_result, 2, 3)
        out_result = torch.transpose(out_result, 1, 2)

        assert out_result.shape == out_golden.shape

        # Debug
        # out_result_first_image = out_result[0][:][:][:]
        # out_golden_first_image = out_golden[0][:][:][:]
        # first_pcc, _ = comp_pcc(out_golden_first_image, out_result_first_image, pcc=0.9998)
        # assert first_pcc
        # out_result_sec_image = out_result[1][:][:][:]
        # out_golden_sec_image = out_golden[1][:][:][:]
        # sec_pcc, _ = comp_pcc(out_golden_sec_image, out_result_sec_image, pcc=0.9998)
        # assert sec_pcc

        # Compare against golden
        passing_pcc, output_pcc = comp_pcc(out_golden, out_result, 0.99)
        print("Passing=", passing_pcc)
        print("Output pcc=", output_pcc)
        assert passing_pcc
