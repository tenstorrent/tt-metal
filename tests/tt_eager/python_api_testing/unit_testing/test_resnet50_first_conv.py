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
    create_conv_weight_tensor,
    create_conv_weight_tensor_special_padding,
)
import torch

@pytest.mark.parametrize("untilize_out", (False,True))
def test_resnet50_first_conv(use_program_cache, device, untilize_out):
    (K, C, padded_C, H, W, R, S, padded_S, stride_h, stride_w, pad_h, pad_w) = (
        64,
        3,
        16,
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
        a_activation_shape = [1, C, H, W]
        A_pyt = torch.randn(a_activation_shape, dtype=torch.bfloat16).float()
        b_weights_shape = [K, C, R, S]
        B_pyt = torch.randn(b_weights_shape, dtype=torch.bfloat16).float()

        # Parameters to define block dims
        act_block_h = 4
        assert padded_C * padded_C % 32 == 0
        act_block_w = (int)((padded_C * padded_S) / 32)
        weight_block_h = act_block_w
        weight_block_w = 2
        out_subblock_h = 4
        out_subblock_w = 2
        # pad filter from 7x7 to 7x8
        OH = ((int)((H - R + 2 * pad_h) / stride_h)) + 1
        OW = ((int)((W - padded_S + (2 * pad_w) + 1) / stride_w)) + 1
        conv_output_shape = [1, OH, OW, K]

        # Prepare activations
        A_cl_host = create_conv_act_tensor(
            A_pyt, 1, C, H, W, pad_h, pad_w, extra_pad_w_right=1
        )
        memory_config = ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1)
        A = A_cl_host.to(device, memory_config)

        # Prepare weights
        B_tiled_host = create_conv_weight_tensor_special_padding(
            B_pyt, K, C, R, S, weight_block_h, weight_block_w, padded_S
        )
        B_tiled = B_tiled_host.to(device)
        # Calculate conv result with golden result. Run Pytorch conv
        out_golden = torch.nn.functional.conv2d(
            A_pyt, B_pyt, stride=(stride_h, stride_w), padding=(pad_h, pad_w)
        )

        # Run TT metal OP
        out = ttl.tensor.optimized_conv(
            A,
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
            ttl.tensor.MathFidelity.HiFi4
        )
        if not untilize_out:
           out_unpadded_shape = [1, 1, OH*OW, K]
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

        # Compare against golden
        assert out_result.shape == out_golden.shape
        passing_pcc, output_pcc = comp_pcc(out_golden, out_result, 0.99)
        print("Passing=", passing_pcc)
        print("Output pcc=", output_pcc)
        assert passing_pcc
