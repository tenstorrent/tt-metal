# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
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
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose_and_pcc,
    comp_pcc,
)
from models.utility_functions import is_wormhole_b0
from tests.tt_eager.python_api_testing.conv.conv_unit_test_utils import (
    create_conv_act_tensor,
    create_conv_act_tensor_special,
    create_conv_weight_tensor,
    create_conv_weight_tensor_special_special,
    create_conv_bias_tensor,
)
import torch


@pytest.mark.parametrize("untilize_out", (False,))
@pytest.mark.parametrize("has_bias", (True,))
@pytest.mark.parametrize("fuse_relu", (True,))
@pytest.mark.parametrize(
    "N",
    (
        1,
        2,
        8,
    ),
)
@pytest.mark.parametrize("extra_padding_for_32B_alignment", (25,))
@pytest.mark.parametrize("sharded_out", (True, False))
def test_resnet50_first_conv(
    device,
    use_program_cache,
    N,
    extra_padding_for_32B_alignment,
    untilize_out,
    has_bias,
    fuse_relu,
    sharded_out,
):
    pytest.skip("This conv path is deprecated and will be removed during conv refactor efforts")
    compute_grid_size = device.compute_with_storage_grid_size()
    is_e75_grid_size = (compute_grid_size.x * compute_grid_size.y) == 88
    if N == 8 and is_e75_grid_size:
        pytest.skip(
            f"Skipping batch 8 on E75 because expected grid size is 12x9 but E75 grid size is {compute_grid_size}"
        )
    if N == 2:
        pytest.skip("Skipping batch 2 due to pcc error!")
    if N == 8 and is_wormhole_b0():
        pytest.skip("Parallelization unsupported for WH B0")
    if sharded_out and N != 8:
        pytest.skip("Tensor sharding unsupported for shape")
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

    if has_bias and untilize_out:
        ## bias is only supported without untilize out
        pytest.skip()

    num_iterations = 1  # run twice to test op caching flow for conv op
    for i in range(num_iterations):
        # torch.set_printoptions(threshold=10000)
        torch.manual_seed(0)
        a_activation_shape = [N, C, H, W]
        A_pyt = torch.randn(a_activation_shape, dtype=torch.bfloat16).float()
        b_weights_shape = [K, C, R, S]
        B_pyt = torch.randn(b_weights_shape, dtype=torch.bfloat16).float()
        bias_shape = [1, 1, 1, K]
        bias_pyt = torch.randn(bias_shape)

        # Parameters to define block dims
        # [128, 32], [32, 64], [128, 64]
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

        if N == 1:
            act_block_h_datums = 256
            grid_size = (7, 7)
            per_core_out_h_ntiles = 8
        elif N == 2:
            act_block_h_datums = 256
            grid_size = (7, 7)
            per_core_out_h_ntiles = 16
        elif N == 8:
            act_block_h_datums = 1024
            grid_size = (12, 9)
            per_core_out_h_ntiles = 32
            # act_block_h_datums = 256
            # grid_size = (7,8)
            # per_core_out_h_ntiles = 56
        act_block_h = (int)(act_block_h_datums / 32)
        out_block_h = act_block_h

        # Prepare activations

        A_cl_host = create_conv_act_tensor_special(
            A_pyt,
            N,
            C,
            H,
            W,
            pad_h,
            pad_w,
            extra_pad_w_right=1 + extra_padding_for_32B_alignment,
        )
        print("A_cl_host shape", A_cl_host.get_legacy_shape())
        memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

        # save original shape (N, H, W, C)
        original_A_cl_host_shape = A_cl_host.get_legacy_shape()

        # re-shape to (N, H, 1, W*C)
        A_cl_host = A_cl_host.reshape(
            A_cl_host.get_legacy_shape()[0],
            A_cl_host.get_legacy_shape()[1],
            1,
            A_cl_host.get_legacy_shape()[2] * A_cl_host.get_legacy_shape()[3],
        )
        print("A_cl_host shape after re-shape (only for transfer)", A_cl_host.get_legacy_shape())
        A_cl_device = A_cl_host.to(device, memory_config)

        print(original_A_cl_host_shape)
        # re-shape back to original shape (N, H, W, C)
        A_cl_device = A_cl_device.reshape(
            original_A_cl_host_shape[0],
            original_A_cl_host_shape[1],
            original_A_cl_host_shape[2],
            original_A_cl_host_shape[3],
        )
        print("A_cl_device shape into OP", A_cl_device.get_legacy_shape())

        # Prepare weights
        B_tiled_host = create_conv_weight_tensor_special_special(
            B_pyt, K, C, R, S, weight_block_h, weight_block_w, padded_S
        )
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
        per_core_weight_matrix_w_ntiles = (int)(K / 32)
        output_mem_config = (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1) if sharded_out else None
        )
        out = ttnn.operations.conv2d.optimized_conv(
            A_cl_device,
            B_tiled,
            bias_device,
            None,
            [R, padded_S, stride_h, stride_w, 0, 0],
            K,
            untilize_out,
            has_bias,
            fuse_relu,
            ttnn.MathFidelity.HiFi4,
            ttnn.operations.conv2d.OptimizedConvParallelizationConfig(
                grid_size=grid_size,
                num_cores_nhw=grid_size[0],
                per_core_out_matrix_height_ntiles=per_core_out_h_ntiles,
                per_core_out_matrix_width_ntiles=per_core_weight_matrix_w_ntiles,
            ),
            ttnn.operations.conv2d.OptimizedConvBlockConfig(
                act_block_h_ntiles=act_block_h,
                act_block_w_ntiles=act_block_w,
                out_subblock_h_ntiles=out_subblock_h,
                out_subblock_w_ntiles=out_subblock_w,
            ),
            extra_padding_for_32B_alignment,
            output_mem_config,
        )
        if sharded_out:
            out = ttnn.sharded_to_interleaved(out, memory_config)

        if not untilize_out:
            out_unpadded_shape = [1, 1, N * OH * OW, K]
            assert out_unpadded_shape == list(out.shape_without_padding())
            out = ttnn.experimental.tensor.format_output_tensor(
                out, out.shape_without_padding(), device, ttnn.ROW_MAJOR_LAYOUT
            )
            out = out.reshape(
                conv_output_shape[0],
                conv_output_shape[1],
                conv_output_shape[2],
                conv_output_shape[3],
            )
        out = out.cpu()
        assert list(out.get_legacy_shape()) == conv_output_shape
        assert out.get_layout() == ttnn.ROW_MAJOR_LAYOUT

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

        # Sanity check for relu
        if fuse_relu:
            print("Relu enabled. Check for all values >= 0")
            out_bool = out_result >= 0
            out_nonzero = out_result < 0
            indices = out_nonzero.nonzero()
            print("Printing non zero indices -")
            print(indices)
            all_non_neg = torch.all(out_bool)
            assert all_non_neg

        # Compare against golden
        golden_pcc = 0.9999

        passing_pcc, output_pcc = comp_pcc(out_golden, out_result, golden_pcc)
        logger.debug(f"Passing={passing_pcc}")
        logger.debug(f"Output pcc={output_pcc}")
        assert passing_pcc
