# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch
from loguru import logger


def test_global_avg_pool2d(device):
    # Create a random input tensor
    tensor = ttnn.rand((10, 3, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Perform global average pooling
    output = ttnn.global_avg_pool2d(tensor)
    logger.info(f"Output: {output}")


def test_max_pool2d():
    device = ttnn.CreateDevice(0, l1_small_size=8192)
    # Define input parameters
    kernel_h, kernel_w = 2, 2
    stride_h, stride_w = 1, 1
    pad_h, pad_w = 0, 0
    dilation_h, dilation_w = 1, 1
    nchw_shape = (4, 256, 40, 40)
    in_N, in_C, in_H, in_W = nchw_shape
    input_shape = (1, 1, in_N * in_H * in_W, in_C)

    # Create a random input tensor
    input = torch.randn(nchw_shape, dtype=torch.bfloat16)
    input_perm = torch.permute(input, (0, 2, 3, 1))  # this op expects a [N, H, W, C] format
    input_reshape = input_perm.reshape(input_shape)

    tt_input = ttnn.from_torch(input_reshape, ttnn.bfloat16)
    tt_input_dev = ttnn.to_device(tt_input, device)

    # Perform max pooling
    tt_output = ttnn.max_pool2d(
        input_tensor=tt_input_dev,
        batch_size=in_N,
        input_h=in_H,
        input_w=in_W,
        channels=in_C,
        kernel_size=[kernel_h, kernel_w],
        stride=[stride_h, stride_w],
        padding=[pad_h, pad_w],
        dilation=[dilation_h, dilation_w],
        ceil_mode=False,
        memory_config=None,
        applied_shard_scheme=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        deallocate_input=False,
        reallocate_halo_output=True,
        dtype=ttnn.bfloat16,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    logger.info(f"Output: {tt_output}")
    ttnn.close_device(device)


def test_avg_pool2d():
    device = ttnn.CreateDevice(0, l1_small_size=8192)
    # Define input parameters
    kernel_h, kernel_w = 2, 2
    stride_h, stride_w = 1, 1
    pad_h, pad_w = 0, 0
    nchw_shape = (4, 256, 40, 40)
    in_N, in_C, in_H, in_W = nchw_shape
    input_shape = (1, 1, in_N * in_H * in_W, in_C)

    # Create a random input tensor
    input = torch.randn(nchw_shape, dtype=torch.bfloat16)
    input_perm = torch.permute(input, (0, 2, 3, 1))  # this op expects a [N, H, W, C] format
    input_reshape = input_perm.reshape(input_shape)  # this op expects [1, 1, NHW, C]
    tt_input = ttnn.from_torch(input_reshape, device=device)

    # Perform average pooling
    tt_output = ttnn.avg_pool2d(
        input_tensor=tt_input,
        batch_size=in_N,
        input_h=in_H,
        input_w=in_W,
        channels=in_C,
        kernel_size=[kernel_h, kernel_w],
        stride=[stride_h, stride_w],
        padding=[pad_h, pad_w],
        ceil_mode=False,
        count_include_pad=True,
        divisor_override=None,
        memory_config=None,
        applied_shard_scheme=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        deallocate_input=False,
        reallocate_halo_output=True,
        dtype=ttnn.bfloat16,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        compute_kernel_config=None,
    )
    logger.info(f"Output: {tt_output}")
    ttnn.close_device(device)
