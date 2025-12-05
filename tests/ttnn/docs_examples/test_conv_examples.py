# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch
from loguru import logger


def test_conv1d():
    device = ttnn.CreateDevice(0, l1_small_size=8192)
    # Define input parameters
    batch_size, in_channels, length = 1, 3, 80
    out_channels, kernel_size = 16, 3

    # Create input tensor [N, L, C] format
    torch_input = torch.randn(batch_size, length, in_channels, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(torch_input, device=device)

    # Create weight tensor [C_out, C_in, K]
    torch_weight = torch.randn(out_channels, in_channels, kernel_size, dtype=torch.bfloat16)
    tt_weight = ttnn.from_torch(torch_weight)

    # Perform 1D convolution
    output, _ = ttnn.conv1d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        device=device,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        input_length=length,
        kernel_size=kernel_size,
        stride=1,
        padding=1,
        groups=1,
    )
    logger.info(f"Conv1d output: {output}")
    ttnn.close_device(device)


def test_conv2d():
    device = ttnn.CreateDevice(0, l1_small_size=8192)
    # Define input parameters
    batch_size, in_channels, height, width = 1, 3, 32, 32
    out_channels, kernel_h, kernel_w = 16, 3, 3

    # Create input tensor in NHWC format
    torch_input = torch.randn(batch_size, height, width, in_channels, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(torch_input, device=device, layout=ttnn.TILE_LAYOUT)

    # Create weight and bias tensors
    torch_weight = torch.randn(out_channels, in_channels, kernel_h, kernel_w, dtype=torch.bfloat16)
    tt_weight = ttnn.from_torch(torch_weight)

    torch_bias = torch.randn(1, 1, 1, out_channels, dtype=torch.bfloat16)
    tt_bias = ttnn.from_torch(torch_bias)

    # Configure conv2d
    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    )

    # Perform 2D convolution
    [output, _, [_, _]] = ttnn.conv2d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        bias_tensor=tt_bias,
        in_channels=in_channels,
        out_channels=out_channels,
        device=device,
        kernel_size=(kernel_h, kernel_w),
        stride=(1, 1),
        padding=(1, 1),
        batch_size=batch_size,
        input_height=height,
        input_width=width,
        conv_config=conv_config,
    )
    logger.info(f"Conv2d output: {output}")
    ttnn.close_device(device)


def test_conv_transpose2d():
    device = ttnn.CreateDevice(0, l1_small_size=8192)
    # Define input parameters
    batch_size, in_channels, height, width = 1, 32, 16, 16
    out_channels, kernel_h, kernel_w = 16, 2, 2

    # Create input tensor in NHWC format
    torch_input = torch.randn(batch_size, height, width, in_channels, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(torch_input, device=device, layout=ttnn.TILE_LAYOUT)

    # Create weight and bias tensors (note: weights shape is [C_in, C_out, K_h, K_w] for transpose)
    torch_weight = torch.randn(in_channels, out_channels, kernel_h, kernel_w, dtype=torch.bfloat16)
    tt_weight = ttnn.from_torch(torch_weight)

    torch_bias = torch.randn(1, 1, 1, out_channels, dtype=torch.bfloat16)
    tt_bias = ttnn.from_torch(torch_bias)

    # Configure conv_transpose2d
    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    )

    # Perform 2D transposed convolution
    [output, _, [_, _]] = ttnn.conv_transpose2d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        bias_tensor=tt_bias,
        in_channels=in_channels,
        out_channels=out_channels,
        device=device,
        kernel_size=(kernel_h, kernel_w),
        stride=(2, 2),
        padding=(0, 0),
        output_padding=(0, 0),
        batch_size=batch_size,
        input_height=height,
        input_width=width,
        conv_config=conv_config,
    )
    logger.info(f"Conv_transpose2d output: {output}")
    ttnn.close_device(device)


def test_conv3d():
    device = ttnn.CreateDevice(0, l1_small_size=8192)
    # Define input parameters
    batch_size, in_channels, depth, height, width = 1, 3, 8, 8, 8
    out_channels = 16
    kernel_size = (3, 3, 3)

    # Create input tensor [N, C, D, H, W]
    torch_input = torch.randn(batch_size, in_channels, depth, height, width, dtype=torch.bfloat16)

    # Create weight and bias tensors
    torch_weight = torch.randn(out_channels, in_channels, *kernel_size, dtype=torch.bfloat16)
    torch_bias = torch.randn(out_channels, dtype=torch.bfloat16)

    # Convert to ttnn tensors
    tt_input = ttnn.from_torch(torch_input, device=device)
    tt_weight = ttnn.from_torch(torch_weight, device=device)
    tt_bias = ttnn.from_torch(torch_bias, device=device)

    # Configure conv3d
    conv_config = ttnn.Conv3dConfig(
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
    )

    # Perform 3D convolution
    output = ttnn.experimental.conv3d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        bias_tensor=tt_bias,
        kernel_size=kernel_size,
        stride=(1, 1, 1),
        padding=(1, 1, 1),
        conv_config=conv_config,
    )
    logger.info(f"Conv3d output: {output}")
    ttnn.close_device(device)


def test_prepare_conv_weights():
    device = ttnn.CreateDevice(0, l1_small_size=8192)
    # Create a simple weight tensor
    out_channels, in_channels, kernel_h, kernel_w = 16, 3, 3, 3
    torch_weight = torch.randn(out_channels, in_channels, kernel_h, kernel_w, dtype=torch.bfloat16)
    tt_weight = ttnn.from_torch(torch_weight)

    # Create a dummy input tensor to provide memory config
    torch_input = torch.randn(1, 32, 32, in_channels, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(torch_input, device=device)

    # Prepare conv weights
    prepared_weight = ttnn.prepare_conv_weights(
        weight_tensor=tt_weight,
        input_memory_config=tt_input.memory_config(),
        input_dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        input_layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    logger.info(f"Prepared conv weights: {prepared_weight}")
    ttnn.close_device(device)


def test_prepare_conv_bias():
    device = ttnn.CreateDevice(0, l1_small_size=8192)
    # Create a simple bias tensor
    out_channels = 16
    torch_bias = torch.randn(1, 1, 1, out_channels, dtype=torch.bfloat16)
    tt_bias = ttnn.from_torch(torch_bias)

    # Create a dummy input tensor to provide memory config
    torch_input = torch.randn(1, 32, 32, 3, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(torch_input, device=device)

    # Prepare conv bias
    prepared_bias = ttnn.prepare_conv_bias(
        bias_tensor=tt_bias,
        input_memory_config=tt_input.memory_config(),
        input_dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        input_layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    logger.info(f"Prepared conv bias: {prepared_bias}")
    ttnn.close_device(device)


def test_prepare_conv_transpose2d_weights():
    device = ttnn.CreateDevice(0, l1_small_size=8192)
    # Create a simple weight tensor (note: shape is [C_in, C_out, K_h, K_w] for transpose)
    in_channels, out_channels, kernel_h, kernel_w = 32, 16, 2, 2
    torch_weight = torch.randn(in_channels, out_channels, kernel_h, kernel_w, dtype=torch.bfloat16)
    tt_weight = ttnn.from_torch(torch_weight)

    # Create a dummy input tensor
    torch_input = torch.randn(1, 16, 16, in_channels, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(torch_input, device=device)

    # Prepare conv_transpose2d weights
    prepared_weight = ttnn.prepare_conv_transpose2d_weights(
        weight_tensor=tt_weight,
        input_memory_config=tt_input.memory_config(),
        input_dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        input_layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    logger.info(f"Prepared conv_transpose2d weights: {prepared_weight}")
    ttnn.close_device(device)


def test_prepare_conv_transpose2d_bias():
    device = ttnn.CreateDevice(0, l1_small_size=8192)
    # Create a simple bias tensor
    out_channels = 16
    torch_bias = torch.randn(1, 1, 1, out_channels, dtype=torch.bfloat16)
    tt_bias = ttnn.from_torch(torch_bias)

    # Create a dummy input tensor
    torch_input = torch.randn(1, 16, 16, 32, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(torch_input, device=device)

    # Prepare conv_transpose2d bias
    prepared_bias = ttnn.prepare_conv_transpose2d_bias(
        bias_tensor=tt_bias,
        input_memory_config=tt_input.memory_config(),
        input_dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        input_layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    logger.info(f"Prepared conv_transpose2d bias: {prepared_bias}")
    ttnn.close_device(device)
