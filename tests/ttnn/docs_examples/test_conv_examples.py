# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_conv1d(device):
    # Input parameters
    batch_size, in_channels, length = 1, 32, 80
    out_channels, kernel_size = 16, 3

    # Create input tensor [N, L, C]
    torch_input = torch.randn(batch_size, length, in_channels, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(torch_input, device=device)

    # Create weight tensor [C_out, C_in, K]
    torch_weight = torch.randn(out_channels, in_channels, kernel_size, dtype=torch.bfloat16)
    tt_weight = ttnn.from_torch(torch_weight)

    # Perform 1D convolution
    output = ttnn.conv1d(
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
    output = ttnn.from_device(output)
    print(f"Conv1d output shape: {output.shape}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_conv2d(device):
    # Input parameters
    batch_size, in_channels, height, width = 1, 32, 32, 32
    out_channels, kernel_h, kernel_w = 16, 3, 3

    # Create input tensor [N, H, W, C]
    torch_input = torch.randn(batch_size, height, width, in_channels, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(torch_input, device=device, layout=ttnn.TILE_LAYOUT)

    # Create weight and bias tensors
    torch_weight = torch.randn(out_channels, in_channels, kernel_h, kernel_w, dtype=torch.bfloat16)
    tt_weight = ttnn.from_torch(torch_weight)

    torch_bias = torch.randn(1, 1, 1, out_channels, dtype=torch.bfloat16)
    tt_bias = ttnn.from_torch(torch_bias)

    # Perform 2D convolution
    output = ttnn.conv2d(
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
    )
    output = ttnn.from_device(output)
    print(f"Conv2d output shape: {output.shape}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_conv_transpose2d(device):
    # Input parameters
    batch_size, in_channels, height, width = 1, 32, 16, 16
    out_channels, kernel_h, kernel_w = 16, 2, 2

    # Create input tensor [N, H, W, C]
    torch_input = torch.randn(batch_size, height, width, in_channels, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(torch_input, device=device, layout=ttnn.TILE_LAYOUT)

    # Create weight [C_in, C_out, K_h, K_w] and bias tensors
    torch_weight = torch.randn(in_channels, out_channels, kernel_h, kernel_w, dtype=torch.bfloat16)
    tt_weight = ttnn.from_torch(torch_weight)

    torch_bias = torch.randn(1, 1, 1, out_channels, dtype=torch.bfloat16)
    tt_bias = ttnn.from_torch(torch_bias)

    # Perform transposed convolution
    output = ttnn.conv_transpose2d(
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
    )
    output = ttnn.from_device(output)
    print(f"Conv_transpose2d output shape: {output.shape}")


def test_conv3d(device):
    # Input parameters
    batch_size, in_channels, depth, height, width = 1, 32, 8, 8, 8
    out_channels = 32
    kernel_size = (3, 3, 3)

    # Create input tensor [N, C, D, H, W]
    torch_input = torch.randn(batch_size, in_channels, depth, height, width, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(torch_input, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Note: Conv3d requires weights in patchified format [patch_size, out_channels]
    # where patch_size depends on input dimensions and kernel size
    # This requires specialized preprocessing not shown in this minimal example
    patch_size = 216  # Example value - actual calculation depends on config
    torch_weight = torch.randn(patch_size, out_channels, dtype=torch.bfloat16)
    torch_bias = torch.randn(1, out_channels, dtype=torch.bfloat16)
    tt_weight = ttnn.from_torch(torch_weight, device=device, layout=ttnn.TILE_LAYOUT)
    tt_bias = ttnn.from_torch(torch_bias, device=device, layout=ttnn.TILE_LAYOUT)

    # Configure and perform 3D convolution
    config = ttnn.Conv3dConfig(
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        output_channels=out_channels,
        kernel_size=list(kernel_size),
        stride=[1, 1, 1],
        padding=[0, 1, 1],
    )
    output = ttnn.experimental.conv3d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        bias_tensor=tt_bias,
        config=config,
    )
    output = ttnn.from_device(output)
    print(f"Conv3d output shape: {output.shape}")


def test_prepare_conv_weights(device):
    # Create weight tensor
    batch_size, in_channels, height, width = 1, 32, 32, 32
    out_channels, kernel_h, kernel_w = 16, 3, 3
    torch_weight = torch.randn(out_channels, in_channels, kernel_h, kernel_w, dtype=torch.bfloat16)
    tt_weight = ttnn.from_torch(torch_weight)

    # Create input tensor for memory config
    torch_input = torch.randn(batch_size, height, width, in_channels, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(torch_input, device=device)

    # Prepare conv weights
    prepared_weight = ttnn.prepare_conv_weights(
        weight_tensor=tt_weight,
        input_memory_config=tt_input.memory_config(),
        input_layout=ttnn.TILE_LAYOUT,
        weights_format="OIHW",
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        input_height=height,
        input_width=width,
        kernel_size=[kernel_h, kernel_w],
        stride=[1, 1],
        padding=[1, 1],
        dilation=[1, 1],
        has_bias=False,
        groups=1,
        device=device,
        input_dtype=ttnn.bfloat16,
    )
    prepared_weight = ttnn.from_device(prepared_weight)
    print(f"Prepared conv weights shape: {prepared_weight.shape}")


def test_prepare_conv_bias(device):
    # Create bias tensor
    batch_size, in_channels, height, width = 1, 32, 32, 32
    out_channels, kernel_h, kernel_w = 16, 3, 3
    torch_bias = torch.randn(1, 1, 1, out_channels, dtype=torch.bfloat16)
    tt_bias = ttnn.from_torch(torch_bias)

    # Create input tensor for memory config
    torch_input = torch.randn(batch_size, height, width, in_channels, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(torch_input, device=device)

    # Prepare conv bias
    conv_config = ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat16)
    prepared_bias = ttnn.prepare_conv_bias(
        bias_tensor=tt_bias,
        input_memory_config=tt_input.memory_config(),
        input_layout=ttnn.TILE_LAYOUT,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        input_height=height,
        input_width=width,
        kernel_size=[kernel_h, kernel_w],
        stride=[1, 1],
        padding=[1, 1],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.bfloat16,
        conv_config=conv_config,
    )
    prepared_bias = ttnn.from_device(prepared_bias)
    print(f"Prepared conv bias shape: {prepared_bias.shape}")


def test_prepare_conv_transpose2d_weights(device):
    # Create weight tensor [C_in, C_out, K_h, K_w]
    batch_size, in_channels, height, width = 1, 32, 16, 16
    out_channels, kernel_h, kernel_w = 16, 2, 2
    torch_weight = torch.randn(in_channels, out_channels, kernel_h, kernel_w, dtype=torch.bfloat16)
    tt_weight = ttnn.from_torch(torch_weight)

    # Create input tensor for memory config
    torch_input = torch.randn(batch_size, height, width, in_channels, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(torch_input, device=device)

    # Prepare conv_transpose2d weights
    prepared_weight = ttnn.prepare_conv_transpose2d_weights(
        weight_tensor=tt_weight,
        input_memory_config=tt_input.memory_config(),
        input_layout=ttnn.TILE_LAYOUT,
        weights_format="IOHW",
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        input_height=height,
        input_width=width,
        kernel_size=[kernel_h, kernel_w],
        stride=[2, 2],
        padding=[0, 0],
        dilation=[1, 1],
        has_bias=False,
        groups=1,
        device=device,
        input_dtype=ttnn.bfloat16,
    )
    prepared_weight = ttnn.from_device(prepared_weight)
    print(f"Prepared conv_transpose2d weights shape: {prepared_weight.shape}")


def test_prepare_conv_transpose2d_bias(device):
    # Create bias tensor
    batch_size, in_channels, height, width = 1, 32, 16, 16
    out_channels, kernel_h, kernel_w = 16, 2, 2
    torch_bias = torch.randn(1, 1, 1, out_channels, dtype=torch.bfloat16)
    tt_bias = ttnn.from_torch(torch_bias)

    # Create input tensor for memory config
    torch_input = torch.randn(batch_size, height, width, in_channels, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(torch_input, device=device)

    # Prepare conv_transpose2d bias
    conv_config = ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat16)
    prepared_bias = ttnn.prepare_conv_transpose2d_bias(
        bias_tensor=tt_bias,
        input_memory_config=tt_input.memory_config(),
        input_layout=ttnn.TILE_LAYOUT,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        input_height=height,
        input_width=width,
        kernel_size=[kernel_h, kernel_w],
        stride=[2, 2],
        padding=[0, 0],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.bfloat16,
        conv_config=conv_config,
    )
    prepared_bias = ttnn.from_device(prepared_bias)
    print(f"Prepared conv_transpose2d bias shape: {prepared_bias.shape}")
