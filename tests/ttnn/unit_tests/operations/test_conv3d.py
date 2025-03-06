# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import torch
import pytest
import ttnn
import torch.nn as nn
from tests.ttnn.utils_for_testing import check_with_pcc


def _out_size(in_size, pad, k):
    return in_size + 2 * pad - (k - 1)


ALIGNMENT = 32  # Valid L1 alignment for Wormhole and Blackhole


def prepare_input_tensor(input_tensor, C, device, alignment=ALIGNMENT):
    """Prepare input tensor for TTNN by permuting and padding."""
    tt_input = input_tensor.permute(0, 2, 3, 4, 1)
    ALIGN_PAD = alignment - C % alignment
    if C % alignment != 0:
        tt_input = torch.nn.functional.pad(tt_input, (0, ALIGN_PAD))
    return ttnn.from_torch(tt_input, device=device, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.ROW_MAJOR_LAYOUT)


def prepare_weights(conv3d_module, C, out_channels, device, C_in_block=0, alignment=ALIGNMENT):
    """Prepare weights and bias for TTNN."""
    w = conv3d_module.weight.data  # out_chan, C, kD, kH, kW
    w = w.permute(2, 3, 4, 1, 0)  # kD, kH, kW, C, out_chan
    ALIGN_PAD = alignment - C % alignment
    if C % alignment != 0:
        w = torch.nn.functional.pad(w, (0, 0, 0, ALIGN_PAD))

    # Reshape weights so that num_C_in_blocks is the first dimension
    kD, kH, kW, C_in_aligned, out_channels = w.shape
    C_in_block = C_in_aligned if C_in_block == 0 else C_in_block
    num_C_in_blocks = C_in_aligned // C_in_block
    assert num_C_in_blocks * C_in_block == C_in_aligned
    w = w.reshape(kD, kH, kW, num_C_in_blocks, C_in_block, out_channels)
    w = w.permute(3, 0, 1, 2, 4, 5)
    w = w.reshape(-1, out_channels)

    tt_weight = ttnn.from_torch(w, device=device, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT, pad_value=0)
    tt_bias = ttnn.from_torch(
        conv3d_module.bias.data.reshape(1, -1),
        device=device,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
        pad_value=0,
    )
    return tt_weight, tt_bias


def reshape_output(tt_output, N, D_out, H_out, W_out, out_channels, device):
    """Reshape and permute TTNN output to match PyTorch format."""
    tt_output = ttnn.to_torch(tt_output, device=device, dtype=torch.float32)
    tt_output = tt_output.reshape(N, D_out, H_out, W_out, out_channels)
    return tt_output.permute(0, 4, 1, 2, 3)


def create_conv3d_config(
    out_channels,
    kernel_size,
    stride,
    padding,
    padding_mode,
    T_out_block=1,
    W_out_block=1,
    H_out_block=1,
    C_out_block=0,
    C_in_block=0,
    compute_with_storage_grid_size=(1, 1),
):
    """Create Conv3d configuration."""
    return ttnn.Conv3dConfig(
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        T_out_block=T_out_block,
        W_out_block=W_out_block,
        H_out_block=H_out_block,
        C_out_block=C_out_block,
        C_in_block=C_in_block,
        output_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        padding_mode=padding_mode,
        groups=1,
        compute_with_storage_grid_size=compute_with_storage_grid_size,
    )


def setup_conv3d_test(input_shape, out_channels, kernel_size, stride, padding, padding_mode, device):
    """Common setup for Conv3D tests, preparing inputs and ground truth."""
    torch.manual_seed(42)

    # Define input dimensions
    N, C, D, H, W = input_shape
    D_out = _out_size(D, padding[0], kernel_size[0])
    H_out = _out_size(H, padding[1], kernel_size[1])
    W_out = _out_size(W, padding[2], kernel_size[2])

    # Create input tensor and PyTorch Conv3d module
    input_tensor = torch.randn(N, C, D, H, W, dtype=torch.float32)

    conv3d_module = nn.Conv3d(
        C,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=(1, 1, 1),
        bias=True,
        padding_mode=padding_mode,
    )

    gt_output = conv3d_module(input_tensor)

    # Prepare input for TTNN
    tt_input = prepare_input_tensor(input_tensor, C, device)

    kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    return tt_input, conv3d_module, gt_output, kernel_config, (N, D_out, H_out, W_out)


def run_conv3d_test(device, input_shape, out_channels, kernel_size, stride, padding, padding_mode, grid_size=(1, 1)):
    tt_input, conv3d_module, gt_output, kernel_config, output_dims = setup_conv3d_test(
        input_shape, out_channels, kernel_size, stride, padding, padding_mode, device
    )
    N, D_out, H_out, W_out = output_dims
    C = input_shape[1]

    # Prepare weights and bias for TTNN
    tt_weight, tt_bias = prepare_weights(conv3d_module, C, out_channels, device, C_in_block=0)

    # Create config and run TTNN conv3d
    config = create_conv3d_config(
        out_channels, kernel_size, stride, padding, padding_mode, compute_with_storage_grid_size=grid_size
    )

    tt_output = ttnn.experimental.conv3d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        bias_tensor=tt_bias,
        config=config,
        compute_kernel_config=kernel_config,
    )

    # Reshape output and verify results
    tt_output = reshape_output(tt_output, N, D_out, H_out, W_out, out_channels, device)

    print(f"gt output shape = {gt_output.shape}")
    print(f"tt output shape = {tt_output.shape}")
    assert tt_output.shape == gt_output.shape

    pcc_passed, pcc_message = check_with_pcc(gt_output, tt_output, pcc=0.999)
    logger.info(f"Compare conv3d torch vs ttnn: {pcc_message}")
    assert pcc_passed, pcc_message


@pytest.mark.parametrize("B", [1])
@pytest.mark.parametrize("C_in", [12, 64])
@pytest.mark.parametrize("C_out", [64])
@pytest.mark.parametrize("T", [8, 11])
@pytest.mark.parametrize("H", [10, 13])
@pytest.mark.parametrize("W", [9, 12])
@pytest.mark.parametrize("kernel_size", [(3, 3, 3), (1, 1, 1)], ids=["kernel_333", "kernel_111"])
@pytest.mark.parametrize("stride", [(1, 1, 1)], ids=["stride_111"])
@pytest.mark.parametrize("padding", [(0, 1, 1)], ids=["padding_011"])
@pytest.mark.parametrize("padding_mode", ["zeros", "replicate"])
def test_conv3d_sweep_shapes(device, B, C_in, C_out, T, H, W, kernel_size, stride, padding, padding_mode):
    input_shape = (B, C_in, T, H, W)
    out_channels = C_out
    kernel_size = kernel_size
    stride = stride
    padding = padding
    padding_mode = padding_mode
    grid_size = device.compute_with_storage_grid_size()
    run_conv3d_test(device, input_shape, out_channels, kernel_size, stride, padding, padding_mode, grid_size=grid_size)


@pytest.mark.parametrize(
    "input_shape, out_channels, kernel_size, stride, padding, padding_mode",
    [
        [(1, 64, 16, 16, 16), 64, (3, 3, 3), (1, 1, 1), (0, 1, 1), "replicate"],
    ],
)
def test_conv3d_cache_address(
    device, input_shape, out_channels, kernel_size, stride, padding, padding_mode, use_program_cache
):
    # Test that program cache updates the addresses of the inputs
    grid_size = device.compute_with_storage_grid_size()
    dummy = []
    for _ in range(3):
        dummy.append(ttnn.from_torch(torch.randn(input_shape), device=device, layout=ttnn.TILE_LAYOUT))
        run_conv3d_test(
            device, input_shape, out_channels, kernel_size, stride, padding, padding_mode, grid_size=grid_size
        )


@pytest.mark.parametrize(
    "input_shape, out_channels, kernel_size, stride, padding, padding_mode",
    [
        [(1, 64, 16, 16, 16), 64, (3, 3, 3), (1, 1, 1), (0, 1, 1), "replicate"],
    ],
)
def test_conv3d_cache_hash(
    device, input_shape, out_channels, kernel_size, stride, padding, padding_mode, use_program_cache
):
    # Test that program cache does not re-use the same program for different inputs
    grid_size = device.compute_with_storage_grid_size()
    dummy = []
    for _ in range(3):
        for i in range(2):
            new_shape = (input_shape[0], input_shape[1] * (i + 1), input_shape[2], input_shape[3], input_shape[4])
            dummy.append(ttnn.from_torch(torch.randn(new_shape), device=device, layout=ttnn.TILE_LAYOUT))
            run_conv3d_test(
                device, new_shape, out_channels, kernel_size, stride, padding, padding_mode, grid_size=grid_size
            )

    assert device.num_program_cache_entries() == 2
