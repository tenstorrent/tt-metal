# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def prepare_grid_sample_grid_pytorch(grid, input_shape):
    """
    PyTorch implementation of prepare_grid_sample_grid using vectorized operations.

    Args:
        grid: Tensor of shape (N, H_out, W_out, 2) with normalized coordinates in [-1, 1]
        input_shape: List [N, H_in, W_in, C] in NHWC format

    Returns:
        Tensor of shape (N, H_out, W_out, 6) with precomputed grid data
    """
    batch_size, input_h, input_w, channels = input_shape
    grid_n, grid_h, grid_w, _ = grid.shape

    # Extract x and y coordinates
    x_coord = grid[..., 0]  # Shape: (N, H_out, W_out)
    y_coord = grid[..., 1]  # Shape: (N, H_out, W_out)

    # Scale factors for coordinate transformation (align_corners=False)
    height_scale = float(input_h) * 0.5
    height_offset = height_scale - 0.5
    width_scale = float(input_w) * 0.5
    width_offset = width_scale - 0.5

    # Transform to image coordinates
    h_coord_image = y_coord * height_scale + height_offset
    w_coord_image = x_coord * width_scale + width_offset

    # Get corner pixel coordinates (floor operation)
    h0 = torch.floor(h_coord_image).to(torch.int32)
    w0 = torch.floor(w_coord_image).to(torch.int32)
    h1 = h0 + 1
    w1 = w0 + 1

    # Boundary checks
    h0_valid = (h0 >= 0) & (h0 < input_h)
    h1_valid = (h1 >= 0) & (h1 < input_h)
    w0_valid = (w0 >= 0) & (w0 < input_w)
    w1_valid = (w1 >= 0) & (w1 < input_w)

    # Calculate interpolation weights
    h_frac = h_coord_image - h0.float()
    w_frac = w_coord_image - w0.float()
    h_frac_inv = 1.0 - h_frac
    w_frac_inv = 1.0 - w_frac

    # Compute bilinear weights with boundary conditions
    weight_nw = torch.where(h0_valid & w0_valid, h_frac_inv * w_frac_inv, torch.tensor(0.0))
    weight_ne = torch.where(h0_valid & w1_valid, h_frac_inv * w_frac, torch.tensor(0.0))
    weight_sw = torch.where(h1_valid & w0_valid, h_frac * w_frac_inv, torch.tensor(0.0))
    weight_se = torch.where(h1_valid & w1_valid, h_frac * w_frac, torch.tensor(0.0))

    # Clamp coordinates to 16-bit range
    h0_clamped = torch.clamp(h0, -32768, 32767).to(torch.int16)
    w0_clamped = torch.clamp(w0, -32768, 32767).to(torch.int16)

    # Convert int16 bit representation to bfloat16 (reinterpret bits, not values)
    # First convert int16 to uint16, then reinterpret as bfloat16
    h0_bits = h0_clamped.view(torch.uint16)
    w0_bits = w0_clamped.view(torch.uint16)

    # Create bfloat16 tensors with the same bit pattern
    # We need to create a tensor where the bfloat16 bits match the uint16 bits
    h0_as_bf16 = torch.zeros_like(h0_bits, dtype=torch.bfloat16)
    w0_as_bf16 = torch.zeros_like(w0_bits, dtype=torch.bfloat16)

    # Copy the bit pattern by viewing as bytes and reconstructing
    h0_as_bf16.view(torch.uint16).copy_(h0_bits)
    w0_as_bf16.view(torch.uint16).copy_(w0_bits)

    # Stack results into output tensor
    output = torch.stack(
        [
            h0_as_bf16,  # North-west height coordinate (as bit pattern)
            w0_as_bf16,  # North-west width coordinate (as bit pattern)
            weight_nw,  # Weight for north-west pixel
            weight_ne,  # Weight for north-east pixel
            weight_sw,  # Weight for south-west pixel
            weight_se,  # Weight for south-east pixel
        ],
        dim=-1,
    )

    # Return as float32, conversion to bfloat16 done later
    return output.float()


@pytest.mark.parametrize(
    "input_shape, grid_shape",
    [
        ((2, 32, 24, 16), (2, 12, 8, 2)),
        ((1, 256, 48, 160), (1, 7, 25281, 2)),
        ((4, 64, 20, 20), (4, 10, 10, 2)),
    ],
)
def test_prepare_grid_sample_grid_comparison(input_shape, grid_shape):
    """Test that PyTorch implementation matches TTNN prepare_grid_sample_grid"""
    torch.manual_seed(42)

    # Create random grid tensor with normalized coordinates [-1, 1]
    torch_grid = torch.rand(grid_shape, dtype=torch.float32) * 2.0 - 1.0

    # PyTorch implementation
    pytorch_result = prepare_grid_sample_grid_pytorch(torch_grid, list(input_shape))

    # TTNN implementation
    ttnn_grid = ttnn.from_torch(torch_grid, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32)
    ttnn_result_tensor = ttnn.prepare_grid_sample_grid(
        ttnn_grid, list(input_shape), padding_mode="zeros", output_dtype=ttnn.bfloat16
    )
    ttnn_result = ttnn.to_torch(ttnn_result_tensor)

    # Check shapes match
    assert pytorch_result.shape == ttnn_result.shape, f"Shape mismatch: {pytorch_result.shape} vs {ttnn_result.shape}"

    # Compare coordinate bit patterns directly (first 2 elements)
    coords_pytorch = pytorch_result[..., :2]
    coords_ttnn = ttnn_result[..., :2]
    pytorch_bits = coords_pytorch.view(torch.uint16)
    ttnn_bits = coords_ttnn.view(torch.uint16)

    coords_bits_equal = torch.equal(pytorch_bits, ttnn_bits)

    # Last 4 indices should pass PCC test (weights)
    weights_pytorch = pytorch_result[..., 2:]
    weights_ttnn = ttnn_result[..., 2:]

    pcc_passed, pcc_message = assert_with_pcc(weights_pytorch, weights_ttnn, pcc=0.99)

    # Assertions
    assert coords_bits_equal, "Coordinate bit patterns must be exactly equal"
    assert pcc_passed, f"Weight values must pass PCC test: {pcc_message}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
@pytest.mark.parametrize(
    "input_shape_nchw, base_grid_shape, channel_extent_factor",
    [
        ((1, 256, 48, 160), (1, 25281, 7, 2), 7),
    ],
)
def test_pytorch_precomputed_grid_channel_extending(device, input_shape_nchw, base_grid_shape, channel_extent_factor):
    """Test PyTorch precomputed grid with channel extending functionality"""
    torch.manual_seed(0)

    batch_size, channels, height, width = input_shape_nchw
    input_shape_nhwc = [batch_size, height, width, channels]
    grid_n, grid_h, grid_w, grid_coords = base_grid_shape

    # Step 1: Get the normal pytorch grid in fp32
    torch_grid = torch.rand(base_grid_shape, dtype=torch.float32) * 2.0 - 1.0

    # Step 2: Preprocess it in python
    pytorch_precomputed = prepare_grid_sample_grid_pytorch(torch_grid, input_shape_nhwc)

    # Step 3: Reshape it so that the grid, instead of being 1, H_out, W_out, 6, make it into 1, H_out, W_out/channel_extent_factor, 6*channel_extent_factor
    new_grid_w = grid_w // channel_extent_factor
    final_last_dim = 6 * channel_extent_factor
    pytorch_reshaped = pytorch_precomputed.view(batch_size, grid_h, new_grid_w, final_last_dim)

    # Step 4: Convert that grid to bfloat16
    pytorch_reshaped_bf16 = pytorch_reshaped.to(torch.bfloat16)

    # Step 5: Send it to ttnn on device
    ttnn_grid_device = ttnn.from_torch(pytorch_reshaped_bf16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # Create input tensor
    torch_input_nchw = torch.randn(input_shape_nchw, dtype=torch.float32)
    torch_input_nhwc = torch_input_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    # Step 6: Run ttnn grid sample
    ttnn_output = ttnn.grid_sample(ttnn_input, ttnn_grid_device, use_precomputed_grid=True)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    # Expected output using PyTorch grid_sample
    torch_output_nchw = F.grid_sample(
        torch_input_nchw, torch_grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    torch_output_nhwc = torch_output_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    # Reshape expected output for channel extending
    torch_expected_reshaped = (
        torch_output_nhwc.view(batch_size, grid_h, new_grid_w, channel_extent_factor, channels)
        .contiguous()
        .view(batch_size, grid_h, new_grid_w, channels * channel_extent_factor)
    )

    # Step 7: Compare to torch with pcc
    pcc_passed, pcc_message = assert_with_pcc(torch_expected_reshaped, ttnn_output_torch, pcc=0.99)
    assert pcc_passed, f"PCC test failed: {pcc_message}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
@pytest.mark.parametrize("use_precomputed_grid", [False, True])
@pytest.mark.parametrize(
    "input_shape, grid_shape",
    [
        # ((1, 256, 12, 40), (1, 7, 25281, 2)),
        # ((1, 256, 24, 80), (1, 7, 25281, 2)),
        ((1, 256, 48, 160), (1, 7, 25281, 2)),
        # ((16, 32, 100, 100), (16, 10000, 4, 2)),
        # ((48, 32, 12, 20), (48, 3567, 8, 2)),
        # ((8, 32, 100, 100), (8, 300, 4, 2)),
        # ((8, 32, 100, 100), (8, 2000, 4, 2)),
        # ((16, 32, 50, 50), (16, 10000, 1, 2)),
        # ((48, 32, 80, 45), (48, 4832, 1, 2)),
        # ((48, 32, 40, 23), (48, 4832, 1, 2)),
        # ((48, 32, 20, 12), (48, 4832, 1, 2)),
        # ((48, 32, 10, 6), (48, 4832, 1, 2)),
        # ((8, 32, 50, 50), (8, 3604, 1, 2)),
    ],
)
def test_grid_sample_near_uniform_grid(device, input_shape, grid_shape, use_precomputed_grid):
    torch.manual_seed(0)

    batch_size, grid_h, grid_w, _ = grid_shape

    batch_size, channels, height, width = input_shape

    input_shape_nhwc = [batch_size, height, width, channels]

    # PyTorch CPU grid_sample has bad behaviour for bfloat16 inputs
    torch_input_nchw = torch.randn(input_shape, dtype=torch.float32)
    torch_input_nhwc = torch_input_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    # Generates a uniform grid using torch affine grid
    theta = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    theta_batched = theta.unsqueeze(0).expand(batch_size, -1, -1)
    shape = (batch_size, 1, grid_h, grid_w)
    torch_grid = F.affine_grid(theta_batched, shape, align_corners=False)

    # Add small noise to the grid
    torch_grid += torch.randn(grid_shape) * 0.05

    torch_output_nchw = F.grid_sample(
        torch_input_nchw, torch_grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )

    torch_grid_bf16 = torch_grid.to(torch.bfloat16)

    torch_output_nhwc = torch_output_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    if use_precomputed_grid:
        ttnn_grid = ttnn.from_torch(torch_grid_bf16, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32)
        prepared_grid = ttnn.prepare_grid_sample_grid(
            ttnn_grid, input_shape_nhwc, padding_mode="zeros", output_dtype=ttnn.bfloat16
        )
        prepared_grid = ttnn.to_device(prepared_grid, device)
        ttnn_output = ttnn.grid_sample(ttnn_input, prepared_grid, use_precomputed_grid=True)
    else:
        # Use regular grid
        ttnn_grid = ttnn.from_torch(torch_grid_bf16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.bfloat16)
        ttnn_output = ttnn.grid_sample(ttnn_input, ttnn_grid)

    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    pcc_passed, pcc_message = assert_with_pcc(torch_output_nhwc, ttnn_output_torch, pcc=0.99)
    logger.info(pcc_message)


@pytest.mark.parametrize("use_precomputed_grid", [True])  # , False])
@pytest.mark.parametrize(
    "input_shape, grid_shape, channel_extent_factor",
    [
        ((1, 256, 48, 160), (1, 25281, 7, 2), 7),
        # ((1, 32, 16, 16), (1, 8, 8, 2), 1),
        # ((2, 64, 24, 24), (2, 12, 12, 2), 2),
        # ((2, 64, 24, 24), (2, 12, 12, 2), 3),
        # ((1, 128, 32, 32), (1, 16, 16, 2), 2),
        # ((1, 128, 32, 32), (1, 16, 16, 2), 4),
        # ((4, 64, 20, 20), (4, 10, 10, 2), 2),
        # ((4, 64, 20, 20), (4, 10, 10, 2), 5),
        # ((1, 96, 24, 24), (1, 12, 12, 2), 3),
        # ((1, 96, 24, 24), (1, 12, 12, 2), 6),
    ],
)
def test_grid_sample_channel_extending(device, input_shape, grid_shape, channel_extent_factor, use_precomputed_grid):
    """Test grid sample with channel extending functionality (multiple coordinate sets)"""
    torch.manual_seed(523)

    batch_size, channels, height, width = input_shape
    grid_n, grid_h, grid_w, grid_coords = grid_shape

    # Validate that channel_extent_factor is a divisor of width
    assert (
        grid_w % channel_extent_factor == 0
    ), f"channel_extent_factor {channel_extent_factor} must divide grid width {grid_w}"

    # Create input tensor (NCHW -> NHWC)
    torch_input_nchw = torch.randn(input_shape, dtype=torch.float32)
    torch_input_nhwc = torch_input_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    # Generate random grid tensor
    grid_tensor = torch.rand(batch_size, grid_h, grid_w, 2, dtype=torch.float32) * 2.0 - 1.0

    # Create host ttnn tensor
    ttnn_grid_host = ttnn.from_torch(grid_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32)

    if use_precomputed_grid:
        input_shape_nhwc = [batch_size, height, width, channels]
        ttnn_grid_precomputed = ttnn.prepare_grid_sample_grid(
            ttnn_grid_host, input_shape_nhwc, padding_mode="zeros", output_dtype=ttnn.bfloat16
        )
        new_grid_w = grid_w // channel_extent_factor
        final_last_dim = 6 * channel_extent_factor
        ttnn_grid_reshaped = ttnn.reshape(ttnn_grid_precomputed, (batch_size, grid_h, new_grid_w, final_last_dim))
        ttnn_grid_device = ttnn.to_device(ttnn_grid_reshaped, device)
    else:
        new_grid_w = grid_w // channel_extent_factor
        new_last_dim = 2 * channel_extent_factor
        ttnn_grid_host = ttnn.from_torch(grid_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
        ttnn_grid_reshaped = ttnn.reshape(ttnn_grid_host, (batch_size, grid_h, new_grid_w, new_last_dim))
        ttnn_grid_device = ttnn.to_device(ttnn_grid_reshaped, device)

    ttnn_input = ttnn.from_torch(
        torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    ttnn_output = ttnn.grid_sample(ttnn_input, ttnn_grid_device, use_precomputed_grid=use_precomputed_grid)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    torch_output_nchw = F.grid_sample(
        torch_input_nchw, grid_tensor, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    torch_output_nhwc_ = torch_output_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    torch_expected_nhwc = (
        torch_output_nhwc_.view(batch_size, grid_h, new_grid_w, channel_extent_factor, channels)
        .contiguous()
        .view(batch_size, grid_h, new_grid_w, channels * channel_extent_factor)
    )

    # Check output shape
    expected_shape = (batch_size, grid_h, new_grid_w, channels * channel_extent_factor)
    assert ttnn_output_torch.shape == expected_shape, f"Expected {expected_shape}, got {ttnn_output_torch.shape}"
    pcc_passed, pcc_message = assert_with_pcc(torch_expected_nhwc, ttnn_output_torch, pcc=0.99)
    logger.info(
        f"Channel extending test (extent_factor={channel_extent_factor}, precomputed={use_precomputed_grid}): {pcc_message}"
    )
