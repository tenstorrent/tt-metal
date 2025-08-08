# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
@pytest.mark.parametrize(
    "input_shape, grid_shape",
    [
        ((1, 256, 12, 40), (1, 7, 25281, 2)),
        ((1, 256, 24, 80), (1, 7, 25281, 2)),
        ((1, 256, 48, 160), (1, 7, 25281, 2)),
        ((16, 32, 100, 100), (16, 10000, 4, 2)),
        ((48, 32, 12, 20), (48, 3567, 8, 2)),
        ((8, 32, 100, 100), (8, 300, 4, 2)),
        ((8, 32, 100, 100), (8, 2000, 4, 2)),
        ((16, 32, 50, 50), (16, 10000, 1, 2)),
        ((48, 32, 80, 45), (48, 4832, 1, 2)),
        ((48, 32, 40, 23), (48, 4832, 1, 2)),
        ((48, 32, 20, 12), (48, 4832, 1, 2)),
        ((48, 32, 10, 6), (48, 4832, 1, 2)),
        ((8, 32, 50, 50), (8, 3604, 1, 2)),
        # Small test cases for correctness verification
        # ((1, 4, 4, 8), (1, 2, 2, 2)),      # Small input, small grid
        # ((1, 8, 8, 16), (1, 4, 4, 2)),     # Medium input, small grid
        # ((1, 16, 16, 32), (1, 8, 8, 2)),   # Medium input, medium grid
        # # Different aspect ratios
        # ((1, 8, 16, 12), (1, 4, 8, 2)),    # Rectangular input, rectangular grid
        # # Single pixel grids
        # ((1, 16, 16, 32), (1, 1, 1, 2)),   # Single output pixel
        # ((1, 32, 32, 64), (1, 1, 5, 2)),   # Single row output
        # ((1, 32, 32, 64), (1, 5, 1, 2)),   # Single column output
    ],
)
def test_grid_sample_near_uniform_grid(device, input_shape, grid_shape):
    torch.manual_seed(0)

    batch_size, channels, input_h, input_w = input_shape  # NCHW format
    _, grid_h, grid_w, _ = grid_shape

    torch_input_nchw = torch.randn(input_shape, dtype=torch.float32)

    torch_input_nhwc = torch_input_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    torch_grid = torch.zeros(grid_shape, dtype=torch.bfloat16)

    # Generate varied grid coordinates to test interpolation
    for h in range(grid_h):
        for w in range(grid_w):
            # Create some variation in coordinates to test interpolation properly
            if grid_w > 1:
                x_coord = 2.0 * (w + 0.5) / grid_w - 1.0
                # Add some variation to test interpolation between pixels
                x_coord += 0.1 * torch.randn(1).item()
                x_coord = max(-1.0, min(1.0, x_coord))  # Clamp to valid range
            else:
                x_coord = 0.0

            if grid_h > 1:
                y_coord = 2.0 * (h + 0.5) / grid_h - 1.0
                # Add some variation to test interpolation between pixels
                y_coord += 0.1 * torch.randn(1).item()
                y_coord = max(-1.0, min(1.0, y_coord))  # Clamp to valid range
            else:
                y_coord = 0.0

            torch_grid[:, h, w, 0] = x_coord  # x coordinate
            torch_grid[:, h, w, 1] = y_coord  # y coordinate

    # Convert to NCHW for PyTorch grid_sample (PyTorch has errors if grid is bfloat16)
    torch_grid_float = torch_grid.to(torch.float32)

    # Run PyTorch grid_sample (golden reference) - input is already in NCHW format
    torch_output_nchw = F.grid_sample(
        torch_input_nchw, torch_grid_float, mode="bilinear", padding_mode="zeros", align_corners=False
    )

    # Convert PyTorch output back to NHWC
    torch_output_nhwc = torch_output_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_grid = ttnn.from_torch(torch_grid, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    ttnn_output = ttnn.grid_sample(ttnn_input, ttnn_grid)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    pcc_passed, pcc_message = assert_with_pcc(torch_output_nhwc, ttnn_output_torch, pcc=0.99)
    logger.info(pcc_message)

    print(f"✓ PCC test passed for {input_shape} -> {grid_shape}: {pcc_message}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 16, 16, 32),
        (1, 32, 32, 64),
        (2, 16, 16, 32),  # Batch size > 1
    ],
)
def test_grid_sample_identity_transform(device, input_shape):
    """Test grid_sample with identity transformation (should return original image)"""

    torch.manual_seed(0)

    batch_size, height, width, channels = input_shape
    grid_shape = (batch_size, height, width, 2)

    torch_input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)

    # Create identity grid (maps each output pixel to corresponding input pixel)
    torch_grid = torch.zeros(grid_shape, dtype=torch.bfloat16)
    for h in range(height):
        for w in range(width):
            # Normalize to [-1, 1] range
            x_coord = 2.0 * (w + 0.5) / width - 1.0 if width > 1 else 0.0
            y_coord = 2.0 * (h + 0.5) / height - 1.0 if height > 1 else 0.0
            torch_grid[:, h, w, 0] = x_coord
            torch_grid[:, h, w, 1] = y_coord

    # Convert to NCHW for PyTorch
    torch_input_nchw = torch_input_nhwc.permute(0, 3, 1, 2).to(torch.float32)
    torch_grid_float = torch_grid.to(torch.float32)

    # PyTorch reference (should be very close to original input)
    torch_output_nchw = F.grid_sample(
        torch_input_nchw, torch_grid_float, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    torch_output_nhwc = torch_output_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    # TTNN implementation
    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_grid = ttnn.from_torch(torch_grid, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    ttnn_output = ttnn.grid_sample(ttnn_input, ttnn_grid)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    # High PCC threshold since this should be very close to identity
    pcc_passed, pcc_message = assert_with_pcc(torch_output_nhwc, ttnn_output_torch, pcc=0.99)
    logger.info(pcc_message)

    # Also check that PyTorch identity is close to original (sanity check)
    pcc_identity, identity_message = assert_with_pcc(torch_input_nhwc, torch_output_nhwc, pcc=0.95)
    logger.info(f"Identity check: {identity_message}")

    print(f"✓ Identity transform test for {input_shape}: {pcc_message}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 16, 16, 32),
        (1, 32, 32, 64),
        (2, 16, 16, 32),
    ],
)
@pytest.mark.parametrize(
    "scale_factor",
    [0.5, 2.0, 1.5],  # Downsampling, upsampling, and mixed scaling
)
def test_grid_sample_scaling_patterns(device, input_shape, scale_factor):
    """Test grid_sample with different scaling patterns"""

    torch.manual_seed(0)

    # Calculate output size based on scale factor
    output_n = input_shape[0]
    output_h = int(input_shape[1] * scale_factor)
    output_w = int(input_shape[2] * scale_factor)
    grid_shape = (output_n, output_h, output_w, 2)

    # Create input with smooth gradient pattern
    torch_input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16) * 0.1
    # Add smooth gradient
    for h in range(input_shape[1]):
        for w in range(input_shape[2]):
            torch_input_nhwc[0, h, w, :] += h * 0.1 + w * 0.05

    # Create scaling grid
    torch_grid = torch.zeros(grid_shape, dtype=torch.bfloat16)
    for h in range(output_h):
        for w in range(output_w):
            # Map output coordinates back to input coordinates
            x_coord = 2.0 * w / (output_w - 1) - 1.0 if output_w > 1 else 0.0
            y_coord = 2.0 * h / (output_h - 1) - 1.0 if output_h > 1 else 0.0
            torch_grid[:, h, w, 0] = x_coord
            torch_grid[:, h, w, 1] = y_coord

    # Convert to NCHW for PyTorch
    torch_input_nchw = torch_input_nhwc.permute(0, 3, 1, 2).to(torch.float32)
    torch_grid_float = torch_grid.to(torch.float32)

    # PyTorch reference
    torch_output_nchw = F.grid_sample(
        torch_input_nchw, torch_grid_float, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    torch_output_nhwc = torch_output_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    # TTNN implementation
    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_grid = ttnn.from_torch(torch_grid, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    ttnn_output = ttnn.grid_sample(ttnn_input, ttnn_grid)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    # PCC check for scaling
    pcc_passed, pcc_message = assert_with_pcc(torch_output_nhwc, ttnn_output_torch, pcc=0.90)
    logger.info(pcc_message)

    print(f"✓ Scaling {scale_factor}x test ({input_shape[1:3]} -> {(output_h, output_w)}): {pcc_message}")
