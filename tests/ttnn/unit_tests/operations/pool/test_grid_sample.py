# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.pool import golden_grid_sample


@pytest.mark.parametrize(
    "input_shape, grid_shape",
    [
        ((1, 32, 8, 8), (1, 6, 6, 2)),
        ((2, 64, 16, 16), (2, 12, 12, 2)),
        ((1, 96, 24, 24), (1, 20, 20, 2)),
        ((4, 128, 32, 32), (4, 28, 28, 2)),
        ((8, 160, 48, 48), (8, 40, 40, 2)),
        ((2, 192, 64, 64), (2, 56, 56, 2)),
        ((1, 96, 8, 32), (1, 6, 28, 2)),
        ((2, 160, 32, 8), (2, 28, 6, 2)),
    ],
)
@pytest.mark.parametrize(
    "mode",
    [
        "bilinear",
    ],
)
@pytest.mark.parametrize(
    "align_corners",
    [
        False,
    ],
)
@pytest.mark.parametrize("grid_dtype", [ttnn.bfloat16, ttnn.float32])
def test_grid_sample_random_grid(device, input_shape, mode, align_corners, grid_shape, grid_dtype):
    """Test grid_sample with completely random grid coordinates"""

    torch.manual_seed(0)

    torch_input_nchw = torch.randn(input_shape, dtype=torch.float32)
    torch_input_nhwc = torch_input_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    # Create a random grid with coordinates in range [-1, 1]
    torch_grid_f32 = torch.rand(grid_shape, dtype=torch.float32) * 2.0 - 1.0

    batch_size, channels, height, width = input_shape
    _, grid_h, grid_w, _ = grid_shape

    # Convert inputs to proper format for golden function
    torch_input_formatted = torch_input_nchw.permute(0, 2, 3, 1).reshape(1, 1, batch_size * height * width, channels)
    torch_grid_formatted = torch_grid_f32.reshape(1, 1, batch_size * grid_h * grid_w, 2)

    torch_output_formatted = golden_grid_sample(
        input_tensor=torch_input_formatted.to(torch.bfloat16),
        grid=torch_grid_formatted.to(torch.bfloat16),
        batch_size=batch_size,
        input_h=height,
        input_w=width,
        channels=channels,
        output_h=grid_h,
        output_w=grid_w,
        mode=mode,
        padding_mode="zeros",
        align_corners=align_corners,
    )
    torch_output_nchw = (
        torch_output_formatted.reshape(batch_size, grid_h, grid_w, channels).permute(0, 3, 1, 2).to(torch.float32)
    )

    torch_output_nhwc = torch_output_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    ttnn_grid = ttnn.from_torch(torch_grid_f32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=grid_dtype)

    ttnn_output = ttnn.grid_sample(ttnn_input, ttnn_grid, mode=mode, align_corners=align_corners)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)
    # Test allclose with grid type specific tolerances
    if grid_dtype == ttnn.float32:
        atol, rtol = 0.02, 1e-2
    else:  # bfloat16
        atol, rtol = 1.0, 1e-1

    allclose_passed = torch.allclose(torch_output_nhwc, ttnn_output_torch, atol=atol, rtol=rtol)

    assert allclose_passed, f"Test failed allclose comparison (atol={atol}, rtol={rtol})"


@pytest.mark.parametrize(
    "input_shape, grid_shape",
    [
        ((1, 32, 16, 16), (1, 12, 12, 2)),
        ((2, 64, 20, 20), (2, 16, 16, 2)),
        ((1, 96, 24, 24), (1, 18, 18, 2)),
        ((4, 128, 28, 28), (4, 22, 22, 2)),
        ((2, 160, 32, 32), (2, 24, 24, 2)),
        ((1, 192, 36, 36), (1, 28, 28, 2)),
        ((3, 224, 40, 40), (3, 32, 32, 2)),
    ],
)
@pytest.mark.parametrize(
    "mode",
    [
        "bilinear",
    ],
)
@pytest.mark.parametrize(
    "align_corners",
    [
        False,
    ],
)
@pytest.mark.parametrize("grid_dtype", [ttnn.bfloat16, ttnn.float32])
def test_grid_sample_near_uniform_grid(device, input_shape, mode, align_corners, grid_shape, grid_dtype):
    torch.manual_seed(0)

    batch_size, grid_h, grid_w, _ = grid_shape

    torch_input_nchw = torch.randn(input_shape, dtype=torch.float32)
    torch_input_nhwc = torch_input_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    # Generates a uniform grid using torch affine grid
    theta = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    theta_batched = theta.unsqueeze(0).expand(batch_size, -1, -1)
    shape = (batch_size, 1, grid_h, grid_w)
    torch_grid = F.affine_grid(theta_batched, shape, align_corners=False)

    # Add small noise to the grid
    torch_grid += torch.randn(grid_shape) * 0.05

    # Use golden function
    from ttnn.operations.pool import golden_grid_sample

    batch_size, channels, height, width = input_shape
    grid_batch_size, grid_h, grid_w, _ = grid_shape

    # Convert inputs to proper format for golden function
    torch_input_formatted = torch_input_nchw.permute(0, 2, 3, 1).reshape(1, 1, batch_size * height * width, channels)
    torch_grid_formatted = torch_grid.reshape(1, 1, batch_size * grid_h * grid_w, 2)

    torch_output_formatted = golden_grid_sample(
        input_tensor=torch_input_formatted.to(torch.bfloat16),
        grid=torch_grid_formatted.to(torch.bfloat16),
        batch_size=batch_size,
        input_h=height,
        input_w=width,
        channels=channels,
        output_h=grid_h,
        output_w=grid_w,
        mode=mode,
        padding_mode="zeros",
        align_corners=align_corners,
    )
    torch_output_nchw = (
        torch_output_formatted.reshape(batch_size, grid_h, grid_w, channels).permute(0, 3, 1, 2).to(torch.float32)
    )

    torch_output_nhwc = torch_output_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    ttnn_grid = ttnn.from_torch(torch_grid, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=grid_dtype)

    ttnn_output = ttnn.grid_sample(ttnn_input, ttnn_grid, mode=mode, align_corners=align_corners)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    pcc_passed, pcc_message = assert_with_pcc(torch_output_nhwc, ttnn_output_torch, pcc=0.99)

    # Test allclose with grid type specific tolerances
    if grid_dtype == ttnn.float32:
        atol, rtol = 0.02, 1e-2
    else:  # bfloat16
        atol, rtol = 1.0, 1e-1

    allclose_passed = torch.allclose(torch_output_nhwc, ttnn_output_torch, atol=atol, rtol=rtol)

    assert pcc_passed, f"Test failed with PCC below threshold"
    assert allclose_passed, f"Test failed allclose comparison (atol={atol}, rtol={rtol})"


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 16, 16, 32),
        (1, 32, 32, 64),
        (2, 16, 16, 32),
    ],
)
@pytest.mark.parametrize(
    "mode",
    [
        "bilinear",
    ],
)
@pytest.mark.parametrize(
    "align_corners",
    [
        False,
    ],
)
def test_grid_sample_identity_transform(device, input_shape, mode, align_corners):
    """Test grid_sample with identity transformation (should return original image)"""

    torch.manual_seed(0)

    batch_size, height, width, _ = input_shape

    torch_input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)

    # Generates a grid that corresponds to the identity transformation
    theta = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    theta_batched = theta.unsqueeze(0).expand(batch_size, -1, -1)
    shape = (batch_size, 1, height, width)
    torch_grid = F.affine_grid(theta_batched, shape, align_corners=align_corners)

    torch_input_nchw = torch_input_nhwc.permute(0, 3, 1, 2).to(torch.float32)

    # Use golden function
    from ttnn.operations.pool import golden_grid_sample

    batch_size, channels, height, width = input_shape
    grid_batch_size, grid_h, grid_w, _ = grid_shape

    # Convert inputs to proper format for golden function
    torch_input_formatted = torch_input_nchw.permute(0, 2, 3, 1).reshape(1, 1, batch_size * height * width, channels)
    torch_grid_formatted = torch_grid.reshape(1, 1, batch_size * grid_h * grid_w, 2)

    torch_output_formatted = golden_grid_sample(
        input_tensor=torch_input_formatted.to(torch.bfloat16),
        grid=torch_grid_formatted.to(torch.bfloat16),
        batch_size=batch_size,
        input_h=height,
        input_w=width,
        channels=channels,
        output_h=grid_h,
        output_w=grid_w,
        mode=mode,
        padding_mode="zeros",
        align_corners=align_corners,
    )
    torch_output_nchw = (
        torch_output_formatted.reshape(batch_size, grid_h, grid_w, channels).permute(0, 3, 1, 2).to(torch.float32)
    )
    torch_output_nhwc = torch_output_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_grid = ttnn.from_torch(torch_grid, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    ttnn_output = ttnn.grid_sample(ttnn_input, ttnn_grid, mode=mode, align_corners=align_corners)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    pcc_passed, pcc_message = assert_with_pcc(torch_output_nhwc, ttnn_output_torch, pcc=0.99)
    logger.info(pcc_message)

    pcc_identity, identity_message = assert_with_pcc(torch_input_nhwc, ttnn_output_torch, pcc=0.99)
    logger.info(f"Identity check: {identity_message}")


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
@pytest.mark.parametrize(
    "mode",
    [
        "bilinear",
    ],
)
@pytest.mark.parametrize(
    "align_corners",
    [
        False,
    ],
)
def test_grid_sample_scaling_patterns(device, input_shape, mode, align_corners, scale_factor):
    """Test grid_sample with different scaling patterns"""

    torch.manual_seed(0)

    # Calculate output size based on scale factor
    batch_size = input_shape[0]
    output_h = int(input_shape[1] * scale_factor)
    output_w = int(input_shape[2] * scale_factor)

    torch_input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)

    # Generates a grid that corresponds to downsampling / integer factor upscaling and fractional factor upsampling
    theta = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    theta_batched = theta.unsqueeze(0).expand(batch_size, -1, -1)
    shape = (batch_size, 1, output_h, output_w)
    torch_grid = F.affine_grid(theta_batched, shape, align_corners=align_corners)

    torch_input_nchw = torch_input_nhwc.permute(0, 3, 1, 2).to(torch.float32)

    # Use golden function
    from ttnn.operations.pool import golden_grid_sample

    batch_size = input_shape[0]
    channels = input_shape[3]  # NHWC format
    height = input_shape[1]
    width = input_shape[2]
    grid_h = torch_grid.shape[2]
    grid_w = torch_grid.shape[3]

    # Convert inputs to proper format for golden function
    torch_input_formatted = torch_input_nchw.permute(0, 2, 3, 1).reshape(1, 1, batch_size * height * width, channels)
    torch_grid_formatted = torch_grid.reshape(1, 1, batch_size * grid_h * grid_w, 2)

    torch_output_formatted = golden_grid_sample(
        input_tensor=torch_input_formatted.to(torch.bfloat16),
        grid=torch_grid_formatted.to(torch.bfloat16),
        batch_size=batch_size,
        input_h=height,
        input_w=width,
        channels=channels,
        output_h=grid_h,
        output_w=grid_w,
        mode=mode,
        padding_mode="zeros",
        align_corners=align_corners,
    )
    torch_output_nchw = (
        torch_output_formatted.reshape(batch_size, grid_h, grid_w, channels).permute(0, 3, 1, 2).to(torch.float32)
    )
    torch_output_nhwc = torch_output_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    torch_grid_bf16 = torch_grid.to(torch.bfloat16)

    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_grid = ttnn.from_torch(torch_grid_bf16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    ttnn_output = ttnn.grid_sample(ttnn_input, ttnn_grid, mode=mode, align_corners=align_corners)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    pcc_passed, pcc_message = assert_with_pcc(torch_output_nhwc, ttnn_output_torch, pcc=0.99)
    logger.info(pcc_message)
