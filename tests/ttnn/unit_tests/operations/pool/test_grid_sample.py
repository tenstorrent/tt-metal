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
def test_grid_sample_random_grid(device, input_shape, grid_shape):
    """Test grid_sample with completely random grid coordinates"""

    torch.manual_seed(0)

    torch_input_nchw = torch.randn(input_shape, dtype=torch.float32)
    torch_input_nhwc = torch_input_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    # Makes a random grid with coordinates in range [-1, 1]
    torch_grid = torch.rand(grid_shape, dtype=torch.bfloat16) * 2.0 - 1.0

    torch_grid_float = torch_grid.to(torch.float32)

    torch_output_nchw = F.grid_sample(
        torch_input_nchw, torch_grid_float, mode="bilinear", padding_mode="zeros", align_corners=False
    )

    torch_output_nhwc = torch_output_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_grid = ttnn.from_torch(torch_grid, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    ttnn_output = ttnn.grid_sample(ttnn_input, ttnn_grid)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    pcc_passed, pcc_message = assert_with_pcc(torch_output_nhwc, ttnn_output_torch, pcc=0.99)
    logger.info(pcc_message)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
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
def test_grid_sample_near_uniform_grid(device, input_shape, grid_shape):
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

    torch_output_nchw = F.grid_sample(
        torch_input_nchw, torch_grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )

    torch_grid_bf16 = torch_grid.to(torch.bfloat16)

    torch_output_nhwc = torch_output_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_grid = ttnn.from_torch(torch_grid_bf16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    ttnn_output = ttnn.grid_sample(ttnn_input, ttnn_grid)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    pcc_passed, pcc_message = assert_with_pcc(torch_output_nhwc, ttnn_output_torch, pcc=0.99)
    logger.info(pcc_message)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 16, 16, 32),
        (1, 32, 32, 64),
        (2, 16, 16, 32),
    ],
)
def test_grid_sample_identity_transform(device, input_shape):
    """Test grid_sample with identity transformation (should return original image)"""

    torch.manual_seed(0)

    batch_size, height, width, _ = input_shape

    torch_input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)

    # Generates a grid that corresponds to the identity transformation
    theta = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    theta_batched = theta.unsqueeze(0).expand(batch_size, -1, -1)
    shape = (batch_size, 1, height, width)
    torch_grid = F.affine_grid(theta_batched, shape, align_corners=False)

    torch_input_nchw = torch_input_nhwc.permute(0, 3, 1, 2).to(torch.float32)

    torch_output_nchw = F.grid_sample(
        torch_input_nchw, torch_grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    torch_output_nhwc = torch_output_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    torch_grid_bf16 = torch_grid.to(torch.bfloat16)

    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_grid = ttnn.from_torch(torch_grid_bf16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    ttnn_output = ttnn.grid_sample(ttnn_input, ttnn_grid)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    pcc_passed, pcc_message = assert_with_pcc(torch_output_nhwc, ttnn_output_torch, pcc=0.99)
    logger.info(pcc_message)

    pcc_identity, identity_message = assert_with_pcc(torch_input_nhwc, ttnn_output_torch, pcc=0.99)
    logger.info(f"Identity check: {identity_message}")


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
    batch_size = input_shape[0]
    output_h = int(input_shape[1] * scale_factor)
    output_w = int(input_shape[2] * scale_factor)

    torch_input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)

    # Generates a grid that corresponds to downsampling / integer factor upscaling and fractional factor upsampling
    theta = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    theta_batched = theta.unsqueeze(0).expand(batch_size, -1, -1)
    shape = (batch_size, 1, output_h, output_w)
    torch_grid = F.affine_grid(theta_batched, shape, align_corners=False)

    torch_input_nchw = torch_input_nhwc.permute(0, 3, 1, 2).to(torch.float32)

    torch_output_nchw = F.grid_sample(
        torch_input_nchw, torch_grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    torch_output_nhwc = torch_output_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    torch_grid_bf16 = torch_grid.to(torch.bfloat16)

    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_grid = ttnn.from_torch(torch_grid_bf16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    ttnn_output = ttnn.grid_sample(ttnn_input, ttnn_grid)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    pcc_passed, pcc_message = assert_with_pcc(torch_output_nhwc, ttnn_output_torch, pcc=0.99)
    logger.info(pcc_message)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
@pytest.mark.parametrize("use_precomputed_grid", [True])  # , False])
@pytest.mark.parametrize(
    "input_shape, grid_shape",
    [
        # ((1, 32, 64, 64), (1, 32, 32, 2)),
        # ((2, 64, 96, 96), (2, 64, 64, 2)),
        # ((1, 128, 128, 128), (1, 96, 96, 2)),
        # ((1, 96, 32, 32), (1, 64, 32, 2)),
        # ((2, 160, 48, 48), (2, 17, 100, 2)),
        ((1, 256, 48, 160), (1, 7, 25280, 2)),
    ],
)
def test_grid_sample_tiled_grid(device, input_shape, grid_shape, use_precomputed_grid):
    """Test grid_sample with TILE layout grid tensor"""

    torch.manual_seed(0)

    # Create input and grid tensors
    torch_input_nchw = torch.randn(input_shape, dtype=torch.float32)
    torch_input_nhwc = torch_input_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    # Convert input_shape from NCHW to NHWC format for prepare_grid_sample_grid
    batch_size, channels, height, width = input_shape
    input_shape_nhwc = [batch_size, height, width, channels]

    # Create near-uniform grid with small random perturbation for meaningful testing
    # torch_grid = torch.rand(grid_shape, dtype=torch.bfloat16) * 2.0 - 1.0  # Small values near zero

    torch_grid = torch.zeros(grid_shape, dtype=torch.bfloat16)

    # Get expected PyTorch output
    torch_grid_float = torch_grid.to(torch.float32)
    torch_output_nchw = F.grid_sample(
        torch_input_nchw, torch_grid_float, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    torch_output_nhwc = torch_output_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    # Create TTNN tensors
    ttnn_input = ttnn.from_torch(
        torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    if use_precomputed_grid:
        # IMPORTANT: Precompute grid while it's in ROW_MAJOR layout on HOST with float32 dtype
        ttnn_grid_rowmajor_host = ttnn.from_torch(torch_grid_float, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32)

        # Precompute the grid (this function expects ROW_MAJOR float32 input on host with NHWC shape)
        ttnn_precomputed_grid_host = ttnn.prepare_grid_sample_grid(
            ttnn_grid_rowmajor_host, input_shape_nhwc, padding_mode="zeros", output_dtype=ttnn.bfloat16
        )

        # NOW convert the precomputed grid to TILE layout and move to device
        ttnn_grid_tiled = ttnn.to_layout(ttnn_precomputed_grid_host, ttnn.TILE_LAYOUT)
        print("Grid shapes")
        print(ttnn_grid_tiled.shape)
        print(ttnn_grid_tiled.padded_shape)
        ttnn_grid_device = ttnn.to_device(ttnn_grid_tiled, device)

        print(ttnn_grid_device.layout)
        # Test with tiled precomputed grid
        ttnn_output = ttnn.grid_sample(ttnn_input, ttnn_grid_device, use_precomputed_grid=True)
        print("Output shapes")
        print(ttnn_output.shape)
        print(ttnn_output.padded_shape)

    else:
        # Convert regular grid to TILE layout
        ttnn_grid_rowmajor = ttnn.from_torch(torch_grid, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        ttnn_grid_tiled = ttnn.to_layout(ttnn_grid_rowmajor, ttnn.TILE_LAYOUT)
        # print("Started grid sample")
        # Test with tiled regular grid
        ttnn_output = ttnn.grid_sample(ttnn_input, ttnn_grid_tiled, use_precomputed_grid=False)

    # Verify results
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    pcc_passed, pcc_message = assert_with_pcc(torch_output_nhwc, ttnn_output_torch, pcc=0.99)
    logger.info(f"Tiled grid test (precomputed={use_precomputed_grid}): {pcc_message}")
    assert pcc_passed
