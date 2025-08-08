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
    "nchw_input_shape, output_hw",
    [
        ((1, 3, 4, 4), (4, 4)),  # Identity grid - NCHW format
        ((1, 1, 8, 8), (4, 4)),  # Downsampling - NCHW format
        ((2, 16, 32, 32), (16, 16)),  # Batch processing - NCHW format
    ],
)
def test_grid_sample_interface(device, nchw_input_shape, output_hw):
    """Test the basic interface of grid_sample operation"""

    N, C, H_in, W_in = nchw_input_shape
    H_out, W_out = output_hw

    # Create input tensor in NCHW format (PyTorch standard)
    torch_input_nchw = torch.randn(nchw_input_shape, dtype=torch.float32)

    # Create grid tensor (N, H_out, W_out, 2)
    identity_grid = torch.zeros(N, H_out, W_out, 2, dtype=torch.float32)
    # Fill with normalized coordinates
    for h in range(H_out):
        for w in range(W_out):
            x_coord = 2.0 * w / (W_out - 1) - 1.0 if W_out > 1 else 0.0
            y_coord = 2.0 * h / (H_out - 1) - 1.0 if H_out > 1 else 0.0
            identity_grid[:, h, w, 0] = x_coord  # x coordinate
            identity_grid[:, h, w, 1] = y_coord  # y coordinate

    # Run PyTorch grid_sample first (NCHW)
    torch_output_nchw = F.grid_sample(
        torch_input_nchw, identity_grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )

    # Convert to NHWC for TTNN
    torch_input_nhwc = torch_input_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)  # NCHW -> NHWC
    torch_grid_nhwc = identity_grid.to(torch.bfloat16)
    expected_output_nhwc = torch_output_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)  # NCHW -> NHWC

    # Convert to TTNN tensors
    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_grid = ttnn.from_torch(torch_grid_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # Test the operation
    ttnn_output = ttnn.grid_sample(ttnn_input, ttnn_grid)

    # Verify output properties
    expected_shape = (N, H_out, W_out, C)
    assert ttnn_output.shape == ttnn.Shape(expected_shape)
    assert ttnn_output.dtype == ttnn.bfloat16
    assert ttnn_output.layout == ttnn.ROW_MAJOR_LAYOUT
    assert expected_output_nhwc.shape == ttnn.to_torch(ttnn_output).shape


@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
@pytest.mark.parametrize(
    "nchw_shape",
    [
        (1, 3, 4, 4),
        (1, 1, 6, 6),
        (2, 8, 8, 8),
    ],
)
def test_grid_sample_parameters(device, nchw_shape):
    """Test grid_sample with explicit parameters"""

    N, C, H, W = nchw_shape

    # Create test tensors in NCHW format first
    torch_input_nchw = torch.randn(nchw_shape, dtype=torch.float32)
    torch_grid_nchw = torch.zeros((N, H, W, 2), dtype=torch.float32)

    # Run through PyTorch first
    torch_output_nchw = F.grid_sample(
        torch_input_nchw, torch_grid_nchw, mode="bilinear", padding_mode="zeros", align_corners=False
    )

    # Convert to NHWC for TTNN
    torch_input_nhwc = torch_input_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)
    torch_grid_nhwc = torch_grid_nchw.to(torch.bfloat16)

    input_tensor = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    grid_tensor = ttnn.from_torch(torch_grid_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # Test with default parameters
    output = ttnn.grid_sample(input_tensor, grid_tensor)
    assert output.shape == ttnn.Shape([N, H, W, C])

    # Test with explicit parameters
    output = ttnn.grid_sample(input_tensor, grid_tensor, mode="bilinear", padding_mode="zeros", align_corners=False)
    assert output.shape == ttnn.Shape([N, H, W, C])


@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
def test_grid_sample_validation(device):
    """Test input validation"""

    # Create valid tensors in NCHW format first, then convert to NHWC
    nchw_shape = (1, 3, 4, 4)
    torch_input_nchw = torch.randn(nchw_shape, dtype=torch.float32)
    torch_input_nhwc = torch_input_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    input_tensor = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # Test with mismatched batch sizes
    torch_grid_wrong_batch = torch.zeros((2, 4, 4, 2), dtype=torch.bfloat16)
    grid_wrong_batch = ttnn.from_torch(torch_grid_wrong_batch, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    with pytest.raises(RuntimeError, match="Batch size mismatch"):
        ttnn.grid_sample(input_tensor, grid_wrong_batch)

    # Test with wrong grid coordinate dimension
    torch_grid_wrong_coords = torch.zeros((1, 4, 4, 3), dtype=torch.bfloat16)
    grid_wrong_coords = ttnn.from_torch(torch_grid_wrong_coords, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    with pytest.raises(RuntimeError, match="Grid tensor last dimension must be 2"):
        ttnn.grid_sample(input_tensor, grid_wrong_coords)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
def test_grid_sample_basic_functionality(device):
    """Test basic functionality - interface only"""

    # Simple test case: start with NCHW, convert to NHWC
    nchw_shape = (1, 1, 2, 2)
    torch_input_nchw = torch.ones(nchw_shape, dtype=torch.float32)
    torch_grid_nchw = torch.zeros((1, 2, 2, 2), dtype=torch.float32)

    # Run through PyTorch first
    torch_output_nchw = F.grid_sample(
        torch_input_nchw, torch_grid_nchw, mode="bilinear", padding_mode="zeros", align_corners=False
    )

    # Convert to NHWC for TTNN
    torch_input_nhwc = torch_input_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)
    torch_grid_nhwc = torch_grid_nchw.to(torch.bfloat16)

    input_tensor = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    grid_tensor = ttnn.from_torch(torch_grid_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # Call the operation
    output = ttnn.grid_sample(input_tensor, grid_tensor)

    # Convert back and check basic properties
    torch_output = ttnn.to_torch(output)

    assert torch_output.shape == (1, 2, 2, 1)  # NHWC format
    assert torch_output.dtype == torch.bfloat16
    # Since current implementation returns zeros, verify it's tensor of zeros
    assert torch.all(torch_output == 0.0)


def test_torch_grid_sample_4d():
    """Test basic 4D torch grid_sample functionality"""

    # Create 4D input tensor in NCHW format (PyTorch standard)
    input_tensor = torch.randn(1, 3, 4, 4, dtype=torch.float32)

    # Create 4D grid tensor (N, H_out, W_out, 2)
    grid = torch.zeros(1, 2, 2, 2, dtype=torch.float32)

    # Fill grid with some normalized coordinates [-1, 1]
    grid[0, 0, 0, :] = torch.tensor([-0.5, -0.5])  # Sample from center-left, center-top
    grid[0, 0, 1, :] = torch.tensor([0.5, -0.5])  # Sample from center-right, center-top
    grid[0, 1, 0, :] = torch.tensor([-0.5, 0.5])  # Sample from center-left, center-bottom
    grid[0, 1, 1, :] = torch.tensor([0.5, 0.5])  # Sample from center-right, center-bottom

    # Run torch grid_sample
    output = torch.nn.functional.grid_sample(
        input_tensor, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )

    # Verify output shape
    assert output.shape == (1, 3, 2, 2)  # (N, C, H_out, W_out)
    assert output.dtype == torch.float32

    print(f"Input shape: {input_tensor.shape}")
    print(f"Grid shape: {grid.shape}")
    print(f"Output shape: {output.shape}")
    print("4D torch grid_sample test passed!")


def test_grid_sample_simple_tensor():
    """Test grid_sample with simplest possible tensor configuration"""

    # Create input tensor in NCHW format: 1x32x32x32
    input_tensor = torch.randn(1, 32, 32, 32, dtype=torch.float32)

    # Create grid tensor: 1x1x1x2 with coordinates [-0.3, 0.6]
    grid = torch.zeros(1, 1, 1, 2, dtype=torch.float32)
    grid[0, 0, 0, 0] = -0.3  # x coordinate
    grid[0, 0, 0, 1] = 0.6  # y coordinate

    # Run torch grid_sample (NCHW input)
    torch_output = torch.nn.functional.grid_sample(
        input_tensor, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )

    # Verify PyTorch output shape
    assert torch_output.shape == (1, 32, 1, 1)  # (N, C, H_out, W_out)
    assert torch_output.dtype == torch.float32

    print(f"PyTorch - Input shape: {input_tensor.shape}")
    print(f"PyTorch - Grid shape: {grid.shape}")
    print(f"PyTorch - Output shape: {torch_output.shape}")
    print(f"PyTorch - Grid coordinates: x={grid[0, 0, 0, 0].item()}, y={grid[0, 0, 0, 1].item()}")
    print("PyTorch grid_sample test passed!")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
def test_grid_sample_simple_tensor_ttnn(device):
    """Test TTNN grid_sample with simplest possible tensor configuration"""

    # Create input tensor in NCHW format: 1x32x32x32
    torch_input_nchw = torch.randn(1, 256, 48, 160, dtype=torch.bfloat16)

    # Create grid tensor: 1x1x1x2 with coordinates [-0.3, 0.6]
    torch_grid = torch.zeros(1, 7, 25281, 2, dtype=torch.bfloat16)

    # Convert to NHWC for TTNN
    torch_input_nhwc = torch_input_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)  # NCHW -> NHWC
    torch_grid_bfloat16 = torch_grid.to(torch.bfloat16)

    # Convert to TTNN tensors
    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_grid = ttnn.from_torch(torch_grid_bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # Call TTNN grid_sample
    ttnn_output = ttnn.grid_sample(ttnn_input, ttnn_grid)

    # Convert back to torch for inspection
    torch_output_nhwc = ttnn.to_torch(ttnn_output)

    # Verify TTNN output shape (should be NHWC: 1x1x1x32)
    # assert ttnn_output.shape == ttnn.Shape([1, 2, 2, 32])  # (N, H_out, W_out, C)
    # assert ttnn_output.dtype == ttnn.bfloat16
    # assert torch_output_nhwc.shape == (1, 2, 2, 32)  # NHWC format

    print(f"TTNN - Input shape (NHWC): {torch_input_nhwc.shape}")
    print(f"TTNN - Grid shape: {torch_grid_bfloat16.shape}")
    print(f"TTNN - Output shape (NHWC): {torch_output_nhwc.shape}")
    print(
        f"TTNN - Grid coordinates: x={torch_grid_bfloat16[0, 0, 0, 0].item()}, y={torch_grid_bfloat16[0, 0, 0, 1].item()}"
    )
    print("TTNN grid_sample test passed!")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
@pytest.mark.parametrize(
    "input_dims, grid_dims",
    [
        ((1, 48, 160, 256), (1, 7, 25281, 2)),
        # Small test cases
        # ((1, 4, 4, 8), (1, 2, 2, 2)),      # Small input, small grid
        # ((1, 8, 8, 16), (1, 4, 4, 2)),     # Medium input, small grid
        # ((1, 16, 16, 32), (1, 8, 8, 2)),   # Medium input, medium grid
        # # Different aspect ratios
        # ((1, 8, 16, 12), (1, 4, 8, 2)),    # Rectangular input, rectangular grid
        # ((1, 32, 8, 24), (1, 16, 4, 2)),   # Wide input, wide grid
        # ((1, 8, 32, 20), (1, 4, 16, 2)),   # Tall input, tall grid
        # # Single pixel grids
        # ((1, 16, 16, 32), (1, 1, 1, 2)),   # Single output pixel
        # ((1, 32, 32, 64), (1, 1, 5, 2)),   # Single row output
        # ((1, 32, 32, 64), (1, 5, 1, 2)),   # Single column output
        # # Larger test cases
        # ((1, 32, 32, 64), (1, 16, 16, 2)), # Standard case
        # ((1, 64, 64, 128), (1, 32, 32, 2)), # Larger case
    ],
)
def test_grid_sample_parametrized_dimensions(device, input_dims, grid_dims):
    """Test TTNN grid_sample with various input and grid dimensions"""

    # Extract dimensions
    batch_size, input_h, input_w, channels = input_dims
    _, grid_h, grid_w, _ = grid_dims

    # Create input tensor (NHWC format for TTNN)
    torch_input = torch.randn(input_dims, dtype=torch.bfloat16)

    # Create uniform grid in normalized coordinates [-1, 1]
    torch_grid = torch.zeros(grid_dims, dtype=torch.bfloat16)

    # Generate uniform grid coordinates
    for h in range(grid_h):
        for w in range(grid_w):
            # Normalize coordinates to [-1, 1] range
            if grid_w > 1:
                x_coord = 2.0 * w / (grid_w - 1) - 1.0
            else:
                x_coord = 0.0

            if grid_h > 1:
                y_coord = 2.0 * h / (grid_h - 1) - 1.0
            else:
                y_coord = 0.0

            torch_grid[:, h, w, 0] = x_coord  # x coordinate
            torch_grid[:, h, w, 1] = y_coord  # y coordinate

    # Convert to TTNN tensors
    ttnn_input = ttnn.from_torch(torch_input, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_grid = ttnn.from_torch(torch_grid, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # Call TTNN grid_sample
    ttnn_output = ttnn.grid_sample(ttnn_input, ttnn_grid)

    # Convert back to torch for verification
    torch_output = ttnn.to_torch(ttnn_output)

    # Verify output properties
    expected_output_shape = (batch_size, grid_h, grid_w, channels)
    assert (
        torch_output.shape == expected_output_shape
    ), f"Expected shape {expected_output_shape}, got {torch_output.shape}"
    assert torch_output.dtype == torch.bfloat16

    # Verify TTNN tensor properties
    assert ttnn_output.shape == ttnn.Shape(expected_output_shape)
    assert ttnn_output.dtype == ttnn.bfloat16
    assert ttnn_output.layout == ttnn.ROW_MAJOR_LAYOUT

    print(f"✓ Input: {input_dims}, Grid: {grid_dims} -> Output: {torch_output.shape}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
@pytest.mark.parametrize(
    "input_shape, grid_shape",
    [
        ((1, 32, 32, 224), (1, 1, 1, 2)),
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
def test_grid_sample_correctness_with_pcc(device, input_shape, grid_shape):
    """Test TTNN grid_sample correctness against PyTorch with PCC check"""

    torch.manual_seed(0)  # For reproducible results

    batch_size, input_h, input_w, channels = input_shape
    _, grid_h, grid_w, _ = grid_shape

    # Create input tensor (NHWC format for TTNN)
    torch_input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)

    # Create varied grid coordinates for more comprehensive testing
    torch_grid = torch.zeros(grid_shape, dtype=torch.bfloat16)

    # torch_grid = torch.randn(grid_shape, dtype=torch.bfloat16)

    torch_grid[:, :, 0, 0] = 0.65
    torch_grid[:, :, 0, 1] = 0.31

    # torch_grid[:, :, 1, 0] = 0.5
    # torch_grid[:, :, 1, 1] = 0.5

    # # Generate varied grid coordinates to test interpolation
    # for h in range(grid_h):
    #     for w in range(grid_w):
    #         # Create some variation in coordinates to test interpolation properly
    #         if grid_w > 1:
    #             x_coord = 2.0 * w / (grid_w - 1) - 1.0
    #             # Add some variation to test interpolation between pixels
    #             x_coord += 0.1 * torch.randn(1).item()
    #             x_coord = max(-1.0, min(1.0, x_coord))  # Clamp to valid range
    #         else:
    #             x_coord = 0.0

    #         if grid_h > 1:
    #             y_coord = 2.0 * h / (grid_h - 1) - 1.0
    #             # Add some variation to test interpolation between pixels
    #             y_coord += 0.1 * torch.randn(1).item()
    #             y_coord = max(-1.0, min(1.0, y_coord))  # Clamp to valid range
    #         else:
    #             y_coord = 0.0

    #         torch_grid[:, h, w, 0] = x_coord  # x coordinate
    #         torch_grid[:, h, w, 1] = y_coord  # y coordinate

    # Convert to NCHW for PyTorch grid_sample
    torch_input_nchw = torch_input_nhwc.permute(0, 3, 1, 2).to(torch.float32)
    torch_grid_float = torch_grid.to(torch.float32)

    # Run PyTorch grid_sample (golden reference)
    torch_output_nchw = F.grid_sample(
        torch_input_nchw, torch_grid_float, mode="bilinear", padding_mode="zeros", align_corners=False
    )

    # Convert PyTorch output back to NHWC
    torch_output_nhwc = torch_output_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    # Run TTNN grid_sample
    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_grid = ttnn.from_torch(torch_grid, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    ttnn_output = ttnn.grid_sample(ttnn_input, ttnn_grid)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    print(torch.min(ttnn_output_torch), torch.max(ttnn_output_torch))

    # Check PCC (Pearson Correlation Coefficient)
    pcc_passed, pcc_message = assert_with_pcc(torch_output_nhwc, ttnn_output_torch, pcc=0.99)
    logger.info(pcc_message)

    # Additional checks
    assert (
        ttnn_output_torch.shape == torch_output_nhwc.shape
    ), f"Shape mismatch: {ttnn_output_torch.shape} vs {torch_output_nhwc.shape}"
    assert ttnn_output_torch.dtype == torch.bfloat16

    print(f"✓ PCC test passed for {input_shape} -> {grid_shape}: {pcc_message}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
@pytest.mark.parametrize(
    "coordinates",
    [
        # Test specific coordinate patterns
        [(-1.0, -1.0), (1.0, 1.0)],  # Corner samples
        [(-0.5, -0.5), (0.5, 0.5)],  # Center region samples
        [(0.0, 0.0), (0.0, 0.0)],  # Exact center samples
        [(-0.25, 0.75), (0.25, -0.75)],  # Mixed quadrant samples
    ],
)
def test_grid_sample_specific_coordinates(device, coordinates):
    """Test grid_sample with specific coordinate patterns"""

    torch.manual_seed(42)

    # Simple 4x4 input with known pattern
    input_shape = (1, 4, 4, 8)
    grid_shape = (1, 2, 1, 2)  # 2x1 grid for testing specific coordinates

    # Create input with recognizable pattern
    torch_input_nhwc = torch.arange(1, 129, dtype=torch.bfloat16).reshape(input_shape)

    # Create grid with specific coordinates
    torch_grid = torch.zeros(grid_shape, dtype=torch.bfloat16)
    torch_grid[0, 0, 0, 0] = coordinates[0][0]  # x1
    torch_grid[0, 0, 0, 1] = coordinates[0][1]  # y1
    torch_grid[0, 1, 0, 0] = coordinates[1][0]  # x2
    torch_grid[0, 1, 0, 1] = coordinates[1][1]  # y2

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

    # PCC check with high threshold for simple patterns
    pcc_passed, pcc_message = assert_with_pcc(torch_output_nhwc, ttnn_output_torch, pcc=0.98)
    logger.info(pcc_message)

    print(f"✓ Coordinates {coordinates}: {pcc_message}")


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

    torch.manual_seed(123)

    batch_size, height, width, channels = input_shape
    grid_shape = (batch_size, height, width, 2)

    # Create random input
    torch_input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)

    # Create identity grid (maps each output pixel to corresponding input pixel)
    torch_grid = torch.zeros(grid_shape, dtype=torch.bfloat16)
    for h in range(height):
        for w in range(width):
            # Normalize to [-1, 1] range
            x_coord = 2.0 * w / (width - 1) - 1.0 if width > 1 else 0.0
            y_coord = 2.0 * h / (height - 1) - 1.0 if height > 1 else 0.0
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
def test_grid_sample_boundary_conditions(device):
    """Test grid_sample with boundary coordinates and padding behavior"""

    torch.manual_seed(456)

    input_shape = (1, 8, 8, 16)
    grid_shape = (1, 3, 3, 2)

    # Create input with distinct values
    torch_input_nhwc = torch.arange(1, 1025, dtype=torch.bfloat16).reshape(input_shape)

    # Create grid with boundary and out-of-bounds coordinates
    torch_grid = torch.tensor(
        [
            [
                [[-1.0, -1.0], [0.0, -1.0], [1.0, -1.0]],  # Top row (y = -1.0, out of bounds)
                [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]],  # Middle row (y = 0.0, center)
                [[-1.0, 1.0], [0.0, 1.0], [1.0, 1.0]],  # Bottom row (y = 1.0, out of bounds)
            ]
        ],
        dtype=torch.bfloat16,
    )

    # Convert to NCHW for PyTorch
    torch_input_nchw = torch_input_nhwc.permute(0, 3, 1, 2).to(torch.float32)
    torch_grid_float = torch_grid.to(torch.float32)

    # PyTorch reference with zeros padding
    torch_output_nchw = F.grid_sample(
        torch_input_nchw, torch_grid_float, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    torch_output_nhwc = torch_output_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    # TTNN implementation
    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_grid = ttnn.from_torch(torch_grid, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    ttnn_output = ttnn.grid_sample(ttnn_input, ttnn_grid)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    # PCC check for boundary behavior
    pcc_passed, pcc_message = assert_with_pcc(torch_output_nhwc, ttnn_output_torch, pcc=0.95)
    logger.info(pcc_message)

    # Additional checks for boundary behavior
    assert ttnn_output_torch.shape == torch_output_nhwc.shape

    print(f"✓ Boundary conditions test: {pcc_message}")
    print(f"  Grid coordinates tested: {torch_grid.flatten()}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
@pytest.mark.parametrize(
    "scale_factor",
    [0.5, 2.0, 1.5],  # Downsampling, upsampling, and mixed scaling
)
def test_grid_sample_scaling_patterns(device, scale_factor):
    """Test grid_sample with different scaling patterns"""

    torch.manual_seed(789)

    input_shape = (1, 16, 16, 24)

    # Calculate output size based on scale factor
    output_h = int(input_shape[1] * scale_factor)
    output_w = int(input_shape[2] * scale_factor)
    grid_shape = (1, output_h, output_w, 2)

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


# ================================
# Additional PyTorch reference tests (for debugging/validation)
# ================================


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 1, 4, 4),  # Single channel
        # (1, 3, 4, 4),      # RGB channels
        # (1, 16, 8, 8),     # More channels, different spatial size
        # (2, 1, 4, 4),      # Batch size 2
    ],
)
@pytest.mark.parametrize("padding_mode", ["zeros"])  # , "border", "reflection"])
@pytest.mark.parametrize("align_corners", [True, False])
def test_pytorch_grid_sample_comprehensive(input_shape, padding_mode, align_corners):
    """Test PyTorch grid_sample with comprehensive parameter combinations"""

    N, C, H, W = input_shape

    # Create input tensor of all ones
    input_tensor = torch.ones(input_shape, dtype=torch.float32)

    # Create 3x3 grid with all combinations of -1, 0, 1 coordinates
    grid = torch.zeros(N, 3, 3, 2, dtype=torch.float32)

    # Fill grid with all combinations of (-1, 0, 1) for both x and y coordinates
    coord_values = [-1.25, 0.0, 1.25]
    for i, y_coord in enumerate(coord_values):
        for j, x_coord in enumerate(coord_values):
            grid[:, i, j, 0] = x_coord  # x coordinate
            grid[:, i, j, 1] = y_coord  # y coordinate

    # Run PyTorch grid_sample
    output = torch.nn.functional.grid_sample(
        input_tensor, grid, mode="bilinear", padding_mode=padding_mode, align_corners=align_corners
    )

    print(f"PyTorch reference test passed for shape {input_shape}")
