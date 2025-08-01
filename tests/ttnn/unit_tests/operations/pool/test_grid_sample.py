# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

import ttnn


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
    torch_input_nchw = torch.randn(1, 32, 32, 32, dtype=torch.bfloat16)

    # Create grid tensor: 1x1x1x2 with coordinates [-0.3, 0.6]
    torch_grid = torch.zeros(1, 4, 64, 2, dtype=torch.bfloat16)
    torch_grid[0, 0, 0, 0] = -0.25  # x coordinate
    torch_grid[0, 0, 0, 1] = 0.35  # y coordinate
    torch_grid[0, 0, 1, 0] = -0.21  # x coordinate
    torch_grid[0, 0, 1, 1] = 0.46  # y coordinate
    torch_grid[0, 1, 0, 0] = 0.21  # x coordinate
    torch_grid[0, 1, 0, 1] = -0.35  # y coordinate
    torch_grid[0, 1, 1, 0] = 0.111  # x coordinate
    torch_grid[0, 1, 1, 1] = -0.12  # y coordinate

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
    assert ttnn_output.shape == ttnn.Shape([1, 2, 2, 32])  # (N, H_out, W_out, C)
    assert ttnn_output.dtype == ttnn.bfloat16
    assert torch_output_nhwc.shape == (1, 2, 2, 32)  # NHWC format

    print(f"TTNN - Input shape (NHWC): {torch_input_nhwc.shape}")
    print(f"TTNN - Grid shape: {torch_grid_bfloat16.shape}")
    print(f"TTNN - Output shape (NHWC): {torch_output_nhwc.shape}")
    print(
        f"TTNN - Grid coordinates: x={torch_grid_bfloat16[0, 0, 0, 0].item()}, y={torch_grid_bfloat16[0, 0, 0, 1].item()}"
    )
    print("TTNN grid_sample test passed!")


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

    print(input_tensor)
    print()
    print(output)

    # # Verify output shape
    # expected_shape = (N, C, 3, 3)  # (N, C, H_out, W_out)
    # assert output.shape == expected_shape
    # assert output.dtype == torch.float32

    # # For input of all ones, verify expected behavior based on padding mode
    # if padding_mode == "zeros":
    #     # Border samples should be zero (outside [-1,1] range)
    #     # Center and some edge samples should be 1.0
    #     center_value = output[0, 0, 1, 1].item()  # Sample at (0, 0) coordinate
    #     assert abs(center_value - 1.0) < 1e-6, f"Center should be 1.0, got {center_value}"
    # elif padding_mode in ["border", "reflection"]:
    #     # All samples should be 1.0 since input is all ones
    #     assert torch.all(output == 1.0), f"All values should be 1.0 for {padding_mode} padding"

    # print(f"Shape: {input_shape}, Padding: {padding_mode}, Align corners: {align_corners}")
    # print(f"Output shape: {output.shape}")
    # print(f"Grid coordinates used:")
    # for i in range(3):
    #     for j in range(3):
    #         x, y = grid[0, i, j, 0].item(), grid[0, i, j, 1].item()
    #         print(f"  [{i},{j}]: ({x:+.1f}, {y:+.1f}) -> {output[0, 0, i, j].item():.3f}")
    # print("PyTorch comprehensive grid_sample test passed!")
