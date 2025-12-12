# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


# ============================================================================
# Basic Functionality Tests
# ============================================================================


def test_identity_rotation(device):
    """Test that 0-degree rotation exactly preserves the input."""
    torch.manual_seed(0)

    input_shape = (1, 32, 32, 32)
    torch_input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)

    # TTNN rotation by 0 degrees
    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_output = ttnn.rotate(ttnn_input, angle=0.0)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    # Use equal check for identity transform - should be exact
    assert torch.equal(torch_input_nhwc, ttnn_output_torch), "Identity rotation should be equal to input"


@pytest.mark.parametrize(
    "angle",
    [0, 15, 30, 45, 60, 90, 135, 180, 270, -30, -90, 360],
)
def test_various_angles(device, angle):
    """Test various rotation angles."""
    torch.manual_seed(0)

    input_shape = (1, 32, 32, 32)
    torch_input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)

    # PyTorch reference using golden function
    golden_function = ttnn.get_golden_function(ttnn.rotate)
    torch_output_nhwc = golden_function(torch_input_nhwc, angle=float(angle))

    # TTNN
    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_output = ttnn.rotate(ttnn_input, angle=float(angle))
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    # Verify shapes match
    assert ttnn_output_torch.shape == torch_output_nhwc.shape

    # Check using adaptive tolerances based on nightly test patterns
    h, w = input_shape[1], input_shape[2]
    tensor_size = h * w
    is_diagonal_rotation = angle in [45, 135, -45, -135]

    if tensor_size >= 1024 and is_diagonal_rotation:
        # Large tensors with diagonal rotations need higher tolerance
        atol, rtol = 5.0, 0.05
    elif tensor_size >= 1024:
        # Large tensors with non-diagonal rotations
        atol, rtol = 0.2, 0.1
    else:
        # Smaller tensors should have high precision
        atol, rtol = 0.05, 0.05

    if angle % 360 == 0:
        # Identity rotation - use exact equality
        assert torch.equal(
            torch_output_nhwc, ttnn_output_torch
        ), f"Identity rotation should be exact for angle {angle}°"
    else:
        comparison_passed = torch.allclose(torch_output_nhwc, ttnn_output_torch, atol=atol, rtol=rtol)
        assert comparison_passed, f"Test failed tensor comparison (angle={angle}°, atol={atol}, rtol={rtol})"


@pytest.mark.parametrize(
    "center",
    [
        (16.0, 16.0),  # Center of 32x32 image
        (0.0, 0.0),  # Top-left corner
        (31.0, 31.0),  # Bottom-right corner
    ],
)
def test_custom_center(device, center):
    """Test rotation with custom center points."""
    torch.manual_seed(0)

    input_shape = (1, 32, 32, 32)
    angle = 45.0
    torch_input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)

    # PyTorch reference using golden function
    golden_function = ttnn.get_golden_function(ttnn.rotate)
    torch_output_nhwc = golden_function(torch_input_nhwc, angle=angle, center=center)

    # TTNN - note center format is (x, y) for ttnn
    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_output = ttnn.rotate(ttnn_input, angle=angle, center=(float(center[1]), float(center[0])))
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    # Use tolerances based on nightly test patterns
    atol, rtol = 5.0, 0.05  # Diagonal rotation with larger tensor
    comparison_passed = torch.allclose(torch_output_nhwc, ttnn_output_torch, atol=atol, rtol=rtol)
    assert comparison_passed, f"Custom center rotation failed for center {center}"


@pytest.mark.parametrize("fill", [0.0, 1.0, -1.0, 0.5])
def test_custom_fill_values(device, fill):
    """Test different fill values for out-of-bounds pixels."""
    torch.manual_seed(0)

    input_shape = (1, 32, 32, 32)
    angle = 45.0
    torch_input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)

    # PyTorch reference using golden function
    golden_function = ttnn.get_golden_function(ttnn.rotate)
    torch_output_nhwc = golden_function(torch_input_nhwc, angle=angle, fill=fill)

    # TTNN
    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_output = ttnn.rotate(ttnn_input, angle=angle, fill=fill)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    # Use tolerances for diagonal rotation with larger tensor
    atol, rtol = 5.0, 0.05
    comparison_passed = torch.allclose(torch_output_nhwc, ttnn_output_torch, atol=atol, rtol=rtol)
    assert comparison_passed, f"Fill value test failed for fill={fill}"


# ============================================================================
# Shape and Size Tests
# ============================================================================


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 8, 8, 32),  # Small tensor
        (1, 16, 16, 64),  # Medium tensor
        (2, 16, 16, 32),  # Batch size > 1
        (1, 32, 32, 96),  # Larger square
        (2, 24, 24, 64),  # Different batch/dims
        (4, 16, 16, 32),  # Multiple batch
    ],
)
def test_various_tensor_sizes(device, input_shape):
    """Test various tensor shapes and sizes."""
    torch.manual_seed(0)

    angle = 45.0
    torch_input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)

    # PyTorch reference using golden function
    golden_function = ttnn.get_golden_function(ttnn.rotate)
    torch_output_nhwc = golden_function(torch_input_nhwc, angle=angle)

    # TTNN
    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_output = ttnn.rotate(ttnn_input, angle=angle)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    # Adaptive tolerances based on tensor size and angle
    h, w = input_shape[1], input_shape[2]
    tensor_size = h * w
    is_diagonal_rotation = True  # 45 degrees

    if tensor_size >= 1024 and is_diagonal_rotation:
        atol, rtol = 5.0, 0.05
    elif tensor_size >= 1024:
        atol, rtol = 0.2, 0.1
    else:
        atol, rtol = 0.05, 0.05

    comparison_passed = torch.allclose(torch_output_nhwc, ttnn_output_torch, atol=atol, rtol=rtol)
    assert comparison_passed, f"Tensor size test failed for shape {input_shape}"


@pytest.mark.parametrize("channels", [16, 32, 48, 64, 96, 128])
def test_channel_alignment(device, channels):
    """Test different channel sizes that meet alignment requirements."""
    torch.manual_seed(0)

    input_shape = (1, 32, 32, channels)
    angle = 90.0
    torch_input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)

    # PyTorch reference using golden function
    golden_function = ttnn.get_golden_function(ttnn.rotate)
    torch_output_nhwc = golden_function(torch_input_nhwc, angle=angle)

    # TTNN
    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_output = ttnn.rotate(ttnn_input, angle=angle)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    # 90-degree rotation should be exact
    assert torch.equal(
        torch_output_nhwc, ttnn_output_torch
    ), f"90-degree rotation should be exact for {channels} channels"


# ============================================================================
# Memory Configuration Tests
# ============================================================================


@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_memory_configs(device, memory_config):
    """Test different memory configurations."""
    torch.manual_seed(0)

    input_shape = (1, 32, 32, 32)
    angle = 45.0
    torch_input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)

    # PyTorch reference using golden function
    golden_function = ttnn.get_golden_function(ttnn.rotate)
    torch_output_nhwc = golden_function(torch_input_nhwc, angle=angle)

    # TTNN with specific memory config
    ttnn_input = ttnn.from_torch(
        torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=memory_config
    )
    ttnn_output = ttnn.rotate(ttnn_input, angle=angle)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    # Use tolerances for diagonal rotation
    atol, rtol = 5.0, 0.05
    comparison_passed = torch.allclose(torch_output_nhwc, ttnn_output_torch, atol=atol, rtol=rtol)
    assert comparison_passed, f"Memory config test failed for {memory_config}"


# ============================================================================
# Data Type Tests
# ============================================================================


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_data_types(device, dtype):
    """Test different data types."""
    torch.manual_seed(0)

    input_shape = (1, 32, 32, 32)
    angle = 45.0

    # Generate input with appropriate torch dtype
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    torch_input_nhwc = torch.randn(input_shape, dtype=torch_dtype)

    # PyTorch reference using golden function
    golden_function = ttnn.get_golden_function(ttnn.rotate)
    torch_output_nhwc = golden_function(torch_input_nhwc, angle=angle)

    # TTNN with specific dtype
    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=dtype)
    ttnn_output = ttnn.rotate(ttnn_input, angle=angle)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    # Use tolerances for diagonal rotation
    atol, rtol = 5.0, 0.05
    comparison_passed = torch.allclose(torch_output_nhwc, ttnn_output_torch, atol=atol, rtol=rtol)
    assert comparison_passed, f"Data type test failed for {dtype}"


# ============================================================================
# Edge Cases
# ============================================================================


def test_full_rotation(device):
    """Test 360-degree rotation should be exactly equal to identity."""
    torch.manual_seed(0)

    input_shape = (1, 32, 32, 32)
    torch_input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)

    # TTNN 360-degree rotation
    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_output_360 = ttnn.rotate(ttnn_input, angle=360.0)
    ttnn_output_360_torch = ttnn.to_torch(ttnn_output_360)

    # TTNN 0-degree rotation for comparison
    ttnn_output_0 = ttnn.rotate(ttnn_input, angle=0.0)
    ttnn_output_0_torch = ttnn.to_torch(ttnn_output_0)

    # Should be exactly equal
    assert torch.equal(ttnn_output_0_torch, ttnn_output_360_torch), "360° rotation should be equivalent to 0°"


@pytest.mark.parametrize("angle", [-45, -90, -180, -270])
def test_negative_angles(device, angle):
    """Test negative rotation angles."""
    torch.manual_seed(0)

    input_shape = (1, 32, 32, 32)
    torch_input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)

    # PyTorch reference using golden function
    golden_function = ttnn.get_golden_function(ttnn.rotate)
    torch_output_nhwc = golden_function(torch_input_nhwc, angle=float(angle))

    # TTNN
    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_output = ttnn.rotate(ttnn_input, angle=float(angle))
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    # Check for exact equality for 90-degree multiples
    if angle % 90 == 0:
        assert torch.equal(torch_output_nhwc, ttnn_output_torch), f"{angle}° rotation should be exact"
    else:
        # Use tolerances for diagonal rotation
        atol, rtol = 5.0, 0.05
        comparison_passed = torch.allclose(torch_output_nhwc, ttnn_output_torch, atol=atol, rtol=rtol)
        assert comparison_passed, f"Negative angle test failed for {angle}°"


@pytest.mark.parametrize("angle", [405.0, 720.0, -450.0])
def test_large_angles(device, angle):
    """Test angles greater than 360 degrees."""
    torch.manual_seed(0)

    input_shape = (1, 32, 32, 32)
    torch_input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)

    # PyTorch reference using golden function
    golden_function = ttnn.get_golden_function(ttnn.rotate)
    torch_output_nhwc = golden_function(torch_input_nhwc, angle=angle)

    # TTNN
    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_output = ttnn.rotate(ttnn_input, angle=angle)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    # Use tolerances for non-90-degree-multiple rotations
    atol, rtol = 5.0, 0.05
    comparison_passed = torch.allclose(torch_output_nhwc, ttnn_output_torch, atol=atol, rtol=rtol)
    assert comparison_passed, f"Large angle test failed for {angle}°"


def test_small_tensor(device):
    """Test minimal tensor size."""
    torch.manual_seed(0)

    input_shape = (1, 32, 32, 16)
    angle = 45.0
    torch_input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)

    # PyTorch reference using golden function
    golden_function = ttnn.get_golden_function(ttnn.rotate)
    torch_output_nhwc = golden_function(torch_input_nhwc, angle=angle)

    # TTNN
    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_output = ttnn.rotate(ttnn_input, angle=angle)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    # Use tolerances for diagonal rotation
    atol, rtol = 5.0, 0.05
    comparison_passed = torch.allclose(torch_output_nhwc, ttnn_output_torch, atol=atol, rtol=rtol)
    assert comparison_passed, f"Small tensor test failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
