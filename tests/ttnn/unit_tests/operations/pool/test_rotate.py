# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
import torchvision.transforms.functional as TF


def run_rotate_test(
    device,
    input_shape,
    angle,
    center=None,
    fill=0.0,
    interpolation_mode="nearest",
    input_dtype=ttnn.bfloat16,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    check_values=True,
):
    """Common function to run rotate tests with specified parameters."""
    torch.manual_seed(0)

    # Generate input tensor
    input_torch = torch.randn(input_shape, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(
        input_torch, device=device, dtype=input_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=memory_config
    )

    # Run TTNN rotate
    if center is not None:
        ttnn_result = ttnn.rotate(input_tensor, angle, center=center, fill=fill, interpolation_mode=interpolation_mode)
    else:
        ttnn_result = ttnn.rotate(input_tensor, angle, fill=fill, interpolation_mode=interpolation_mode)

    # Verify output properties
    assert ttnn_result is not None
    assert list(ttnn_result.shape) == list(input_tensor.shape)
    assert ttnn_result.dtype == input_tensor.dtype

    # Convert TTNN result back to torch
    ttnn_result_torch = ttnn.to_torch(ttnn_result)

    # Perform value comparison against torch reference if requested
    if check_values:
        # Prepare input for torch reference - convert to float and NCHW format
        # input_shape is (N, H, W, C), torch expects (N, C, H, W)
        input_nchw = input_torch.permute(0, 3, 1, 2).float()

        # Run torch reference implementation
        if interpolation_mode == "nearest":
            torch_interp = TF.InterpolationMode.NEAREST
        else:
            torch_interp = TF.InterpolationMode.BILINEAR

        if center is not None:
            torch_result_nchw = TF.rotate(input_nchw, angle, interpolation=torch_interp, fill=fill, center=center)
        else:
            torch_result_nchw = TF.rotate(input_nchw, angle, interpolation=torch_interp, fill=fill)

        # Convert torch result back to NHWC format and bfloat16
        torch_result = torch_result_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

        # Compare TTNN result against torch reference
        if angle % 360.0 == 0.0:
            # Identity rotation - use stricter tolerance
            is_close = torch.allclose(ttnn_result_torch, torch_result, rtol=1e-3, atol=1e-4)
            assert is_close, f"Identity rotation comparison failed for angle {angle} degrees"
        else:
            # Non-identity rotation - use looser tolerances
            if input_dtype == ttnn.bfloat16:
                is_close = torch.allclose(ttnn_result_torch, torch_result, rtol=0.1, atol=0.1)
            else:  # float32
                is_close = torch.allclose(ttnn_result_torch, torch_result, rtol=0.1, atol=0.1)
            assert is_close, f"TTNN vs Torch comparison failed for angle {angle} degrees"

    return input_torch, ttnn_result_torch


# ============================================================================
# Basic Functionality Tests
# ============================================================================


def test_identity_rotation(device):
    """Test that 0-degree rotation exactly preserves the input."""
    input_torch, result_torch = run_rotate_test(device, (1, 32, 32, 32), 0.0)

    # For identity rotation with nearest neighbor, expect exact equality
    # Since it's just data movement without interpolation
    torch.testing.assert_close(result_torch, input_torch, rtol=0, atol=0)


def test_various_angles(device):
    """Test various rotation angles."""
    angles = [0.0, 45.0, 90.0, 180.0, -45.0, 360.0, 23.7, 142.8, 201.3]

    for angle in angles:
        run_rotate_test(device, (1, 32, 32, 32), angle)
        # Just verify the operation succeeds and maintains shape


def test_custom_center(device):
    """Test rotation with custom center points."""
    test_cases = [
        (0.0, 0.0),  # Top-left corner
        (16.0, 16.0),  # Center of 32x32 image
        (31.0, 31.0),  # Bottom-right corner
        (10.5, 20.5),  # Non-integer center
    ]

    for center in test_cases:
        run_rotate_test(device, (1, 32, 32, 32), 45.0, center=center)


def test_custom_fill_values(device):
    """Test different fill values for out-of-bounds pixels."""
    fill_values = [0.0, 1.0, -1.0, 0.5]

    for fill in fill_values:
        run_rotate_test(device, (1, 32, 32, 32), 45.0, fill=fill)


# ============================================================================
# Shape and Size Tests
# ============================================================================


def test_various_tensor_sizes(device):
    """Test various tensor shapes and sizes."""
    test_shapes = [
        (1, 32, 32, 32),  # Small square
        (1, 64, 64, 32),  # Medium square
        (1, 32, 64, 32),  # Rectangle
        (2, 32, 32, 32),  # Batch size > 1
        (1, 96, 96, 64),  # Larger tensor
        (4, 48, 48, 16),  # Multiple batch, different dims
    ]

    for shape in test_shapes:
        run_rotate_test(device, shape, 45.0)


def test_channel_alignment(device):
    """Test different channel sizes that meet alignment requirements."""
    # Channels must align to 32 bytes for bfloat16 (16 channels = 32 bytes)
    channel_sizes = [16, 32, 48, 64, 96, 128]

    for channels in channel_sizes:
        run_rotate_test(device, (1, 32, 32, channels), 90.0)


# ============================================================================
# Memory Configuration Tests
# ============================================================================


@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_memory_configs(device, memory_config):
    """Test different memory configurations."""
    run_rotate_test(device, (1, 32, 32, 32), 45.0, memory_config=memory_config)


# ============================================================================
# Data Type Tests
# ============================================================================


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_data_types(device, dtype):
    """Test different data types."""
    run_rotate_test(device, (1, 32, 32, 32), 45.0, input_dtype=dtype)


# ============================================================================
# Validation Tests
# ============================================================================


def test_validation_wrong_rank(device):
    """Test that wrong tensor rank is rejected."""
    input_3d = torch.randn(32, 32, 32, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(input_3d, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    with pytest.raises(RuntimeError, match="4D"):
        ttnn.rotate(input_tensor, 45.0)


def test_validation_expand_true(device):
    """Test that expand=True is rejected."""
    input_data = torch.randn(1, 32, 32, 32, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(input_data, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    with pytest.raises(RuntimeError, match="expand"):
        ttnn.rotate(input_tensor, 45.0, expand=True)


def test_validation_interpolation_mode(device):
    """Test that only 'nearest' interpolation mode is accepted."""
    input_data = torch.randn(1, 32, 32, 32, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(input_data, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # Should accept "nearest"
    result = ttnn.rotate(input_tensor, 45.0, interpolation_mode="nearest")
    assert result is not None

    # Should reject "bilinear" (now that it's removed)
    with pytest.raises(RuntimeError, match="interpolation_mode"):
        ttnn.rotate(input_tensor, 45.0, interpolation_mode="bilinear")


def test_validation_channel_alignment(device):
    """Test that unaligned channel dimensions are rejected."""
    # Create tensor with 15 channels (15 * 2 bytes = 30 bytes, not aligned to 32)
    input_data = torch.randn(1, 32, 32, 15, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(input_data, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    with pytest.raises(RuntimeError, match="Channel dimension must be aligned"):
        ttnn.rotate(input_tensor, 45.0)


# ============================================================================
# Edge Cases
# ============================================================================


def test_full_rotation(device):
    """Test 360-degree rotation should be exactly equal to identity."""
    input_torch, result_torch = run_rotate_test(device, (1, 32, 32, 32), 360.0)

    # 360-degree rotation should be exactly equal to identity for data movement
    torch.testing.assert_close(result_torch, input_torch, rtol=0, atol=0)


def test_negative_angles(device):
    """Test negative rotation angles."""
    angles = [-45.0, -90.0, -180.0, -270.0]

    for angle in angles:
        run_rotate_test(device, (1, 32, 32, 32), angle)


def test_large_angles(device):
    """Test angles greater than 360 degrees."""
    angles = [405.0, 720.0, -450.0]

    for angle in angles:
        run_rotate_test(device, (1, 32, 32, 32), angle)


def test_small_tensor(device):
    """Test minimal tensor size."""
    # Test with minimum practical size
    run_rotate_test(device, (1, 32, 32, 16), 45.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
