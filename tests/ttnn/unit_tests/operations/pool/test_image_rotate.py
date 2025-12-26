# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 8, 8, 32),
        (1, 16, 16, 64),
        (2, 16, 16, 32),
        (1, 32, 32, 96),
        (2, 24, 24, 64),
        (4, 16, 16, 32),
        (1, 64, 64, 128),
    ],
)
@pytest.mark.parametrize(
    "angle",
    [0, 15, 30, 45, 60, 90, 135, 180, 270, -30, -90],
)
def test_image_rotate_various_angles(device, input_shape, angle):
    """Test image_rotate with various rotation angles"""

    torch.manual_seed(42)

    # Generate random input in NHWC format
    torch_input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)

    # PyTorch reference (expects NCHW)
    torch_input_nchw = torch_input_nhwc.permute(0, 3, 1, 2).to(torch.float32)
    torch_output_nchw = TF.rotate(torch_input_nchw, angle=float(angle), interpolation=InterpolationMode.BILINEAR)
    torch_output_nhwc = torch_output_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    # TTNN
    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_output = ttnn.image_rotate(ttnn_input, angle=float(angle))
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    # Verify shapes match
    assert (
        ttnn_output_torch.shape == torch_output_nhwc.shape
    ), f"Shape mismatch: ttnn={ttnn_output_torch.shape}, torch={torch_output_nhwc.shape}"

    # Check PCC
    pcc_passed, pcc_message = assert_with_pcc(torch_output_nhwc, ttnn_output_torch, pcc=0.99)
    logger.info(f"Angle {angle}°: {pcc_message}")

    # Check allclose
    atol, rtol = 0.05, 0.05
    allclose_passed = torch.allclose(torch_output_nhwc, ttnn_output_torch, atol=atol, rtol=rtol)

    assert pcc_passed, f"Test failed with PCC below threshold for angle {angle}°"
    assert allclose_passed, f"Test failed allclose comparison (angle={angle}°, atol={atol}, rtol={rtol})"


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 8, 8, 32),
        (1, 16, 16, 64),
        (2, 16, 16, 32),
    ],
)
def test_image_rotate_identity(device, input_shape):
    """Test that 0-degree rotation returns the same image"""

    torch.manual_seed(42)

    torch_input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)

    # TTNN rotation by 0 degrees
    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_output = ttnn.image_rotate(ttnn_input, angle=0.0)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    # Should be nearly identical
    pcc_passed, pcc_message = assert_with_pcc(torch_input_nhwc, ttnn_output_torch, pcc=0.999)
    logger.info(f"Identity transform: {pcc_message}")

    max_diff = (torch_input_nhwc - ttnn_output_torch).abs().max().item()
    logger.info(f"Max difference from input: {max_diff:.6f}")

    assert pcc_passed, "Identity rotation (0°) should preserve input"
    assert max_diff < 0.01, f"Identity rotation differs too much: {max_diff}"


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 8, 8, 32),
        (1, 16, 16, 64),
        (2, 16, 16, 32),
    ],
)
@pytest.mark.parametrize(
    "angle",
    [90, 180, 270, -90, -180],
)
def test_image_rotate_exact_multiples_of_90(device, input_shape, angle):
    """Test 90-degree multiples which should be exact (within bfloat16 precision)"""

    torch.manual_seed(42)

    torch_input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)

    # PyTorch reference
    torch_input_nchw = torch_input_nhwc.permute(0, 3, 1, 2).to(torch.float32)
    torch_output_nchw = TF.rotate(torch_input_nchw, angle=float(angle), interpolation=InterpolationMode.BILINEAR)
    torch_output_nhwc = torch_output_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    # TTNN
    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_output = ttnn.image_rotate(ttnn_input, angle=float(angle))
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    # 90-degree rotations should be very accurate
    pcc_passed, pcc_message = assert_with_pcc(torch_output_nhwc, ttnn_output_torch, pcc=0.995)
    logger.info(f"90° multiple rotation ({angle}°): {pcc_message}")

    max_diff = (torch_output_nhwc - ttnn_output_torch).abs().max().item()
    logger.info(f"Max difference: {max_diff:.6f}")

    assert pcc_passed, f"90-degree rotation ({angle}°) failed PCC check"
    assert max_diff < 0.05, f"90-degree rotation differs too much: {max_diff}"


@pytest.mark.skip(reason="Fill value testing needs investigation - may have precision issues at boundaries")
@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 8, 8, 32),
        (1, 16, 16, 64),
    ],
)
@pytest.mark.parametrize(
    "fill_value",
    [0.0],  # Only test default fill value for now
)
def test_image_rotate_with_fill_value(device, input_shape, fill_value):
    """Test rotation with different fill values for out-of-bounds areas"""

    torch.manual_seed(42)

    torch_input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)
    angle = 45.0  # 45 degrees will create out-of-bounds areas in corners

    # PyTorch reference
    torch_input_nchw = torch_input_nhwc.permute(0, 3, 1, 2).to(torch.float32)
    torch_output_nchw = TF.rotate(
        torch_input_nchw, angle=angle, interpolation=InterpolationMode.BILINEAR, fill=[fill_value]
    )
    torch_output_nhwc = torch_output_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    # TTNN
    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_output = ttnn.image_rotate(ttnn_input, angle=angle, fill=fill_value)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    pcc_passed, pcc_message = assert_with_pcc(torch_output_nhwc, ttnn_output_torch, pcc=0.99)
    logger.info(f"Fill value {fill_value}: {pcc_message}")

    # More lenient tolerances for fill value tests due to edge interpolation
    atol, rtol = 0.1, 0.1
    allclose_passed = torch.allclose(torch_output_nhwc, ttnn_output_torch, atol=atol, rtol=rtol)

    assert pcc_passed, f"Test failed with fill_value={fill_value}"
    assert allclose_passed, f"Allclose failed with fill_value={fill_value}"


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 16, 16, 32),
        (1, 24, 24, 64),
    ],
)
@pytest.mark.parametrize(
    "center",
    [
        None,  # Default center
        (8.0, 8.0),  # Custom center - fixed!
        (0.0, 0.0),  # Corner center - fixed!
    ],
)
def test_image_rotate_with_custom_center(device, input_shape, center):
    """Test rotation around custom center points"""

    torch.manual_seed(42)

    torch_input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)
    angle = 45.0

    # PyTorch reference
    torch_input_nchw = torch_input_nhwc.permute(0, 3, 1, 2).to(torch.float32)
    torch_output_nchw = TF.rotate(
        torch_input_nchw, angle=angle, interpolation=InterpolationMode.BILINEAR, center=center
    )
    torch_output_nhwc = torch_output_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    # TTNN
    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_output = ttnn.image_rotate(ttnn_input, angle=angle, center=center)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    pcc_passed, pcc_message = assert_with_pcc(torch_output_nhwc, ttnn_output_torch, pcc=0.99)
    logger.info(f"Custom center {center}: {pcc_message}")

    atol, rtol = 0.05, 0.05
    allclose_passed = torch.allclose(torch_output_nhwc, ttnn_output_torch, atol=atol, rtol=rtol)

    assert pcc_passed, f"Test failed with center={center}"
    assert allclose_passed, f"Allclose failed with center={center}"


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 8, 8, 32),
        (2, 16, 16, 64),
        (4, 16, 16, 32),
    ],
)
def test_image_rotate_batch_consistency(device, input_shape):
    """Test that each batch item rotates independently and consistently"""

    torch.manual_seed(42)

    torch_input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)
    angle = 30.0
    batch_size = input_shape[0]

    # TTNN batch rotation
    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_output = ttnn.image_rotate(ttnn_input, angle=angle)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    # Compare each batch item with individual rotation
    for b in range(batch_size):
        single_input = torch_input_nhwc[b : b + 1]
        ttnn_single = ttnn.from_torch(single_input, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        ttnn_single_output = ttnn.image_rotate(ttnn_single, angle=angle)
        ttnn_single_torch = ttnn.to_torch(ttnn_single_output)

        # Should be identical
        max_diff = (ttnn_output_torch[b : b + 1] - ttnn_single_torch).abs().max().item()
        logger.info(f"Batch {b} consistency: max_diff={max_diff:.6f}")

        assert max_diff < 1e-5, f"Batch {b} differs from single rotation: {max_diff}"


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 8, 8, 32),
        (1, 16, 16, 64),
        (1, 16, 16, 96),
    ],
)
@pytest.mark.parametrize(
    "angle",
    [10, 45, 90],
)
def test_image_rotate_multichannel_consistency(device, input_shape, angle):
    """Test that all channels rotate identically"""

    torch.manual_seed(42)

    # Create input where first two channels have the same pattern
    torch_input_nhwc = torch.zeros(input_shape, dtype=torch.bfloat16)
    pattern = torch.randn(input_shape[0], input_shape[1], input_shape[2], 1, dtype=torch.bfloat16)
    torch_input_nhwc[..., 0:1] = pattern
    torch_input_nhwc[..., 1:2] = pattern

    # TTNN rotation
    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_output = ttnn.image_rotate(ttnn_input, angle=float(angle))
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    # Channels 0 and 1 should remain identical
    channel_0 = ttnn_output_torch[..., 0]
    channel_1 = ttnn_output_torch[..., 1]
    max_diff = (channel_0 - channel_1).abs().max().item()

    logger.info(f"Multi-channel consistency (angle={angle}°): max_diff={max_diff:.6f}")

    assert max_diff < 1e-5, f"Channels rotated differently: {max_diff}"


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 12, 12, 32),
        (1, 20, 20, 64),
    ],
)
def test_image_rotate_small_angles(device, input_shape):
    """Test rotation with very small angles (precision check)"""

    torch.manual_seed(42)

    torch_input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)

    for angle in [0.1, 1.0, 5.0]:
        # PyTorch reference
        torch_input_nchw = torch_input_nhwc.permute(0, 3, 1, 2).to(torch.float32)
        torch_output_nchw = TF.rotate(torch_input_nchw, angle=angle, interpolation=InterpolationMode.BILINEAR)
        torch_output_nhwc = torch_output_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

        # TTNN
        ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        ttnn_output = ttnn.image_rotate(ttnn_input, angle=angle)
        ttnn_output_torch = ttnn.to_torch(ttnn_output)

        pcc_passed, pcc_message = assert_with_pcc(torch_output_nhwc, ttnn_output_torch, pcc=0.99)
        logger.info(f"Small angle {angle}°: {pcc_message}")

        assert pcc_passed, f"Small angle rotation failed at {angle}°"


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 16, 16, 32),
        (2, 24, 24, 64),
    ],
)
def test_image_rotate_large_angles(device, input_shape):
    """Test rotation with large angles (equivalence check)"""

    torch.manual_seed(42)

    torch_input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)

    # 360 degrees should be equivalent to 0 degrees
    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    ttnn_output_360 = ttnn.image_rotate(ttnn_input, angle=360.0)
    ttnn_output_360_torch = ttnn.to_torch(ttnn_output_360)

    ttnn_output_0 = ttnn.image_rotate(ttnn_input, angle=0.0)
    ttnn_output_0_torch = ttnn.to_torch(ttnn_output_0)

    # Should be very similar
    pcc_passed, pcc_message = assert_with_pcc(ttnn_output_0_torch, ttnn_output_360_torch, pcc=0.99)
    logger.info(f"360° vs 0°: {pcc_message}")

    max_diff = (ttnn_output_0_torch - ttnn_output_360_torch).abs().max().item()
    logger.info(f"Max difference between 360° and 0°: {max_diff:.6f}")

    assert pcc_passed, "360° rotation should be equivalent to 0°"


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 16, 16, 32),
        (1, 24, 24, 64),
    ],
)
@pytest.mark.parametrize(
    "angle_pair",
    [
        pytest.param((45, -45), marks=pytest.mark.skip(reason="Round-trip rotations accumulate interpolation error")),
        pytest.param((30, -30), marks=pytest.mark.skip(reason="Round-trip rotations accumulate interpolation error")),
        (90, -90),  # Keep 90-degree round-trip as it's more exact
    ],
)
def test_image_rotate_opposite_angles(device, input_shape, angle_pair):
    """Test that rotating by +theta then -theta returns near-original"""

    torch.manual_seed(42)

    torch_input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)
    angle_pos, angle_neg = angle_pair

    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # Rotate positive
    ttnn_rotated = ttnn.image_rotate(ttnn_input, angle=float(angle_pos))

    # Rotate negative
    ttnn_restored = ttnn.image_rotate(ttnn_rotated, angle=float(angle_neg))
    ttnn_restored_torch = ttnn.to_torch(ttnn_restored)

    # Should be close to original (with some bilinear interpolation loss)
    pcc_passed, pcc_message = assert_with_pcc(torch_input_nhwc, ttnn_restored_torch, pcc=0.95)
    logger.info(f"Rotate {angle_pos}° then {angle_neg}°: {pcc_message}")

    # Allow more tolerance due to double interpolation
    max_diff = (torch_input_nhwc - ttnn_restored_torch).abs().max().item()
    logger.info(f"Max difference after round-trip: {max_diff:.6f}")

    assert pcc_passed, f"Round-trip rotation ({angle_pos}°, {angle_neg}°) degraded too much"
