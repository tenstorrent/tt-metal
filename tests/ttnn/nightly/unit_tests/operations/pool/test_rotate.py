# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.unit_tests.operations.pool.test_rotate import get_rotate_tolerances

import ttnn


# ============================================================================
# Basic Functionality Tests
# ============================================================================


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
@pytest.mark.parametrize(
    "interpolation_mode",
    ["nearest", "bilinear"],
)
def test_various_angles(device, input_shape, angle, interpolation_mode):
    """Test rotate with various rotation angles for both interpolation modes"""

    torch.manual_seed(0)

    torch_input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)

    golden_function = ttnn.get_golden_function(ttnn.rotate)
    torch_output_nhwc = golden_function(torch_input_nhwc, angle=float(angle), interpolation_mode=interpolation_mode)

    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_output = ttnn.rotate(ttnn_input, angle=float(angle), interpolation_mode=interpolation_mode)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    assert (
        ttnn_output_torch.shape == torch_output_nhwc.shape
    ), f"Shape mismatch: ttnn={ttnn_output_torch.shape}, torch={torch_output_nhwc.shape}"

    atol, rtol = get_rotate_tolerances(input_shape, angle, interpolation_mode)
    comparison_passed = torch.allclose(torch_output_nhwc, ttnn_output_torch, atol=atol, rtol=rtol)
    assert comparison_passed, f"Test failed (mode={interpolation_mode}, angle={angle}°, atol={atol}, rtol={rtol})"


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 8, 8, 32),
        (1, 16, 16, 64),
        (2, 16, 16, 32),
    ],
)
@pytest.mark.parametrize(
    "interpolation_mode",
    ["nearest", "bilinear"],
)
def test_identity_rotation(device, input_shape, interpolation_mode):
    """Test that 0-degree rotation returns the same image"""

    torch.manual_seed(0)

    torch_input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_output = ttnn.rotate(ttnn_input, angle=0.0, interpolation_mode=interpolation_mode)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    atol, rtol = get_rotate_tolerances(input_shape, 0, interpolation_mode)

    if interpolation_mode == "bilinear":
        pcc_passed, pcc_message = assert_with_pcc(torch_input_nhwc, ttnn_output_torch, pcc=0.999)
        logger.info(f"{interpolation_mode} identity transform: {pcc_message}")
        max_diff = (torch_input_nhwc - ttnn_output_torch).abs().max().item()
        assert pcc_passed, f"{interpolation_mode} identity rotation (0°) should preserve input"
        assert max_diff < 0.01, f"{interpolation_mode} identity rotation differs too much: {max_diff}"
    else:
        comparison_passed = torch.allclose(torch_input_nhwc, ttnn_output_torch, atol=atol, rtol=rtol)
        assert comparison_passed, f"{interpolation_mode} identity rotation (0°) failed comparison check"


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
@pytest.mark.parametrize(
    "interpolation_mode",
    ["nearest", "bilinear"],
)
def test_exact_multiples_of_90(device, input_shape, angle, interpolation_mode):
    """Test 90-degree multiples which should be exact"""

    torch.manual_seed(0)

    torch_input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)

    golden_function = ttnn.get_golden_function(ttnn.rotate)
    torch_output_nhwc = golden_function(torch_input_nhwc, angle=float(angle), interpolation_mode=interpolation_mode)

    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_output = ttnn.rotate(ttnn_input, angle=float(angle), interpolation_mode=interpolation_mode)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    if interpolation_mode == "bilinear":
        pcc_passed, pcc_message = assert_with_pcc(torch_output_nhwc, ttnn_output_torch, pcc=0.995)
        logger.info(f"{interpolation_mode} 90° multiple rotation ({angle}°): {pcc_message}")
        max_diff = (torch_output_nhwc - ttnn_output_torch).abs().max().item()
        assert pcc_passed, f"{interpolation_mode} 90-degree rotation ({angle}°) failed PCC check"
        assert max_diff < 0.05, f"{interpolation_mode} 90-degree rotation differs too much: {max_diff}"
    else:
        equal_passed = torch.equal(torch_output_nhwc, ttnn_output_torch)
        assert equal_passed, f"{interpolation_mode} 90-degree rotation should be equal: angle={angle}°"


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 8, 8, 32),
        (1, 16, 16, 64),
    ],
)
@pytest.mark.parametrize(
    "fill_value",
    [0.0, 1.0, -1.0, 0.5],
)
@pytest.mark.parametrize(
    "interpolation_mode",
    ["nearest", "bilinear"],
)
def test_fill_values(device, input_shape, fill_value, interpolation_mode):
    """Test rotation with different fill values for out-of-bounds areas"""

    torch.manual_seed(0)

    torch_input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)
    angle = 45.0

    golden_function = ttnn.get_golden_function(ttnn.rotate)
    torch_output_nhwc = golden_function(
        torch_input_nhwc, angle=angle, interpolation_mode=interpolation_mode, fill=fill_value
    )

    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_output = ttnn.rotate(ttnn_input, angle=angle, interpolation_mode=interpolation_mode, fill=fill_value)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    atol, rtol = get_rotate_tolerances(input_shape, angle, interpolation_mode)
    comparison_passed = torch.allclose(torch_output_nhwc, ttnn_output_torch, atol=atol, rtol=rtol)
    assert comparison_passed, f"Fill value test failed (mode={interpolation_mode}, fill={fill_value})"


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 8, 8, 32),
        (2, 16, 16, 64),
        (4, 16, 16, 32),
    ],
)
@pytest.mark.parametrize(
    "interpolation_mode",
    ["nearest", "bilinear"],
)
def test_batch_consistency(device, input_shape, interpolation_mode):
    """Test that each batch item rotates independently and consistently"""

    torch.manual_seed(0)

    torch_input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)
    angle = 30.0
    batch_size = input_shape[0]

    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_output = ttnn.rotate(ttnn_input, angle=angle, interpolation_mode=interpolation_mode)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    for b in range(batch_size):
        single_input = torch_input_nhwc[b : b + 1]
        ttnn_single = ttnn.from_torch(single_input, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        ttnn_single_output = ttnn.rotate(ttnn_single, angle=angle, interpolation_mode=interpolation_mode)
        ttnn_single_torch = ttnn.to_torch(ttnn_single_output)

        if interpolation_mode == "bilinear":
            max_diff = (ttnn_output_torch[b : b + 1] - ttnn_single_torch).abs().max().item()
            logger.info(f"{interpolation_mode} batch {b} consistency: max_diff={max_diff:.6f}")
            assert max_diff < 1e-5, f"{interpolation_mode} batch {b} differs from single rotation: {max_diff}"
        else:
            equal_passed = torch.equal(ttnn_output_torch[b : b + 1], ttnn_single_torch)
            assert equal_passed, f"{interpolation_mode} batch {b} should be equal to single rotation"


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
@pytest.mark.parametrize(
    "interpolation_mode",
    ["nearest", "bilinear"],
)
def test_multichannel_consistency(device, input_shape, angle, interpolation_mode):
    """Test that all channels rotate identically"""

    torch.manual_seed(0)

    torch_input_nhwc = torch.zeros(input_shape, dtype=torch.bfloat16)
    pattern = torch.randn(input_shape[0], input_shape[1], input_shape[2], 1, dtype=torch.bfloat16)
    torch_input_nhwc[..., 0:1] = pattern
    torch_input_nhwc[..., 1:2] = pattern

    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_output = ttnn.rotate(ttnn_input, angle=float(angle), interpolation_mode=interpolation_mode)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    channel_0 = ttnn_output_torch[..., 0]
    channel_1 = ttnn_output_torch[..., 1]

    if interpolation_mode == "bilinear":
        max_diff = (channel_0 - channel_1).abs().max().item()
        logger.info(f"{interpolation_mode} multi-channel consistency (angle={angle}°): max_diff={max_diff:.6f}")
        assert max_diff < 1e-5, f"{interpolation_mode} channels rotated differently: {max_diff}"
    else:
        equal_passed = torch.equal(channel_0, channel_1)
        assert equal_passed, f"{interpolation_mode} channels should be equal after rotation"


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 12, 12, 32),
        (1, 20, 20, 64),
    ],
)
@pytest.mark.parametrize(
    "interpolation_mode",
    ["nearest", "bilinear"],
)
def test_small_angles(device, input_shape, interpolation_mode):
    """Test rotation with very small angles"""

    torch.manual_seed(0)

    torch_input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)

    for angle in [0.1, 1.0, 5.0]:
        golden_function = ttnn.get_golden_function(ttnn.rotate)
        torch_output_nhwc = golden_function(torch_input_nhwc, angle=angle, interpolation_mode=interpolation_mode)

        ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        ttnn_output = ttnn.rotate(ttnn_input, angle=angle, interpolation_mode=interpolation_mode)
        ttnn_output_torch = ttnn.to_torch(ttnn_output)

        if interpolation_mode == "bilinear":
            pcc_passed, pcc_message = assert_with_pcc(torch_output_nhwc, ttnn_output_torch, pcc=0.99)
            logger.info(f"{interpolation_mode} small angle {angle}°: {pcc_message}")
            assert pcc_passed, f"{interpolation_mode} small angle rotation failed at {angle}°"
        else:
            equal_passed = torch.equal(ttnn_output_torch, torch_output_nhwc)
            assert equal_passed, f"{interpolation_mode} small angle should be equal after rotation"


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 16, 16, 32),
        (2, 24, 24, 64),
    ],
)
@pytest.mark.parametrize(
    "interpolation_mode",
    ["nearest", "bilinear"],
)
def test_full_rotation(device, input_shape, interpolation_mode):
    """Test rotation with large angles (equivalence check)"""

    torch.manual_seed(0)

    torch_input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    ttnn_output_360 = ttnn.rotate(ttnn_input, angle=360.0, interpolation_mode=interpolation_mode)
    ttnn_output_360_torch = ttnn.to_torch(ttnn_output_360)

    ttnn_output_0 = ttnn.rotate(ttnn_input, angle=0.0, interpolation_mode=interpolation_mode)
    ttnn_output_0_torch = ttnn.to_torch(ttnn_output_0)

    is_equal = torch.equal(ttnn_output_0_torch, ttnn_output_360_torch)
    assert is_equal, f"{interpolation_mode} 360° rotation should be equivalent to 0°"


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
        (90, -90),
    ],
)
@pytest.mark.parametrize(
    "interpolation_mode",
    ["nearest", "bilinear"],
)
def test_opposite_angles(device, input_shape, angle_pair, interpolation_mode):
    """Test that rotating by +theta then -theta returns near-original"""

    torch.manual_seed(0)

    torch_input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)
    angle_pos, angle_neg = angle_pair

    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    ttnn_rotated = ttnn.rotate(ttnn_input, angle=float(angle_pos), interpolation_mode=interpolation_mode)
    ttnn_restored = ttnn.rotate(ttnn_rotated, angle=float(angle_neg), interpolation_mode=interpolation_mode)
    ttnn_restored_torch = ttnn.to_torch(ttnn_restored)

    assert torch.equal(
        torch_input_nhwc, ttnn_restored_torch
    ), f"{interpolation_mode} round-trip rotation failed for angles {angle_pos}° and {angle_neg}°"


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
        None,
        (8.0, 8.0),
        (0.0, 0.0),
    ],
)
@pytest.mark.parametrize(
    "interpolation_mode",
    ["nearest", "bilinear"],
)
def test_custom_center(device, input_shape, center, interpolation_mode):
    """Test rotation around custom center points"""

    torch.manual_seed(0)

    torch_input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)
    angle = 45.0

    golden_function = ttnn.get_golden_function(ttnn.rotate)
    torch_output_nhwc = golden_function(
        torch_input_nhwc, angle=angle, center=center, interpolation_mode=interpolation_mode
    )

    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_output = ttnn.rotate(ttnn_input, angle=angle, center=center, interpolation_mode=interpolation_mode)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    if interpolation_mode == "bilinear":
        pcc_passed, pcc_message = assert_with_pcc(torch_output_nhwc, ttnn_output_torch, pcc=0.99)
        logger.info(f"{interpolation_mode} custom center {center}: {pcc_message}")
        atol, rtol = get_rotate_tolerances(input_shape, angle, interpolation_mode)
        allclose_passed = torch.allclose(torch_output_nhwc, ttnn_output_torch, atol=atol, rtol=rtol)
        assert pcc_passed, f"{interpolation_mode} test failed with center={center}"
        assert allclose_passed, f"{interpolation_mode} allclose failed with center={center}"
    else:
        atol, rtol = get_rotate_tolerances(input_shape, angle, interpolation_mode)
        comparison_passed = torch.allclose(torch_output_nhwc, ttnn_output_torch, atol=atol, rtol=rtol)
        assert comparison_passed, f"{interpolation_mode} custom center test failed for center {center}"


# ==================== VADV2 USE CASE TEST ====================


@pytest.mark.parametrize(
    "input_shape",
    [
        (100, 100, 256),
    ],
)
@pytest.mark.parametrize(
    "batch_size",
    [1, 2, 4],
)
@pytest.mark.parametrize(
    "rotation_angle",
    [0.0, -6.64, -1.33, 1.33, 4.80, 25.0],
)
def test_vadv2_use_case(device, input_shape, batch_size, rotation_angle):
    """Test vadv2-specific rotation use case: BEV feature rotation with custom center"""

    torch.manual_seed(0)

    bev_h, bev_w, embed_dims = input_shape

    flattened_bev = torch.randn(bev_h * bev_w, batch_size, embed_dims, dtype=torch.bfloat16)

    rotate_center_x = 100
    rotate_center_y = 100

    torch_outputs = []
    ttnn_outputs = []

    for i in range(batch_size):
        prev_bev_single = flattened_bev[:, i]

        torch_spatial = prev_bev_single.view(bev_h, bev_w, embed_dims)
        torch_spatial_batched = torch_spatial.unsqueeze(0)
        golden_function = ttnn.get_golden_function(ttnn.rotate)
        torch_spatial_out_batched = golden_function(
            torch_spatial_batched,
            angle=rotation_angle,
            center=(rotate_center_x, rotate_center_y),
            interpolation_mode="nearest",
        )
        torch_spatial_out = torch_spatial_out_batched.squeeze(0)
        torch_outputs.append(torch_spatial_out)

        ttnn_spatial = prev_bev_single.view(bev_h, bev_w, embed_dims).unsqueeze(0)
        ttnn_input = ttnn.from_torch(ttnn_spatial, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        ttnn_rotated = ttnn.rotate(
            ttnn_input,
            angle=rotation_angle,
            center=(float(rotate_center_x), float(rotate_center_y)),
            interpolation_mode="nearest",
        )
        ttnn_spatial_out = ttnn.to_torch(ttnn_rotated).squeeze(0)
        ttnn_outputs.append(ttnn_spatial_out)

    for i in range(batch_size):
        torch_result = torch_outputs[i]
        ttnn_result = ttnn_outputs[i]

        assert (
            torch_result.shape == ttnn_result.shape
        ), f"Batch {i}: Shape mismatch torch={torch_result.shape}, ttnn={ttnn_result.shape}"

        atol = 5.0
        rtol = 0.5
        comparison_passed = torch.allclose(torch_result, ttnn_result, atol=atol, rtol=rtol)
        pcc = assert_with_pcc(torch_result, ttnn_result, 0.998)
        assert (
            comparison_passed
        ), f"VADv2 allclose failed for batch {i}, angle {rotation_angle}° (atol={atol}, rtol={rtol})"
        assert pcc, f"VADv2 PCC failed for batch {i}, angle {rotation_angle}°"
