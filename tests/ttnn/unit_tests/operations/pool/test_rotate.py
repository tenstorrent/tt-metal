# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger
from models.common.utility_functions import skip_for_blackhole
from tests.ttnn.utils_for_testing import assert_with_pcc


def get_rotate_tolerances(input_shape, angle, interpolation_mode):
    """
    Get appropriate tolerances for rotate operation based on tensor size, angle, and interpolation mode.

    The tolerances are calibrated to match the numerical accuracy of the rotate operation,
    which uses fixed-point arithmetic (Q16.16) for coordinate calculations. Bilinear mode
    uses the same tolerance as grid_sample operation.

    Args:
        input_shape (tuple): The shape of the input tensor (N, H, W, C).
        angle (float): The rotation angle in degrees.
        interpolation_mode (str): Either "nearest" or "bilinear".

    Returns:
        tuple: A tuple containing (atol, rtol) tolerances for torch.allclose comparison.
    """
    is_diagonal_rotation = angle in [45, 135, -45, -135]

    if interpolation_mode == "nearest":
        if is_diagonal_rotation:
            return 6.0, 0.01
        else:
            return 5.0, 0.01
    else:
        return 1.0, 0.01


# ============================================================================
# Basic Functionality Tests
# ============================================================================


@pytest.mark.parametrize("interpolation_mode", ["nearest", "bilinear"])
def test_identity_rotation(device, interpolation_mode):
    """Test that 0-degree rotation preserves the input."""
    torch.manual_seed(0)

    input_shape = (1, 16, 16, 64)
    torch_input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_output = ttnn.rotate(ttnn_input, angle=0.0, interpolation_mode=interpolation_mode)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    if interpolation_mode == "nearest":
        assert torch.equal(torch_input_nhwc, ttnn_output_torch), "Nearest identity rotation should be exact"
    else:
        pcc_passed, pcc_message = assert_with_pcc(torch_input_nhwc, ttnn_output_torch, pcc=0.999)
        logger.info(f"Bilinear identity transform: {pcc_message}")
        max_diff = (torch_input_nhwc - ttnn_output_torch).abs().max().item()
        assert pcc_passed, "Bilinear identity rotation (0°) should preserve input"
        assert max_diff < 0.01, f"Bilinear identity rotation differs too much: {max_diff}"


@pytest.mark.parametrize(
    "angle",
    [0, 15, 30, 45, 60, 90, 135, 180, 270, -30, -90, 360],
)
@pytest.mark.parametrize("interpolation_mode", ["nearest", "bilinear"])
def test_various_angles(device, angle, interpolation_mode):
    """Test various rotation angles."""
    torch.manual_seed(0)

    input_shape = (1, 16, 16, 64)
    torch_input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)

    golden_function = ttnn.get_golden_function(ttnn.rotate)
    torch_output_nhwc = golden_function(torch_input_nhwc, angle=float(angle), interpolation_mode=interpolation_mode)

    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_output = ttnn.rotate(ttnn_input, angle=float(angle), interpolation_mode=interpolation_mode)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    assert ttnn_output_torch.shape == torch_output_nhwc.shape

    atol, rtol = get_rotate_tolerances(input_shape, angle, interpolation_mode)

    if interpolation_mode == "nearest" and angle % 360 == 0:
        assert torch.equal(
            torch_output_nhwc, ttnn_output_torch
        ), f"Identity rotation should be exact for angle {angle}°"
    else:
        comparison_passed = torch.allclose(torch_output_nhwc, ttnn_output_torch, atol=atol, rtol=rtol)
        assert comparison_passed, f"Test failed (mode={interpolation_mode}, angle={angle}°, atol={atol}, rtol={rtol})"


@pytest.mark.parametrize(
    "center",
    [
        (8.0, 8.0),
        (0.0, 0.0),
        (15.0, 15.0),
    ],
)
@pytest.mark.parametrize("interpolation_mode", ["nearest", "bilinear"])
def test_custom_center(device, center, interpolation_mode):
    """Test rotation with custom center points."""
    torch.manual_seed(0)

    input_shape = (1, 16, 16, 64)
    angle = 45.0
    torch_input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)

    golden_function = ttnn.get_golden_function(ttnn.rotate)
    torch_output_nhwc = golden_function(
        torch_input_nhwc, angle=angle, center=center, interpolation_mode=interpolation_mode
    )

    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_output = ttnn.rotate(
        ttnn_input, angle=angle, center=(float(center[0]), float(center[1])), interpolation_mode=interpolation_mode
    )
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    atol, rtol = get_rotate_tolerances(input_shape, angle, interpolation_mode)
    comparison_passed = torch.allclose(torch_output_nhwc, ttnn_output_torch, atol=atol, rtol=rtol)
    assert comparison_passed, f"Custom center rotation failed for center {center}, mode={interpolation_mode}"


@pytest.mark.parametrize("fill", [0.0, 1.0, -1.0, 0.5])
@pytest.mark.parametrize("interpolation_mode", ["nearest", "bilinear"])
def test_custom_fill_values(device, fill, interpolation_mode):
    """Test different fill values for out-of-bounds pixels."""
    torch.manual_seed(0)

    input_shape = (1, 16, 16, 64)
    angle = 45.0
    torch_input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)

    golden_function = ttnn.get_golden_function(ttnn.rotate)
    torch_output_nhwc = golden_function(torch_input_nhwc, angle=angle, fill=fill, interpolation_mode=interpolation_mode)

    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_output = ttnn.rotate(ttnn_input, angle=angle, fill=fill, interpolation_mode=interpolation_mode)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    atol, rtol = get_rotate_tolerances(input_shape, angle, interpolation_mode)
    comparison_passed = torch.allclose(torch_output_nhwc, ttnn_output_torch, atol=atol, rtol=rtol)
    assert comparison_passed, f"Fill value test failed for fill={fill}, mode={interpolation_mode}"


@pytest.mark.parametrize(
    "angle",
    [90, 180, 270, -90, -180],
)
@pytest.mark.parametrize("interpolation_mode", ["nearest", "bilinear"])
def test_exact_multiples_of_90(device, angle, interpolation_mode):
    """Test 90-degree multiples which should be exact or near-exact."""
    torch.manual_seed(0)

    input_shape = (1, 16, 16, 64)
    torch_input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)

    golden_function = ttnn.get_golden_function(ttnn.rotate)
    torch_output_nhwc = golden_function(torch_input_nhwc, angle=float(angle), interpolation_mode=interpolation_mode)

    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_output = ttnn.rotate(ttnn_input, angle=float(angle), interpolation_mode=interpolation_mode)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    if interpolation_mode == "nearest":
        assert torch.equal(torch_output_nhwc, ttnn_output_torch), f"{angle}° rotation should be exact for nearest"
    else:
        pcc_passed, pcc_message = assert_with_pcc(torch_output_nhwc, ttnn_output_torch, pcc=0.995)
        logger.info(f"Bilinear 90° multiple rotation ({angle}°): {pcc_message}")
        max_diff = (torch_output_nhwc - ttnn_output_torch).abs().max().item()
        assert pcc_passed, f"Bilinear 90-degree rotation ({angle}°) failed PCC check"
        assert max_diff < 0.05, f"Bilinear 90-degree rotation differs too much: {max_diff}"


# ============================================================================
# Shape and Size Tests
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
    ],
)
@pytest.mark.parametrize("interpolation_mode", ["nearest", "bilinear"])
def test_various_tensor_sizes(device, input_shape, interpolation_mode):
    """Test various tensor shapes and sizes."""
    torch.manual_seed(0)

    angle = 45.0
    torch_input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)

    golden_function = ttnn.get_golden_function(ttnn.rotate)
    torch_output_nhwc = golden_function(torch_input_nhwc, angle=angle, interpolation_mode=interpolation_mode)

    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_output = ttnn.rotate(ttnn_input, angle=angle, interpolation_mode=interpolation_mode)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    atol, rtol = get_rotate_tolerances(input_shape, angle, interpolation_mode)
    comparison_passed = torch.allclose(torch_output_nhwc, ttnn_output_torch, atol=atol, rtol=rtol)
    assert comparison_passed, f"Tensor size test failed for shape {input_shape}, mode={interpolation_mode}"


@pytest.mark.parametrize("channels", [16, 32, 48, 64, 96, 128])
def test_channel_alignment(device, channels):
    """Test different channel sizes that meet alignment requirements."""
    torch.manual_seed(0)

    input_shape = (1, 32, 32, channels)
    angle = 90.0
    torch_input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)

    golden_function = ttnn.get_golden_function(ttnn.rotate)
    torch_output_nhwc = golden_function(torch_input_nhwc, angle=angle)

    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_output = ttnn.rotate(ttnn_input, angle=angle)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

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

    golden_function = ttnn.get_golden_function(ttnn.rotate)
    torch_output_nhwc = golden_function(torch_input_nhwc, angle=angle)

    ttnn_input = ttnn.from_torch(
        torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=memory_config
    )
    ttnn_output = ttnn.rotate(ttnn_input, angle=angle)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    atol, rtol = get_rotate_tolerances(input_shape, angle, "nearest")
    comparison_passed = torch.allclose(torch_output_nhwc, ttnn_output_torch, atol=atol, rtol=rtol)
    assert comparison_passed, f"Memory config test failed for {memory_config}"


@skip_for_blackhole("Incorrect result on BH github issue #36263")
@pytest.mark.parametrize("interpolation_mode", ["nearest", "bilinear"])
def test_height_sharded_memory(device, interpolation_mode):
    """Test rotation with height and block sharded memory configurations using full grid."""
    torch.manual_seed(0)

    grid_size = device.compute_with_storage_grid_size()
    num_cores_x = grid_size.x
    num_cores_y = grid_size.y
    num_cores = num_cores_x * num_cores_y
    angle = 45.0

    total_sticks = num_cores * 4
    input_shape = (1, total_sticks, 1, 64)
    shard_height = (total_sticks + num_cores - 1) // num_cores
    shard_width = input_shape[3]
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1))}),
        (shard_height, shard_width),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    sharded_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    torch_input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn.rotate)
    torch_output_nhwc = golden_function(torch_input_nhwc, angle=angle, interpolation_mode=interpolation_mode)

    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_input_sharded = ttnn.to_memory_config(ttnn_input, sharded_memory_config)
    ttnn_output = ttnn.rotate(ttnn_input_sharded, angle=angle, interpolation_mode=interpolation_mode)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    atol, rtol = get_rotate_tolerances(input_shape, angle, interpolation_mode)
    comparison_passed = torch.allclose(torch_output_nhwc, ttnn_output_torch, atol=atol, rtol=rtol)
    assert comparison_passed, f"{interpolation_mode} height sharded memory test failed"


# ============================================================================
# Data Type Tests
# ============================================================================


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_data_types(device, dtype):
    """Test different data types."""
    torch.manual_seed(0)

    input_shape = (1, 32, 32, 32)
    angle = 45.0

    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    torch_input_nhwc = torch.randn(input_shape, dtype=torch_dtype)

    golden_function = ttnn.get_golden_function(ttnn.rotate)
    torch_output_nhwc = golden_function(torch_input_nhwc, angle=angle)

    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=dtype)
    ttnn_output = ttnn.rotate(ttnn_input, angle=angle)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    atol, rtol = get_rotate_tolerances(input_shape, angle, "nearest")
    comparison_passed = torch.allclose(torch_output_nhwc, ttnn_output_torch, atol=atol, rtol=rtol)
    assert comparison_passed, f"Data type test failed for {dtype}"


# ============================================================================
# Edge Cases
# ============================================================================


@pytest.mark.parametrize("interpolation_mode", ["nearest", "bilinear"])
def test_full_rotation(device, interpolation_mode):
    """Test 360-degree rotation should be equivalent to 0-degree."""
    torch.manual_seed(0)

    input_shape = (1, 16, 16, 32)
    torch_input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_output_360 = ttnn.rotate(ttnn_input, angle=360.0, interpolation_mode=interpolation_mode)
    ttnn_output_360_torch = ttnn.to_torch(ttnn_output_360)

    ttnn_output_0 = ttnn.rotate(ttnn_input, angle=0.0, interpolation_mode=interpolation_mode)
    ttnn_output_0_torch = ttnn.to_torch(ttnn_output_0)

    assert torch.equal(
        ttnn_output_0_torch, ttnn_output_360_torch
    ), f"360° rotation should be equivalent to 0° for {interpolation_mode}"


@pytest.mark.parametrize("angle", [-45, -90, -180, -270])
def test_negative_angles(device, angle):
    """Test negative rotation angles."""
    torch.manual_seed(0)

    input_shape = (1, 32, 32, 32)
    torch_input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)

    golden_function = ttnn.get_golden_function(ttnn.rotate)
    torch_output_nhwc = golden_function(torch_input_nhwc, angle=float(angle))

    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_output = ttnn.rotate(ttnn_input, angle=float(angle))
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    if angle % 90 == 0:
        assert torch.equal(torch_output_nhwc, ttnn_output_torch), f"{angle}° rotation should be exact"
    else:
        atol, rtol = get_rotate_tolerances(input_shape, angle, "nearest")
        comparison_passed = torch.allclose(torch_output_nhwc, ttnn_output_torch, atol=atol, rtol=rtol)
        assert comparison_passed, f"Negative angle test failed for {angle}°"


@pytest.mark.parametrize("angle", [405.0, 720.0, -450.0])
def test_large_angles(device, angle):
    """Test angles greater than 360 degrees."""
    torch.manual_seed(0)

    input_shape = (1, 32, 32, 32)
    torch_input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)

    golden_function = ttnn.get_golden_function(ttnn.rotate)
    torch_output_nhwc = golden_function(torch_input_nhwc, angle=angle)

    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_output = ttnn.rotate(ttnn_input, angle=angle)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    atol, rtol = get_rotate_tolerances(input_shape, angle, "nearest")
    comparison_passed = torch.allclose(torch_output_nhwc, ttnn_output_torch, atol=atol, rtol=rtol)
    assert comparison_passed, f"Large angle test failed for {angle}°"


def test_small_tensor(device):
    """Test minimal tensor size."""
    torch.manual_seed(0)

    input_shape = (1, 32, 32, 16)
    angle = 45.0
    torch_input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)

    golden_function = ttnn.get_golden_function(ttnn.rotate)
    torch_output_nhwc = golden_function(torch_input_nhwc, angle=angle)

    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_output = ttnn.rotate(ttnn_input, angle=angle)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    atol, rtol = get_rotate_tolerances(input_shape, angle, "nearest")
    comparison_passed = torch.allclose(torch_output_nhwc, ttnn_output_torch, atol=atol, rtol=rtol)
    assert comparison_passed, "Small tensor test failed"


@pytest.mark.parametrize(
    "angle_pair",
    [
        (90, -90),
    ],
)
@pytest.mark.parametrize("interpolation_mode", ["nearest", "bilinear"])
def test_opposite_angles(device, angle_pair, interpolation_mode):
    """Test that rotating by +theta then -theta returns near-original."""
    torch.manual_seed(0)

    input_shape = (1, 16, 16, 32)
    torch_input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)
    angle_pos, angle_neg = angle_pair

    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    ttnn_rotated = ttnn.rotate(ttnn_input, angle=float(angle_pos), interpolation_mode=interpolation_mode)
    ttnn_restored = ttnn.rotate(ttnn_rotated, angle=float(angle_neg), interpolation_mode=interpolation_mode)
    ttnn_restored_torch = ttnn.to_torch(ttnn_restored)

    assert torch.equal(
        torch_input_nhwc, ttnn_restored_torch
    ), f"Round-trip rotation failed for angles {angle_pos}° and {angle_neg}°, mode={interpolation_mode}"


# ============================================================================
# Batch and Channel Consistency Tests
# ============================================================================


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 8, 8, 32),
        (2, 16, 16, 32),
        (4, 16, 16, 32),
    ],
)
@pytest.mark.parametrize("interpolation_mode", ["nearest", "bilinear"])
def test_batch_consistency(device, input_shape, interpolation_mode):
    """Test that rotation processes each batch item consistently."""
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

        if interpolation_mode == "nearest":
            assert torch.equal(
                ttnn_output_torch[b : b + 1], ttnn_single_torch
            ), f"Nearest batch {b} should be equal to single rotation"
        else:
            max_diff = (ttnn_output_torch[b : b + 1] - ttnn_single_torch).abs().max().item()
            logger.info(f"Bilinear batch {b} consistency: max_diff={max_diff:.6f}")
            assert max_diff < 1e-5, f"Bilinear batch {b} differs from single rotation: {max_diff}"


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 8, 8, 32),
        (1, 16, 16, 64),
    ],
)
@pytest.mark.parametrize(
    "angle",
    [10, 45, 90],
)
@pytest.mark.parametrize("interpolation_mode", ["nearest", "bilinear"])
def test_multichannel_consistency(device, input_shape, angle, interpolation_mode):
    """Test that rotation treats all channels identically."""
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

    if interpolation_mode == "nearest":
        assert torch.equal(channel_0, channel_1), f"Nearest channels should be equal after rotation"
    else:
        max_diff = (channel_0 - channel_1).abs().max().item()
        logger.info(f"Bilinear multi-channel consistency (angle={angle}°): max_diff={max_diff:.6f}")
        assert max_diff < 1e-5, f"Bilinear channels rotated differently: {max_diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
