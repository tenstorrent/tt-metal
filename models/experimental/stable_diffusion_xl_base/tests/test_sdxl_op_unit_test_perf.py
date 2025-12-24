# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.perf.device_perf_utils import run_device_perf_detailed


def generate_sdxl_groupnorm_perf_inputs():
    """
    Generate SDXL GroupNorm input shapes with expected performance (in nanoseconds).
    Values are actual measurements from device performance tests.
    """
    # Format: (N, C, H, W, expected_device_kernel_duration_ns, description)
    inputs = [
        # UNet inputs - MEASURED VALUES
        (1, 320, 128, 128, 484824, "UNet-320x128x128"),
        # # VAE inputs - MEASURED VALUES
        (1, 512, 128, 128, 223234, "VAE-512x128x128"),
        # Refiner UNet inputs - MEASURED VALUES
        (1, 384, 128, 128, 480500, "Refiner-384x128x128"),
    ]
    return inputs


@pytest.mark.parametrize(
    "N, C, H, W, expected_duration_ns, description",
    generate_sdxl_groupnorm_perf_inputs(),
    ids=[item[5] for item in generate_sdxl_groupnorm_perf_inputs()],
)
@pytest.mark.models_device_performance_bare_metal
def test_sdxl_group_norm_perf_block_sharded(N, C, H, W, expected_duration_ns, description):
    """
    Test performance of individual GroupNorm operation shapes for SDXL using legacy algorithm.
    This test measures device kernel performance for specific input shapes.
    """

    # Create a command that runs the specific test with exact test name
    test_name = (
        f"test_sdxl_base_group_norm[legacy-input_shape=({N}, {C}, {H}, {W})-device_params={{'l1_small_size': 0}}]"
    )
    command = f'pytest "tests/ttnn/unit_tests/operations/fused/test_group_norm.py::{test_name}" -v'
    subdir = f"sdxl_group_norm_perf_{description}"
    cols = ["DEVICE KERNEL"]
    op_name = "GroupNormDeviceOperation"

    # Run the performance test and get detailed results
    results = run_device_perf_detailed(
        command=command,
        subdir=subdir,
        cols=cols,
        op_name=op_name,
    )

    # Extract the device kernel duration result
    device_kernel_duration = results["DEVICE KERNEL"]["AVG"]

    # Log the performance result
    print(
        f"GroupNorm {description} Device Kernel Duration: {device_kernel_duration:.2f} ns (expected: {expected_duration_ns} ns)"
    )

    # Performance validation with 1.5% margin
    margin = 0.015
    lower_bound = expected_duration_ns * (1 - margin)
    upper_bound = expected_duration_ns * (1 + margin)

    # Performance validation - assert if outside expected range
    assert (
        lower_bound <= device_kernel_duration <= upper_bound
    ), f"Performance outside expected range. Got {device_kernel_duration:.2f} ns, expected {expected_duration_ns} ± 1.5% ({lower_bound:.2f}-{upper_bound:.2f} ns)"


@pytest.mark.models_device_performance_bare_metal
def test_dram_group_norm_perf_welford_reciprocal_vae():
    """
    Test performance of DRAM GroupNorm operation with welford_reciprocal algorithm appearing in VAE.
    This test measures device kernel performance for the specific input shape (1, 256, 1024, 1024, 32, 48, 8, 8).
    """

    # Specific DRAM GroupNorm test case parameters
    N, C, H, W = 1, 256, 1024, 1024
    num_groups, num_out_blocks, cores_y, cores_x = 32, 48, 8, 8
    welford_mode = "welford_reciprocal"

    # Create a command that runs the specific DRAM GroupNorm test
    test_params = f"welford_mode={welford_mode}-N={N}-C={C}-H={H}-W={W}-num_groups={num_groups}-num_out_blocks={num_out_blocks}-cores_y={cores_y}-cores_x={cores_x}-device_params={{'l1_small_size': 0}}"
    command = f'pytest "tests/ttnn/nightly/unit_tests/operations/fused/test_group_norm_DRAM.py::test_group_norm_DRAM[{test_params}]" -v'
    subdir = f"dram_group_norm_perf_{C}x{H}x{W}_{welford_mode}"
    cols = ["DEVICE KERNEL"]
    op_name = "GroupNormDeviceOperation"

    # Run the performance test and get detailed results
    results = run_device_perf_detailed(
        command=command,
        subdir=subdir,
        cols=cols,
        op_name=op_name,
    )

    # Extract the device kernel duration result
    device_kernel_duration = results["DEVICE KERNEL"]["AVG"]

    expected_duration_ns = 19380445

    # Log the performance result
    print(
        f"DRAM GroupNorm {C}x{H}x{W} {welford_mode} Device Kernel Duration: {device_kernel_duration:.2f} ns (expected: {expected_duration_ns} ns)"
    )

    # Performance validation with 1.5% margin
    margin = 0.015
    lower_bound = expected_duration_ns * (1 - margin)
    upper_bound = expected_duration_ns * (1 + margin)

    # Performance validation - assert if outside expected range
    assert (
        lower_bound <= device_kernel_duration <= upper_bound
    ), f"Performance outside expected range. Got {device_kernel_duration:.2f} ns, expected {expected_duration_ns} ± 1.5% ({lower_bound:.2f}-{upper_bound:.2f} ns)"


def generate_sdxl_groupnorm_negative_mask_perf_inputs():
    """
    Generate SDXL GroupNorm negative mask input shapes with expected performance (in nanoseconds).
    Values are actual measurements from device performance tests.
    """
    # Format: (N, C, H, W, expected_device_kernel_duration_ns, description)
    inputs = [
        # Negative mask test cases
        (1, 640, 128, 128, 600429, "NegativeMask-640x128x128"),
    ]
    return inputs


@pytest.mark.parametrize(
    "N, C, H, W, expected_duration_ns, description",
    generate_sdxl_groupnorm_negative_mask_perf_inputs(),
    ids=[item[5] for item in generate_sdxl_groupnorm_negative_mask_perf_inputs()],
)
@pytest.mark.models_device_performance_bare_metal
def test_sdxl_group_norm_perf_negative_mask(N, C, H, W, expected_duration_ns, description):
    """
    Test performance of individual GroupNorm negative mask operation shapes for SDXL.
    This test measures device kernel performance for specific input shapes with negative mask.
    """

    # Create a command that runs the specific test with exact test name
    test_name = f"test_sdxl_base_group_norm_negative_mask[input_shape=({N}, {C}, {H}, {W})-device_params={{'l1_small_size': 47000}}]"
    command = f'pytest "tests/ttnn/unit_tests/operations/fused/test_group_norm.py::{test_name}" -v'
    subdir = f"sdxl_group_norm_negative_mask_perf_{description}"
    cols = ["DEVICE KERNEL"]
    op_name = "GroupNormDeviceOperation"

    # Run the performance test and get detailed results
    results = run_device_perf_detailed(
        command=command,
        subdir=subdir,
        cols=cols,
        op_name=op_name,
    )

    # Extract the device kernel duration result
    device_kernel_duration = results["DEVICE KERNEL"]["AVG"]

    # Log the performance result
    print(
        f"GroupNorm NegativeMask {description} Device Kernel Duration: {device_kernel_duration:.2f} ns (expected: {expected_duration_ns} ns)"
    )

    # Performance validation with 1.5% margin
    margin = 0.015
    lower_bound = expected_duration_ns * (1 - margin)
    upper_bound = expected_duration_ns * (1 + margin)

    # Performance validation - assert if outside expected range
    assert (
        lower_bound <= device_kernel_duration <= upper_bound
    ), f"Performance outside expected range. Got {device_kernel_duration:.2f} ns, expected {expected_duration_ns} ± 1.5% ({lower_bound:.2f}-{upper_bound:.2f} ns)"


@pytest.mark.models_device_performance_bare_metal
def test_dram_group_norm_perf_welford_reciprocal():
    """
    Test performance of DRAM GroupNorm operation with welford_reciprocal algorithm.
    This test measures device kernel performance for the specific input shape (1, 256, 1024, 1024, 32, 48, 8, 8).
    """

    # Specific DRAM GroupNorm test case parameters
    N, C, H, W = 1, 256, 1024, 1024
    num_groups, num_out_blocks, cores_y, cores_x = 32, 48, 8, 8
    welford_mode = "welford_reciprocal"

    # Create a command that runs the specific DRAM GroupNorm test
    test_params = f"welford_mode={welford_mode}-N={N}-C={C}-H={H}-W={W}-num_groups={num_groups}-num_out_blocks={num_out_blocks}-cores_y={cores_y}-cores_x={cores_x}-device_params={{'l1_small_size': 0}}"
    command = f'pytest "tests/ttnn/nightly/unit_tests/operations/fused/test_group_norm_DRAM.py::test_group_norm_DRAM[{test_params}]" -v'
    subdir = f"dram_group_norm_perf_{C}x{H}x{W}_{welford_mode}"
    cols = ["DEVICE KERNEL"]
    op_name = "GroupNormDeviceOperation"

    # Run the performance test and get detailed results
    results = run_device_perf_detailed(
        command=command,
        subdir=subdir,
        cols=cols,
        op_name=op_name,
    )

    # Extract the device kernel duration result
    device_kernel_duration = results["DEVICE KERNEL"]["AVG"]

    # Expected performance value (actual measurement)
    expected_duration_ns = 19380445  # Measured: 19.38ms for (1,256,1024,1024) welford_reciprocal

    # Log the performance result
    print(
        f"DRAM GroupNorm {C}x{H}x{W} {welford_mode} Device Kernel Duration: {device_kernel_duration:.2f} ns (expected: {expected_duration_ns} ns)"
    )

    # Performance validation with 1.5% margin
    margin = 0.015
    lower_bound = expected_duration_ns * (1 - margin)
    upper_bound = expected_duration_ns * (1 + margin)

    # Performance validation - assert if outside expected range
    assert (
        lower_bound <= device_kernel_duration <= upper_bound
    ), f"Performance outside expected range. Got {device_kernel_duration:.2f} ns, expected {expected_duration_ns} ± 1.5% ({lower_bound:.2f}-{upper_bound:.2f} ns)"
