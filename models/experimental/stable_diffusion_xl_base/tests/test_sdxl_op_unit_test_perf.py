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
def test_conv2d_sdxl_perf():
    """
    Test performance of SDXL Conv2D operation.
    This test measures device kernel performance for the specific input configuration:
    (1, 2560, 1280, 32, 32) with 3x3 kernel, stride (1,1), padding (1,1).
    """

    # Specific SDXL Conv2D test case parameters
    batch, input_channels, output_channels = 1, 2560, 1280
    input_height, input_width = 32, 32

    # Create a command that runs the specific Conv2D SDXL test
    test_name = "test_conv2d_sdxl[device_params={'l1_small_size': 32768}-batch=1-input_channels=2560-output_channels=1280-input_height=32-input_width=32-weights_dtype=DataType.BFLOAT8_B-output_dtype=DataType.BFLOAT16-groups=1-kernel=(3, 3)-stride=(1, 1)-padding=(1, 1)-dilation=(1, 1)-shard_layout=TensorMemoryLayout.BLOCK_SHARDED-act_block_h_override=64-act_block_w_div=1-deallocate_activation=True-math_fidelity=MathFidelity.HiFi2-fp32_accum=False-packer_l1_acc=True-act_db=True-w_db=True]"
    command = f'pytest "tests/ttnn/nightly/unit_tests/operations/conv/test_conv2d.py::{test_name}" -v'
    subdir = f"conv2d_sdxl_perf_{input_channels}x{output_channels}_{input_height}x{input_width}"
    cols = ["DEVICE KERNEL"]
    op_name = "Conv2dDeviceOperation"

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
    expected_duration_ns = 1115488  # Measured: 1.12ms for Conv2D SDXL (2560->1280, 32x32)

    # Log the performance result
    print(
        f"Conv2D SDXL {input_channels}->{output_channels} {input_height}x{input_width} Device Kernel Duration: {device_kernel_duration:.2f} ns (expected: {expected_duration_ns} ns)"
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
def test_matmul_sdxl_perf_no_gelu():
    """
    Test performance of SDXL Matmul operation without GELU activation.
    This test measures device kernel performance for the configuration:
    M=1024, K=5120, N=1280 (1024x5120x1280).
    """

    # Specific SDXL Matmul test case parameters (no GELU)
    M, K, N = 1024, 5120, 1280

    # Create a command that runs the specific Matmul SDXL test
    test_name = "test_sdxl_matmul[M=1024-K=5120-N=1280-in0_block_w=4-out_subblock_h=1-out_subblock_w=8-per_core_M=4-per_core_N=8-has_gelu=False-core_grid=ttnn.CoreGrid(x=5, y=8)]"
    command = f'pytest "tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul.py::{test_name}" -v'
    subdir = f"matmul_sdxl_perf_{M}x{K}x{N}_no_gelu"
    cols = ["DEVICE KERNEL"]
    op_name = "Matmul"

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
    expected_duration_ns = 209173  # Measured: 209μs for Matmul SDXL (1024x5120x1280, no GELU)

    # Log the performance result
    print(
        f"Matmul SDXL {M}x{K}x{N} (no GELU) Device Kernel Duration: {device_kernel_duration:.2f} ns (expected: {expected_duration_ns} ns)"
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
def test_matmul_sdxl_perf_with_gelu():
    """
    Test performance of SDXL Matmul operation with GELU activation.
    This test measures device kernel performance for the configuration:
    M=1024, K=1280, N=5120 (1024x1280x5120) with GELU.
    """

    # Specific SDXL Matmul test case parameters (with GELU)
    M, K, N = 1024, 1280, 5120

    # Create a command that runs the specific Matmul SDXL test
    test_name = "test_sdxl_matmul[M=1024-K=1280-N=5120-in0_block_w=4-out_subblock_h=1-out_subblock_w=8-per_core_M=4-per_core_N=32-has_gelu=True-core_grid=ttnn.CoreGrid(x=5, y=8)]"
    command = f'pytest "tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul.py::{test_name}" -v'
    subdir = f"matmul_sdxl_perf_{M}x{K}x{N}_with_gelu"
    cols = ["DEVICE KERNEL"]
    op_name = "Matmul"

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
    expected_duration_ns = 238534  # Measured: 238μs for Matmul SDXL (1024x1280x5120, with GELU)

    # Log the performance result
    print(
        f"Matmul SDXL {M}x{K}x{N} (with GELU) Device Kernel Duration: {device_kernel_duration:.2f} ns (expected: {expected_duration_ns} ns)"
    )

    # Performance validation with 1.5% margin
    margin = 0.015
    lower_bound = expected_duration_ns * (1 - margin)
    upper_bound = expected_duration_ns * (1 + margin)

    # Performance validation - assert if outside expected range
    assert (
        lower_bound <= device_kernel_duration <= upper_bound
    ), f"Performance outside expected range. Got {device_kernel_duration:.2f} ns, expected {expected_duration_ns} ± 1.5% ({lower_bound:.2f}-{upper_bound:.2f} ns)"
