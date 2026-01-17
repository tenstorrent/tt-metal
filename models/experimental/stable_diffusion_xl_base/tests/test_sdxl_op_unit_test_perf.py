# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from models.perf.device_perf_utils import run_device_perf_detailed

MARGIN = 0.015
USE_PERF_TEST_MODE = True


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
def test_dram_group_norm_welford_reciprocal_vae(device):
    from tests.ttnn.unit_tests.operations.fused.test_group_norm_DRAM import test_group_norm_DRAM

    test_group_norm_DRAM(device, 1, 256, 256, 256, 32, 4, 8, 8, "welford_reciprocal", perf_test_mode=USE_PERF_TEST_MODE)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
def test_block_sharded_group_norm_sdxl(device):
    from tests.ttnn.unit_tests.operations.fused.test_group_norm import test_sdxl_base_group_norm

    test_sdxl_base_group_norm(device, (1, 1920, 32, 32), use_welford=False, perf_test_mode=USE_PERF_TEST_MODE)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 47000}], indirect=True)
def test_block_sharded_group_norm_negative_mask_sdxl(device):
    from tests.ttnn.unit_tests.operations.fused.test_group_norm import test_sdxl_base_group_norm_negative_mask

    test_sdxl_base_group_norm_negative_mask(device, (1, 640, 128, 128), perf_test_mode=USE_PERF_TEST_MODE)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
def test_ff_matmul_with_gelu_sdxl(device):
    from tests.ttnn.nightly.unit_tests.operations.matmul.test_matmul import test_sdxl_matmul

    test_sdxl_matmul(
        device,
        core_grid=ttnn.CoreGrid(y=8, x=5),
        M=1024,
        K=1280,
        N=5120,
        in0_block_w=4,
        out_subblock_h=1,
        out_subblock_w=8,
        per_core_M=4,
        per_core_N=32,
        has_gelu=True,
        perf_test_mode=USE_PERF_TEST_MODE,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 2 * 16384}], indirect=True)
def test_conv2d_block_sharded_sdxl(device):
    from tests.ttnn.nightly.unit_tests.operations.conv.test_conv2d import test_conv2d_sdxl

    test_conv2d_sdxl(
        device,
        torch_tensor_map={},
        batch=1,
        input_channels=2560,
        output_channels=1280,
        input_height=32,
        input_width=32,
        weights_dtype=ttnn.bfloat8_b,
        output_dtype=ttnn.bfloat16,
        groups=1,
        kernel=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
        shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        act_block_h_override=64,
        act_block_w_div=1,
        deallocate_activation=True,
        math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_accum=False,
        packer_l1_acc=True,
        act_db=True,
        w_db=True,
        perf_test_mode=USE_PERF_TEST_MODE,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 27 * 1024}], indirect=True)
def test_conv2d_auto_sliced_vae(device):
    from tests.ttnn.nightly.unit_tests.operations.conv.test_conv2d import test_conv2d_vae_sdxl

    test_conv2d_vae_sdxl(
        device,
        torch_tensor_map={},
        batch=1,
        input_channels=512,
        output_channels=512,
        input_height=256,
        input_width=256,
        weights_dtype=ttnn.bfloat8_b,
        output_dtype=ttnn.bfloat16,
        groups=1,
        kernel=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
        shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        deallocate_activation=False,
        slice_type=ttnn.Conv2dDRAMSliceWidth,
        num_slices=2,
        act_block_h_override=256,
        throttle=0,
        auto_slice=True,
        perf_test_mode=USE_PERF_TEST_MODE,
    )


@pytest.mark.models_device_performance_bare_metal
def test_dram_group_norm_vae_welford_reciprocal_performance():
    # Create a command that runs the specific test
    command = f'pytest "models/experimental/stable_diffusion_xl_base/tests/test_sdxl_op_unit_test_perf.py::test_dram_group_norm_welford_reciprocal_vae" -v'
    subdir = f"dram_group_norm_vae_welford_reciprocal_perf"
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

    expected_duration_ns = 1516464  # Measured: 1.52ms for GroupNorm VAE welford_reciprocal

    # Log the performance result
    print(
        f"DRAM GroupNorm VAE welford_reciprocal Device Kernel Duration: {device_kernel_duration:.2f} ns (expected: {expected_duration_ns} ns)"
    )

    # Performance validation with 1.5% margin
    lower_bound = expected_duration_ns * (1 - MARGIN)
    upper_bound = expected_duration_ns * (1 + MARGIN)

    # Performance validation - assert if outside expected range
    assert (
        lower_bound <= device_kernel_duration <= upper_bound
    ), f"Performance outside expected range. Got {device_kernel_duration:.2f} ns, expected {expected_duration_ns} ± {MARGIN * 100}% ({lower_bound:.2f}-{upper_bound:.2f} ns)"


@pytest.mark.models_device_performance_bare_metal
def test_block_sharded_group_norm_sdxl_performance():
    # Create a command that runs the specific test
    command = f'pytest "models/experimental/stable_diffusion_xl_base/tests/test_sdxl_op_unit_test_perf.py::test_block_sharded_group_norm_sdxl" -v'
    subdir = f"block_sharded_group_norm_sdxl_perf"
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

    expected_duration_ns = 83180  # Measured: ~83μs for GroupNorm SDXL block sharded

    # Log the performance result
    print(
        f"Block Sharded GroupNorm SDXL Device Kernel Duration: {device_kernel_duration:.2f} ns (expected: {expected_duration_ns} ns)"
    )

    # Performance validation with 1.5% margin
    lower_bound = expected_duration_ns * (1 - MARGIN)
    upper_bound = expected_duration_ns * (1 + MARGIN)

    # Performance validation - assert if outside expected range
    assert (
        lower_bound <= device_kernel_duration <= upper_bound
    ), f"Performance outside expected range. Got {device_kernel_duration:.2f} ns, expected {expected_duration_ns} ± {MARGIN * 100}% ({lower_bound:.2f}-{upper_bound:.2f} ns)"


@pytest.mark.models_device_performance_bare_metal
def test_block_sharded_group_norm_negative_mask_sdxl_performance():
    # Create a command that runs the specific test
    command = f'pytest "models/experimental/stable_diffusion_xl_base/tests/test_sdxl_op_unit_test_perf.py::test_block_sharded_group_norm_negative_mask_sdxl" -v'
    subdir = f"block_sharded_group_norm_negative_mask_sdxl_perf"
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

    expected_duration_ns = 600338  # Measured: ~600μs for GroupNorm SDXL negative mask

    # Log the performance result
    print(
        f"Block Sharded GroupNorm Negative Mask SDXL Device Kernel Duration: {device_kernel_duration:.2f} ns (expected: {expected_duration_ns} ns)"
    )

    # Performance validation with 1.5% margin
    lower_bound = expected_duration_ns * (1 - MARGIN)
    upper_bound = expected_duration_ns * (1 + MARGIN)

    # Performance validation - assert if outside expected range
    assert (
        lower_bound <= device_kernel_duration <= upper_bound
    ), f"Performance outside expected range. Got {device_kernel_duration:.2f} ns, expected {expected_duration_ns} ± {MARGIN * 100}% ({lower_bound:.2f}-{upper_bound:.2f} ns)"


@pytest.mark.models_device_performance_bare_metal
def test_ff_matmul_with_gelu_sdxl_performance():
    # Create a command that runs the specific test
    command = f'pytest "models/experimental/stable_diffusion_xl_base/tests/test_sdxl_op_unit_test_perf.py::test_ff_matmul_with_gelu_sdxl" -v'
    subdir = f"ff_matmul_with_gelu_sdxl_perf"
    cols = ["DEVICE KERNEL"]
    op_name = "MatmulDeviceOperation"

    # Run the performance test and get detailed results
    results = run_device_perf_detailed(
        command=command,
        subdir=subdir,
        cols=cols,
        op_name=op_name,
    )

    # Extract the device kernel duration result
    device_kernel_duration = results["DEVICE KERNEL"]["AVG"]

    expected_duration_ns = 238419  # Measured: 238μs for FF Matmul SDXL with GELU

    # Log the performance result
    print(
        f"FF Matmul SDXL (with GELU) Device Kernel Duration: {device_kernel_duration:.2f} ns (expected: {expected_duration_ns} ns)"
    )

    # Performance validation with 1.5% margin
    lower_bound = expected_duration_ns * (1 - MARGIN)
    upper_bound = expected_duration_ns * (1 + MARGIN)

    # Performance validation - assert if outside expected range
    assert (
        lower_bound <= device_kernel_duration <= upper_bound
    ), f"Performance outside expected range. Got {device_kernel_duration:.2f} ns, expected {expected_duration_ns} ± {MARGIN * 100}% ({lower_bound:.2f}-{upper_bound:.2f} ns)"


@pytest.mark.models_device_performance_bare_metal
def test_conv2d_block_sharded_sdxl_performance():
    # Create a command that runs the specific test
    command = f'pytest "models/experimental/stable_diffusion_xl_base/tests/test_sdxl_op_unit_test_perf.py::test_conv2d_block_sharded_sdxl" -v'
    subdir = f"conv2d_block_sharded_sdxl_perf"
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

    expected_duration_ns = 1115831  # Measured: 1.12ms for Conv2D SDXL block sharded

    # Log the performance result
    print(
        f"Conv2D Block Sharded SDXL Device Kernel Duration: {device_kernel_duration:.2f} ns (expected: {expected_duration_ns} ns)"
    )

    # Performance validation with 1.5% margin
    lower_bound = expected_duration_ns * (1 - MARGIN)
    upper_bound = expected_duration_ns * (1 + MARGIN)

    # Performance validation - assert if outside expected range
    assert (
        lower_bound <= device_kernel_duration <= upper_bound
    ), f"Performance outside expected range. Got {device_kernel_duration:.2f} ns, expected {expected_duration_ns} ± {MARGIN * 100}% ({lower_bound:.2f}-{upper_bound:.2f} ns)"


@pytest.mark.models_device_performance_bare_metal
def test_conv2d_auto_sliced_vae_performance():
    # Create a command that runs the specific test
    command = f'pytest "models/experimental/stable_diffusion_xl_base/tests/test_sdxl_op_unit_test_perf.py::test_conv2d_auto_sliced_vae" -v'
    subdir = f"conv2d_auto_sliced_vae_perf"
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

    expected_duration_ns = 3185244  # Measured: 3.19ms for Conv2D VAE auto sliced

    # Log the performance result
    print(
        f"Conv2D Auto Sliced VAE Device Kernel Duration: {device_kernel_duration:.2f} ns (expected: {expected_duration_ns} ns)"
    )

    # Performance validation with 1.5% margin
    lower_bound = expected_duration_ns * (1 - MARGIN)
    upper_bound = expected_duration_ns * (1 + MARGIN)

    # Performance validation - assert if outside expected range
    assert (
        lower_bound <= device_kernel_duration <= upper_bound
    ), f"Performance outside expected range. Got {device_kernel_duration:.2f} ns, expected {expected_duration_ns} ± {MARGIN * 100}% ({lower_bound:.2f}-{upper_bound:.2f} ns)"
