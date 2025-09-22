# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time
import pytest

import numpy as np

from loguru import logger

from ttnn.device import is_wormhole_b0

from models.perf.perf_utils import prep_perf_report
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report

from models.experimental.functional_unet.tests.common import UNET_TRACE_REGION_SIZE, UNET_L1_SMALL_REGION_SIZE
from models.experimental.functional_unet.tests.test_unet_model import run_unet_model

UNET_DEVICE_TEST_TOTAL_ITERATIONS = 4


@pytest.mark.parametrize("batch", [1])
@pytest.mark.parametrize("groups", [4])
@pytest.mark.parametrize("iterations", [UNET_DEVICE_TEST_TOTAL_ITERATIONS])
@pytest.mark.parametrize("device_params", [{"l1_small_size": UNET_L1_SMALL_REGION_SIZE}], indirect=True)
def test_unet_model(batch, groups, device, iterations, reset_seeds):
    if (
        not is_wormhole_b0(device)
        and device.compute_with_storage_grid_size().x * device.compute_with_storage_grid_size().y != 110
        and device.compute_with_storage_grid_size().x * device.compute_with_storage_grid_size().y != 130
    ):
        pytest.skip(f"Shallow UNet only support 110 or 130 cores on BH (was {device.compute_with_storage_grid_size()})")
    device.disable_and_clear_program_cache()  # Needed to give consistent device perf between iterations
    run_unet_model(batch, groups, device, iterations)


def filter_outliers(measurements, trim_percentage=0.2):
    """Remove outliers using trimmed mean filtering."""
    if len(measurements) < 3:
        return measurements

    # Sort measurements
    sorted_measurements = sorted(measurements)
    n = len(sorted_measurements)

    # Calculate number of elements to trim from each end
    trim_count = int(n * trim_percentage / 2)

    # If trim_count would leave less than 1 element, don't trim
    if n - 2 * trim_count < 1:
        return measurements

    filtered = sorted_measurements[trim_count : n - trim_count]

    logger.info(
        f"Trimmed {len(measurements) - len(filtered)} outliers from {len(measurements)} measurements (trim_percentage={trim_percentage})"
    )
    return filtered


@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "batch, groups, expected_device_perf_fps",
    ((1, 4, 1633.0) if is_wormhole_b0() else (1, 4, 2869.0),),
)
def test_unet_perf_device(batch: int, groups: int, expected_device_perf_fps: float):
    command = f"pytest models/experimental/functional_unet/tests/test_unet_perf.py::test_unet_model"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    total_batch = groups * batch * UNET_DEVICE_TEST_TOTAL_ITERATIONS

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    post_processed_results = run_device_perf(
        command, subdir="unet_shallow", num_iterations=1, cols=cols, batch_size=total_batch
    )
    expected_perf_cols = {inference_time_key: expected_device_perf_fps}
    expected_results = check_device_perf(
        post_processed_results, margin=0.005, expected_perf_cols=expected_perf_cols, assert_on_fail=True
    )
    prep_device_perf_report(
        model_name=f"unet-shallow_batch-{batch}_groups-{groups}",
        batch_size=total_batch,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": UNET_L1_SMALL_REGION_SIZE,
            "trace_region_size": UNET_TRACE_REGION_SIZE,
            "num_command_queues": 2,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch, groups, iterations, expected_compile_time, expected_throughput",
    ((1, 4, 256, 30.0, 1400.0),),
)
def test_unet_trace_perf(
    batch: int,
    groups: int,
    iterations: int,
    expected_compile_time: float,
    expected_throughput: float,
    device,
    reset_seeds,
):
    if (
        not is_wormhole_b0(device)
        and device.compute_with_storage_grid_size().x * device.compute_with_storage_grid_size().y != 110
        and device.compute_with_storage_grid_size().x * device.compute_with_storage_grid_size().y != 130
    ):
        pytest.skip(f"Shallow UNet only support 110 or 130 cores on BH (was {device.compute_with_storage_grid_size()})")

    from models.experimental.functional_unet.tests.test_unet_trace import (
        test_unet_trace_2cq,
    )

    logger.info(f"Invoking underlying model test for {iterations} iterations...")
    result = test_unet_trace_2cq(batch, groups, iterations, device, reset_seeds)

    total_num_samples = result.batch * result.groups * result.num_devices
    expected_inference_time = total_num_samples / expected_throughput
    prep_perf_report(
        model_name="unet_shallow-trace_2cq_same_io",
        batch_size=total_num_samples,
        inference_and_compile_time=result.inference_and_compile_time,
        inference_time=result.inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=f"batch_{result.batch}-groups_{result.groups}-num_devices_{result.num_devices}",
    )
    assert (
        result.get_fps() >= expected_throughput
    ), f"Expected end-to-end performance to exceed {expected_throughput:.2f} fps but was {result.get_fps():.2f} fps"


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": UNET_L1_SMALL_REGION_SIZE,
            "trace_region_size": UNET_TRACE_REGION_SIZE,
            "num_command_queues": 2,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch, groups, num_runs, iterations, expected_compile_time, expected_throughput", ((1, 4, 12, 256, 30.0, 2660.0),)
)
def test_unet_trace_perf_multi_device(
    batch: int,
    groups: int,
    num_runs: int,
    iterations: int,
    expected_compile_time: float,
    expected_throughput: float,
    mesh_device,
    reset_seeds,
):
    from models.experimental.functional_unet.tests.test_unet_trace import (
        test_unet_trace_2cq_multi_device,
    )

    model_name = "unet_shallow-trace_2cq_same_io-multi_device"

    fps_results = []
    inference_times = []
    compile_times = []
    for run_idx in range(num_runs):
        logger.info(f"Measurement run {run_idx + 1}/{num_runs}...")
        result = test_unet_trace_2cq_multi_device(batch, groups, iterations, mesh_device, reset_seeds)

        fps_results.append(result.get_fps())
        inference_times.append(result.inference_time)
        compile_times.append(result.inference_and_compile_time - result.inference_time)

        logger.info(f"Run {run_idx + 1} throughput: {result.get_fps():.2f} fps")

        if run_idx < num_runs - 1:
            time.sleep(1.0)

    filtered_fps = filter_outliers(fps_results)
    final_fps = np.median(filtered_fps) if len(filtered_fps) >= 2 else np.mean(fps_results)
    fps_std = np.std(filtered_fps) if len(filtered_fps) >= 2 else np.std(fps_results)

    final_inference_time = np.median(inference_times)
    final_compile_time = np.median(compile_times)

    logger.info(f"FPS results: {[f'{x:.1f}' for x in fps_results]}")
    logger.info(f"Filtered FPS: {[f'{x:.1f}' for x in filtered_fps]}")
    logger.info(f"Median FPS: {final_fps:.2f} ± {fps_std:.2f}")
    logger.info(f"Median inference time: {final_inference_time:.4f}s")
    logger.info(f"Median compile time: {final_compile_time:.2f}s")
    logger.info(f"Variance reduction: {((np.std(fps_results) - fps_std) / np.std(fps_results) * 100):.1f}%")

    total_num_samples = result.batch * result.groups * result.num_devices
    expected_inference_time = total_num_samples / expected_throughput

    prep_perf_report(
        model_name=model_name,
        batch_size=total_num_samples,
        inference_and_compile_time=final_inference_time + final_compile_time,
        inference_time=final_inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=f"batch_{result.batch}-groups_{result.groups}-num_devices_{result.num_devices}",
    )

    confidence_margin = 2 * fps_std  # 95% confidence interval
    performance_threshold = expected_throughput - confidence_margin

    assert (
        final_fps >= performance_threshold
    ), f"Expected performance {expected_throughput:.2f} ± {confidence_margin:.2f} fps but got {final_fps:.2f} ± {fps_std:.2f} fps"
    logger.success(f"Test passed: {final_fps:.2f} ± {fps_std:.2f} fps")
