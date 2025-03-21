# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

from loguru import logger

from models.perf.perf_utils import prep_perf_report
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report
from models.utility_functions import (
    skip_for_grayskull,
)


@skip_for_grayskull("UNet not currently supported on GS")
@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "batch, groups, expected_device_perf_fps",
    ((1, 2, 1035.0),),
)
def test_unet_perf_device(batch: int, groups: int, expected_device_perf_fps: float):
    command = f"pytest models/experimental/functional_unet/tests/test_unet_model.py::test_unet_model[device_params0-{groups}-{batch}]"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    total_batch = groups * batch

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    post_processed_results = run_device_perf(
        command, subdir="unet_shallow", num_iterations=3, cols=cols, batch_size=total_batch
    )
    expected_perf_cols = {inference_time_key: expected_device_perf_fps}
    expected_results = check_device_perf(
        post_processed_results, margin=0.02, expected_perf_cols=expected_perf_cols, assert_on_fail=True
    )
    prep_device_perf_report(
        model_name=f"unet-shallow_batch-{batch}_groups-{groups}",
        batch_size=total_batch,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )


@skip_for_grayskull("UNet not currently supported on GS")
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 68864, "trace_region_size": 424960, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch, groups, iterations, expected_compile_time, expected_throughput",
    ((1, 2, 128, 25.0, 830.0),),
)
def test_unet_trace_perf(
    batch: int,
    groups: int,
    iterations: int,
    expected_compile_time: float,
    expected_throughput: float,
    device,
    use_program_cache,
    reset_seeds,
):
    from models.experimental.functional_unet.tests.test_unet_trace import (
        test_unet_trace_2cq_same_io,
    )

    logger.info(f"Invoking underlying model test for {iterations} iterations...")
    result = test_unet_trace_2cq_same_io(batch, groups, iterations, device, use_program_cache, reset_seeds)

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


@skip_for_grayskull("UNet not currently supported on GS")
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 68864, "trace_region_size": 424960, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch, groups, iterations, expected_compile_time, expected_throughput, use_async_mode",
    (
        (1, 2, 128, 25.0, 1450.0, True),
        (1, 2, 128, 25.0, 1650.0, False),
    ),
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (1, 2),
    ],
    indirect=True,
)
def test_unet_trace_perf_multi_device(
    batch: int,
    groups: int,
    iterations: int,
    expected_compile_time: float,
    expected_throughput: float,
    use_async_mode: bool,
    mesh_device,
    use_program_cache,
    reset_seeds,
):
    from models.experimental.functional_unet.tests.test_unet_trace import (
        test_unet_trace_2cq_same_io_multi_device,
    )

    mesh_device.enable_async(use_async_mode)
    model_name = "unet_shallow-trace_2cq_same_io-multi_device"
    model_name += "-async" if use_async_mode else "-no_async"

    logger.info(f"Invoking underlying model test for {iterations} iterations...")
    result = test_unet_trace_2cq_same_io_multi_device(
        batch, groups, iterations, mesh_device, use_async_mode, use_program_cache, reset_seeds
    )

    total_num_samples = result.batch * result.groups * result.num_devices
    expected_inference_time = total_num_samples / expected_throughput
    prep_perf_report(
        model_name=model_name,
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
