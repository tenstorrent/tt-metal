# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time
import torch
import pytest
from loguru import logger

import ttnn
from models.perf.perf_utils import prep_perf_report
from models.common.utility_functions import run_for_wormhole_b0
from models.experimental.vovnet.runner.performant_runner import VovnetPerformantRunner

from models.experimental.vovnet.common import VOVNET_L1_SMALL_SIZE


def get_expected_times(name):
    # issue #35265 - treshold needs to be confirmed
    base = {"vovnet": (172, 0.0085)}
    return base[name]


def run_e2e_performant(
    device, device_batch_size, act_dtype, weight_dtype, model_location_generator, resolution, channels=3
):
    total_batch_size = device_batch_size * device.get_num_devices()
    performant_runner = VovnetPerformantRunner(
        device,
        device_batch_size,
        act_dtype,
        weight_dtype,
        resolution=resolution,
        model_location_generator=model_location_generator,
    )
    performant_runner._capture_vovnet_trace_2cqs()
    inference_times = []
    torch_input_tensor = torch.randn(total_batch_size, channels, resolution[0], resolution[1])
    iterations_count = 10
    ttnn.synchronize_device(device)

    t0 = time.time()

    for _ in range(iterations_count):
        _ = performant_runner.run(torch_input_tensor)
    ttnn.synchronize_device(device)
    t1 = time.time()

    performant_runner.release()
    inference_time_avg = round((t1 - t0) / 10, 6)

    logger.info(
        f"ttnn_vovnet_batch_size: {total_batch_size}, resolution: {resolution}. One inference iteration time (sec): {inference_time_avg}, FPS: {round(total_batch_size/inference_time_avg)}"
    )
    expected_compile_time, expected_inference_time = get_expected_times("vovnet")
    prep_perf_report(
        model_name="models/experimental/vovnet",
        batch_size=total_batch_size,
        inference_and_compile_time=inference_time_avg,
        inference_time=inference_time_avg,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments="",
        inference_time_cpu=0.0,
    )


@pytest.mark.parametrize(
    "device_batch_size, act_dtype, weight_dtype",
    ((1, ttnn.bfloat8_b, ttnn.bfloat8_b),),
)
@pytest.mark.parametrize(
    "resolution",
    [
        (224, 224),
    ],
)
@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": VOVNET_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
def test_vovnet_e2e_performant(
    device,
    device_batch_size,
    act_dtype,
    weight_dtype,
    model_location_generator,
    resolution,
):
    return run_e2e_performant(device, device_batch_size, act_dtype, weight_dtype, model_location_generator, resolution)


@pytest.mark.parametrize(
    "device_batch_size, act_dtype, weight_dtype",
    ((1, ttnn.bfloat8_b, ttnn.bfloat8_b),),
)
@pytest.mark.parametrize(
    "resolution",
    [
        (224, 224),
    ],
)
@pytest.mark.models_performance_bare_metal
@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": VOVNET_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
def test_vovnet_e2e_performant_dp(
    mesh_device,
    device_batch_size,
    act_dtype,
    weight_dtype,
    model_location_generator,
    resolution,
):
    return run_e2e_performant(
        mesh_device, device_batch_size, act_dtype, weight_dtype, model_location_generator, resolution
    )
