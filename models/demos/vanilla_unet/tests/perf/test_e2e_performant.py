# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_wormhole_b0
from models.demos.vanilla_unet.common import VANILLA_UNET_L1_SMALL_SIZE
from models.demos.vanilla_unet.runner.performant_runner import VanillaUNetPerformantRunner


def get_expected_times(name):
    base = {"ttnn_vanilla_unet": (0.013)}
    return base[name]


def run_e2e_performant(
    device, device_batch_size, act_dtype, weight_dtype, model_location_generator, resolution, channels=3
):
    total_batch_size = device_batch_size * device.get_num_devices()
    performant_runner = VanillaUNetPerformantRunner(
        device,
        device_batch_size,
        act_dtype,
        weight_dtype,
        resolution=resolution,
        model_location_generator=model_location_generator,
    )
    iterations_count = 10
    inference_times = []
    for i in range(iterations_count):
        input_shape = (total_batch_size, channels, *resolution)
        torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)

        t0 = time.time()
        _ = performant_runner.run(torch_input_tensor)
        if i + 1 == iterations_count:
            ttnn.synchronize_device(device)
        t1 = time.time()
        inference_times.append(t1 - t0)
    performant_runner.release()

    inference_time_avg = round(sum(inference_times) / len(inference_times), 6)

    tolerance = 0.03
    expected_inference_time = get_expected_times("ttnn_vanilla_unet")

    assert abs(inference_time_avg - expected_inference_time) <= tolerance, (
        f"Inference time regression detected! "
        f"Avg: {inference_time_avg:.6f}s, "
        f"Expected: {expected_inference_time:.6f}s ± {tolerance:.2f}s"
    )
    logger.info(
        f"ttnn_vanilla_unet batch_size: {total_batch_size}, resolution: {resolution}. One inference iteration time (sec): {inference_time_avg}, FPS: {round(total_batch_size/inference_time_avg)}"
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": VANILLA_UNET_L1_SMALL_SIZE, "trace_region_size": 1605632, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype",
    ((1, ttnn.bfloat8_b, ttnn.bfloat8_b),),
)
@pytest.mark.parametrize(
    "resolution",
    [
        (480, 640),
    ],
)
def test_e2e_performant(
    device,
    batch_size,
    act_dtype,
    weight_dtype,
    model_location_generator,
    resolution,
):
    return run_e2e_performant(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        model_location_generator,
        resolution,
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": VANILLA_UNET_L1_SMALL_SIZE, "trace_region_size": 1605632, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_batch_size, act_dtype, weight_dtype",
    ((1, ttnn.bfloat8_b, ttnn.bfloat8_b),),
)
@pytest.mark.parametrize(
    "resolution",
    [
        (480, 640),
    ],
)
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
def test_e2e_performant_dp(
    mesh_device,
    device_batch_size,
    act_dtype,
    weight_dtype,
    model_location_generator,
    resolution,
):
    return run_e2e_performant(
        mesh_device,
        device_batch_size,
        act_dtype,
        weight_dtype,
        model_location_generator,
        resolution,
    )
