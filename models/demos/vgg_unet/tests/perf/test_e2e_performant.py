# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.vgg_unet.runner.performant_runner import VggUnetTrace2CQ
from models.utility_functions import run_for_wormhole_b0


def run_vgg_unet_e2e(device, device_batch_size, model_location_generator, channels=3, resolution=256):
    vgg_unet_trace_2cq = VggUnetTrace2CQ()
    vgg_unet_trace_2cq.initialize_vgg_unet_trace_2cqs_inference(
        device,
        model_location_generator=model_location_generator,
        device_batch_size=device_batch_size,
    )
    batch_size = device_batch_size * device.get_num_devices()
    input_shape = (batch_size, channels, resolution, resolution)
    torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)
    inference_iter_count = 10
    inference_time_iter = []
    for iter in range(0, inference_iter_count):
        t0 = time.time()
        output = vgg_unet_trace_2cq.run(torch_input_tensor)
        if iter + 1 == inference_iter_count:
            ttnn.synchronize_device(device)
        t1 = time.time()
        inference_time_iter.append(t1 - t0)
    vgg_unet_trace_2cq.release_vgg_unet_trace_2cqs_inference()
    inference_time_avg = round(sum(inference_time_iter) / len(inference_time_iter), 6)
    logger.info(
        f"ttnn_vgg_unet_256x256_batch_size_{batch_size}. One inference iteration time (sec): {inference_time_avg}, FPS: {round(batch_size/inference_time_avg)}"
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size",
    ((1),),
)
def test_vgg_unet_e2e(device, batch_size, model_location_generator):
    return run_vgg_unet_e2e(device, batch_size, model_location_generator)


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "device_batch_size",
    ((1),),
)
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
def test_vgg_unet_e2e_dp(mesh_device, device_batch_size, model_location_generator):
    return run_vgg_unet_e2e(mesh_device, device_batch_size, model_location_generator)
