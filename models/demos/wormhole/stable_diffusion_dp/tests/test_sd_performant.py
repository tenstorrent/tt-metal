# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from models.utility_functions import run_for_wormhole_b0
from models.demos.wormhole.stable_diffusion_dp.tests.sd_performant import (
    run_sd_inference,
    run_sd_2cqs_inference,
    run_sd_trace_inference,
    run_sd_trace_2cqs_inference,
)


@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, num_inference_steps",
    ((2, 5),),
)
@pytest.mark.parametrize(
    "input_shape",
    ((2, 4, 64, 64),),
)
def test_run_sd_inference(device, use_program_cache, batch_size, num_inference_steps, input_shape):
    # device = mesh_device
    batch_size, C, H, W = input_shape
    is_mesh_device = isinstance(device, ttnn.MeshDevice)
    num_devices = device.get_num_devices() if is_mesh_device else 1
    batch_size = batch_size * num_devices
    input_shape = batch_size, C, H, W
    run_sd_inference(device, batch_size, num_inference_steps, input_shape)


@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 800768}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, num_inference_steps, enable_async_mode",
    (
        (2, 5, True),
        (2, 5, False),
    ),
    indirect=["enable_async_mode"],
)
@pytest.mark.parametrize(
    "input_shape",
    ((2, 4, 64, 64),),
)
def test_run_sd_trace_inference(
    device,
    use_program_cache,
    batch_size,
    num_inference_steps,
    enable_async_mode,
    input_shape,
):
    mode = "async" if enable_async_mode else "sync"
    run_sd_trace_inference(
        device,
        batch_size,
        num_inference_steps=num_inference_steps,
        input_shape=input_shape,
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "num_command_queues": 2}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, num_inference_steps",
    ((2, 5),),
)
@pytest.mark.parametrize(
    "input_shape",
    ((2, 4, 64, 64),),
)
def test_run_sd_2cqs_inference(device, reset_seeds, use_program_cache, batch_size, num_inference_steps, input_shape):
    run_sd_2cqs_inference(
        device,
        batch_size,
        num_inference_steps=num_inference_steps,
        input_shape=input_shape,
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 800768, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size, num_inference_steps, enable_async_mode",
    (
        (2, 5, True),
        (2, 5, False),
    ),
    indirect=["enable_async_mode"],
)
@pytest.mark.parametrize(
    "input_shape",
    ((2, 4, 64, 64),),
)
def test_run_sd_trace_2cqs_inference(
    device, reset_seeds, use_program_cache, batch_size, num_inference_steps, enable_async_mode, input_shape
):
    run_sd_trace_2cqs_inference(
        device,
        batch_size,
        num_inference_steps=num_inference_steps,
        input_shape=input_shape,
    )
