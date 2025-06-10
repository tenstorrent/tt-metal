# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.vgg_unet.tests.vgg_unet_e2e_performant import VggUnetTrace2CQ
from models.demos.vgg_unet.tests.vgg_unet_performant import run_vgg_unet_inference, run_vgg_unet_trace_inference
from models.utility_functions import run_for_wormhole_b0


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_run_vgg_unet_inference(device, use_program_cache, model_location_generator):
    run_vgg_unet_inference(device, model_location_generator)


@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768, "trace_region_size": 1843200}], indirect=True)
def test_run_vgg_unet_trace_inference(
    device,
    use_program_cache,
    model_location_generator,
):
    run_vgg_unet_trace_inference(
        device,
        model_location_generator,
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768, "trace_region_size": 3686400, "num_command_queues": 2}], indirect=True
)
def test_run_vgg_unet_trace_2cqs_inference(
    device,
    use_program_cache,
    model_location_generator,
):
    vgg_unet_trace_2cq = VggUnetTrace2CQ()

    vgg_unet_trace_2cq.initialize_vgg_unet_trace_2cqs_inference(
        device,
        model_location_generator,
    )

    input_shape = (1, 3, 256, 256)
    batch_size = input_shape[0]
    torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)
    n, c, h, w = torch_input_tensor.shape
    torch_input_tensor = torch_input_tensor.permute(0, 2, 3, 1)
    torch_input_tensor = torch_input_tensor.reshape(1, 1, h * w * n, c)
    tt_inputs_host = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_inputs_host = ttnn.pad(tt_inputs_host, [1, 1, n * h * w, 16], [0, 0, 0, 0], 0)
    inference_iter_count = 10
    inference_time_iter = []
    for iter in range(0, inference_iter_count):
        output = vgg_unet_trace_2cq.execute_vgg_unet_trace_2cqs_inference(tt_inputs_host)
    vgg_unet_trace_2cq.release_vgg_unet_trace_2cqs_inference()
