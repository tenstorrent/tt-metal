# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
from models.demos.ttnn_resnet.demo.demo import run_resnet_imagenet_inference, run_resnet_inference
from models.utility_functions import run_for_wormhole_b0
import ttnn
from models.demos.wormhole.resnet50.tests.test_resnet50_performant_imagenet import (
    test_run_resnet50_trace_2cqs_inference,
)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, iterations",
    ((16, 100),),
)
def test_demo_imagenet(
    batch_size, use_program_cache, iterations, imagenet_label_dict, model_location_generator, device
):
    run_resnet_imagenet_inference(batch_size, iterations, imagenet_label_dict, model_location_generator, device)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, input_loc",
    ((16, "models/demos/ttnn_resnet/demo/images/"),),
)
def test_demo_sample(device, use_program_cache, batch_size, input_loc, imagenet_label_dict, model_location_generator):
    run_resnet_inference(batch_size, input_loc, imagenet_label_dict, device, model_location_generator)


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 1605632, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype",
    ((16, ttnn.bfloat8_b, ttnn.bfloat8_b),),
)
def test_demo_trace_with_imagenet(
    device,
    use_program_cache,
    batch_size,
    imagenet_label_dict,
    act_dtype,
    weight_dtype,
    model_location_generator,
    enable_async_mode=True,
):
    test_run_resnet50_trace_2cqs_inference(
        device,
        use_program_cache,
        batch_size,
        imagenet_label_dict,
        act_dtype,
        weight_dtype,
        enable_async_mode,
        model_location_generator,
    )
