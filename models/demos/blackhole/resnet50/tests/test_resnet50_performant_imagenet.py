# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.wormhole.resnet50.tests.test_resnet50_performant_imagenet import (
    test_run_resnet50_trace_2cqs_inference as run_resnet50_trace_2cqs_inference,
)


@run_for_blackhole()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 5554176, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size_per_device, iterations, act_dtype, weight_dtype",
    (
        (16, 100, ttnn.bfloat8_b, ttnn.bfloat8_b),
        (32, 100, ttnn.bfloat8_b, ttnn.bfloat8_b),
    ),
)
def test_run_resnet50_trace_2cqs_inference(
    device,
    batch_size_per_device,
    iterations,
    imagenet_label_dict,
    act_dtype,
    weight_dtype,
    model_location_generator,
):
    run_resnet50_trace_2cqs_inference(
        mesh_device=device,
        batch_size_per_device=batch_size_per_device,
        iterations=iterations,
        imagenet_label_dict=imagenet_label_dict,
        act_dtype=act_dtype,
        weight_dtype=weight_dtype,
        model_location_generator=model_location_generator,
        entire_imagenet_dataset=False,
        expected_accuracy=0.7555288461538462,
    )
