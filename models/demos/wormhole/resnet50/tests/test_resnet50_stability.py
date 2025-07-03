# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
from loguru import logger

import ttnn
from models.demos.wormhole.resnet50.demo.demo import test_demo_trace_with_imagenet
from models.utility_functions import run_for_wormhole_b0

test_demo_trace_with_imagenet.__test__ = False


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 1605632, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size, iterations, act_dtype, weight_dtype",
    ((16, 100, ttnn.bfloat8_b, ttnn.bfloat8_b),),
)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("test_duration_seconds", [24 * 60 * 60, 10], ids=["long", "short"])
def test_resnet_stability(
    mesh_device,
    batch_size,
    iterations,
    imagenet_label_dict,
    act_dtype,
    weight_dtype,
    model_location_generator,
    test_duration_seconds,
):
    logger.info(f"Running ResNet50 stability test for {test_duration_seconds} seconds")

    start = time.time()
    iter = 0

    while True:
        iter += 1

        test_demo_trace_with_imagenet(
            mesh_device,
            batch_size,
            iterations,
            imagenet_label_dict,
            act_dtype,
            weight_dtype,
            model_location_generator,
        )

        if time.time() - start > test_duration_seconds:
            break

    logger.info(f"ResNet50 stability test completed after {iter} iterations and {time.time() - start} seconds")
