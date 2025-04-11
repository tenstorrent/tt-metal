# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
from models.utility_functions import (
    run_for_wormhole_b0,
)
import time
from loguru import logger
from models.demos.wormhole.resnet50.demo.demo import test_demo_trace_with_imagenet

test_demo_trace_with_imagenet.__test__ = False

# Define the duration for the test in seconds (24 hours)
TEST_DURATION_SECONDS = 60 * 60 * 24


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 1605632, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size, iterations, act_dtype, weight_dtype",
    ((16, 100, ttnn.bfloat8_b, ttnn.bfloat8_b),),
)
def test_resnet_stability(
    mesh_device,
    use_program_cache,
    batch_size,
    iterations,
    imagenet_label_dict,
    act_dtype,
    weight_dtype,
    model_location_generator,
    enable_async_mode=True,
):
    logger.info(f"Running ResNet stability test for {TEST_DURATION_SECONDS} seconds")

    start = time.time()
    iter = 0

    while True:
        iter += 1

        test_demo_trace_with_imagenet(
            mesh_device,
            use_program_cache,
            batch_size,
            iterations,
            imagenet_label_dict,
            act_dtype,
            weight_dtype,
            model_location_generator,
            enable_async_mode=True,
        )

        if time.time() - start > TEST_DURATION_SECONDS:
            break

        print(f"Completed iteration {iter}")

    logger.info(f"ResNet stability test completed after {iter} iterations and {time.time() - start} seconds")
