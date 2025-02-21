# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from models.utility_functions import run_for_wormhole_b0
from models.demos.yolov4.tests.yolov4_perfomant import (
    run_yolov4_inference,
    run_yolov4_trace_inference,
    run_yolov4_trace_2cqs_inference,
)


@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype",
    ((1, ttnn.bfloat16, ttnn.bfloat16),),
)
def test_run_yolov4_inference(device, use_program_cache, batch_size, act_dtype, weight_dtype, model_location_generator):
    run_yolov4_inference(device, batch_size, act_dtype, weight_dtype, model_location_generator)


@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 6422528}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype",
    ((1, ttnn.bfloat16, ttnn.bfloat16),),
)
@pytest.mark.parametrize("enable_async_mode", (False, True), indirect=True)
def test_run_yolov4_trace_inference(
    device,
    use_program_cache,
    batch_size,
    act_dtype,
    weight_dtype,
    enable_async_mode,
    model_location_generator,
):
    run_yolov4_trace_inference(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        model_location_generator,
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 6397952, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype",
    ((1, ttnn.bfloat16, ttnn.bfloat16),),
)
@pytest.mark.parametrize("enable_async_mode", (False, True), indirect=True)
def test_run_yolov4_trace_2cqs_inference(
    device,
    use_program_cache,
    batch_size,
    act_dtype,
    weight_dtype,
    enable_async_mode,
    model_location_generator,
):
    run_yolov4_trace_2cqs_inference(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        model_location_generator,
    )
