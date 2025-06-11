# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn
from models.demos.ttnn_resnet.tests.resnet50_performant import (
    run_resnet50_2cqs_inference,
    run_resnet50_inference,
    run_resnet50_trace_2cqs_inference,
    run_resnet50_trace_inference,
)
from models.utility_functions import run_for_blackhole


@run_for_blackhole()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("batch_size", (16, 32))
@pytest.mark.parametrize(
    "act_dtype, weight_dtype, math_fidelity",
    ((ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),),
)
def test_run_resnet50_inference(
    device, use_program_cache, batch_size, act_dtype, weight_dtype, math_fidelity, model_location_generator
):
    run_resnet50_inference(device, batch_size, act_dtype, weight_dtype, math_fidelity, model_location_generator)


@run_for_blackhole()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 5554176}], indirect=True)
@pytest.mark.parametrize("batch_size", (16, 32))
@pytest.mark.parametrize(
    "act_dtype, weight_dtype, math_fidelity",
    ((ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),),
)
def test_run_resnet50_trace_inference(
    device,
    use_program_cache,
    batch_size,
    act_dtype,
    weight_dtype,
    math_fidelity,
    model_location_generator,
):
    run_resnet50_trace_inference(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        math_fidelity,
        model_location_generator,
    )


@run_for_blackhole()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "num_command_queues": 2}], indirect=True)
@pytest.mark.parametrize("batch_size", (16, 32))
@pytest.mark.parametrize(
    "act_dtype, weight_dtype, math_fidelity",
    ((ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),),
)
def test_run_resnet50_2cqs_inference(
    device, use_program_cache, batch_size, act_dtype, weight_dtype, math_fidelity, model_location_generator
):
    run_resnet50_2cqs_inference(device, batch_size, act_dtype, weight_dtype, math_fidelity, model_location_generator)


@run_for_blackhole()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 5554176, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize("batch_size", (16, 32))
@pytest.mark.parametrize(
    "act_dtype, weight_dtype, math_fidelity",
    ((ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),),
)
def test_run_resnet50_trace_2cqs_inference(
    device,
    use_program_cache,
    batch_size,
    act_dtype,
    weight_dtype,
    math_fidelity,
    model_location_generator,
):
    run_resnet50_trace_2cqs_inference(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        math_fidelity,
        model_location_generator,
    )
