# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.ttnn_resnet.tests.common.resnet50_performant import (
    run_resnet50_2cqs_inference,
    run_resnet50_inference,
    run_resnet50_trace_2cqs_inference,
    run_resnet50_trace_inference,
)


@run_for_blackhole()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("batch_size", (16, 32))
@pytest.mark.parametrize(
    "act_dtype, weight_dtype, math_fidelity",
    ((ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),),
)
@pytest.mark.parametrize("skip_compile_run", [True, False])
def test_run_resnet50_inference(
    device, batch_size, act_dtype, weight_dtype, math_fidelity, model_location_generator, skip_compile_run
):
    run_resnet50_inference(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        math_fidelity,
        model_location_generator,
        skip_compile_run=skip_compile_run,
    )


@run_for_blackhole()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 5554176}], indirect=True)
@pytest.mark.parametrize("batch_size", (16, 32))
@pytest.mark.parametrize(
    "act_dtype, weight_dtype, math_fidelity",
    ((ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),),
)
def test_run_resnet50_trace_inference(
    device,
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
    device, batch_size, act_dtype, weight_dtype, math_fidelity, model_location_generator
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
