# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn
from models.demos.vision.classification.resnet50.quasar.tests.common.resnet50_performant import (
    run_resnet50_2cqs_inference,
    run_resnet50_inference,
    run_resnet50_trace_2cqs_inference,
    run_resnet50_trace_inference,
)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype, math_fidelity",
    ((16, ttnn.bfloat16, ttnn.bfloat16, ttnn.MathFidelity.LoFi),),
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


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 845824}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype, math_fidelity",
    ((16, ttnn.bfloat16, ttnn.bfloat16, ttnn.MathFidelity.LoFi),),
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


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "num_command_queues": 2}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype, math_fidelity",
    ((16, ttnn.bfloat16, ttnn.bfloat16, ttnn.MathFidelity.LoFi),),
)
def test_run_resnet50_2cqs_inference(
    device, batch_size, act_dtype, weight_dtype, math_fidelity, model_location_generator
):
    run_resnet50_2cqs_inference(device, batch_size, act_dtype, weight_dtype, math_fidelity, model_location_generator)


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 845824, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype, math_fidelity",
    ((16, ttnn.bfloat16, ttnn.bfloat16, ttnn.MathFidelity.LoFi),),
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
