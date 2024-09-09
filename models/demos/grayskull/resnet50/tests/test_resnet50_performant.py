# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from models.utility_functions import run_for_grayskull
from models.demos.ttnn_resnet.tests.resnet50_performant import (
    run_resnet50_inference,
    run_resnet50_2cqs_inference,
    run_resnet50_trace_inference,
    run_resnet50_trace_2cqs_inference,
)


@run_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype, math_fidelity",
    ((20, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),),
)
def test_run_resnet50_inference(device, batch_size, act_dtype, weight_dtype, math_fidelity, model_location_generator):
    run_resnet50_inference(device, batch_size, act_dtype, weight_dtype, math_fidelity, model_location_generator)


@run_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768, "trace_region_size": 1332224}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype, math_fidelity",
    ((20, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),),
)
@pytest.mark.parametrize("enable_async_mode", (False, True), indirect=True)
def test_run_resnet50_trace_inference(
    device,
    use_program_cache,
    batch_size,
    act_dtype,
    weight_dtype,
    math_fidelity,
    enable_async_mode,
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


@run_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768, "num_hw_cqs": 2}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype, math_fidelity",
    ((20, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),),
)
def test_run_resnet50_2cqs_inference(
    device, use_program_cache, batch_size, act_dtype, weight_dtype, math_fidelity, model_location_generator
):
    run_resnet50_2cqs_inference(device, batch_size, act_dtype, weight_dtype, math_fidelity, model_location_generator)


@run_for_grayskull()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768, "trace_region_size": 1332224, "num_hw_cqs": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype, math_fidelity",
    ((20, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),),
)
@pytest.mark.parametrize("enable_async_mode", (False, True), indirect=True)
def test_run_resnet50_trace_2cqs_inference(
    device,
    use_program_cache,
    batch_size,
    act_dtype,
    weight_dtype,
    math_fidelity,
    enable_async_mode,
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
