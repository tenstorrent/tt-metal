# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
from models.utility_functions import (
    run_for_wormhole_b0,
)
from models.demos.ttnn_resnet.tests.multi_device.resnet50_performant import (
    run_resnet50_inference,
    run_resnet50_2cqs_inference,
    run_resnet50_trace_inference,
    run_resnet50_trace_2cqs_inference,
)


@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "device_batch_size, act_dtype, weight_dtype, math_fidelity",
    ((16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),),
)
@pytest.mark.parametrize("enable_async_mode", [True], indirect=True)
def test_run_resnet50_inference(
    mesh_device,
    use_program_cache,
    device_batch_size,
    act_dtype,
    weight_dtype,
    math_fidelity,
    enable_async_mode,
    model_location_generator,
):
    run_resnet50_inference(
        mesh_device,
        device_batch_size,
        act_dtype,
        weight_dtype,
        math_fidelity,
        model_location_generator,
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 800768}], indirect=True)
@pytest.mark.parametrize(
    "device_batch_size, act_dtype, weight_dtype, math_fidelity",
    ((16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),),
)
@pytest.mark.parametrize("enable_async_mode", [True], indirect=True)
def test_run_resnet50_trace_inference(
    mesh_device,
    use_program_cache,
    device_batch_size,
    act_dtype,
    weight_dtype,
    math_fidelity,
    enable_async_mode,
    model_location_generator,
):
    run_resnet50_trace_inference(
        mesh_device,
        device_batch_size,
        act_dtype,
        weight_dtype,
        math_fidelity,
        model_location_generator,
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "num_command_queues": 2}], indirect=True)
@pytest.mark.parametrize(
    "device_batch_size, act_dtype, weight_dtype, math_fidelity",
    ((16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),),
)
@pytest.mark.parametrize("enable_async_mode", [True], indirect=True)
def test_run_resnet50_2cqs_inference(
    mesh_device,
    use_program_cache,
    device_batch_size,
    act_dtype,
    weight_dtype,
    math_fidelity,
    enable_async_mode,
    model_location_generator,
):
    run_resnet50_2cqs_inference(
        mesh_device,
        device_batch_size,
        act_dtype,
        weight_dtype,
        math_fidelity,
        model_location_generator,
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 800768, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "device_batch_size, act_dtype, weight_dtype, math_fidelity",
    ((16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),),
)
@pytest.mark.parametrize("enable_async_mode", [True], indirect=True)
def test_run_resnet50_trace_2cqs_inference(
    mesh_device,
    use_program_cache,
    device_batch_size,
    act_dtype,
    weight_dtype,
    math_fidelity,
    enable_async_mode,
    model_location_generator,
):
    run_resnet50_trace_2cqs_inference(
        mesh_device,
        device_batch_size,
        act_dtype,
        weight_dtype,
        math_fidelity,
        model_location_generator,
    )
