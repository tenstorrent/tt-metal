# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn
from models.demos.ttnn_resnet.tests.resnet50_performant import (
    run_resnet50_2cqs_inference,
    run_resnet50_inference,
    run_resnet50_trace_2cqs_inference,
    run_resnet50_trace_inference,
)
from models.utility_functions import run_for_wormhole_b0


@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "device_batch_size, act_dtype, weight_dtype, math_fidelity",
    ((16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),),
)
def test_run_resnet50_inference(
    mesh_device,
    device_batch_size,
    act_dtype,
    weight_dtype,
    math_fidelity,
    model_location_generator,
):
    if mesh_device.get_num_devices() != 8:
        pytest.skip("Not T3K!")

    run_resnet50_inference(
        mesh_device,
        device_batch_size,
        act_dtype,
        weight_dtype,
        math_fidelity,
        model_location_generator,
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 803016}], indirect=True)
@pytest.mark.parametrize(
    "device_batch_size, act_dtype, weight_dtype, math_fidelity",
    ((16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),),
)
def test_run_resnet50_trace_inference(
    mesh_device,
    device_batch_size,
    act_dtype,
    weight_dtype,
    math_fidelity,
    model_location_generator,
):
    if mesh_device.get_num_devices() != 8:
        pytest.skip("Not T3K!")

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
def test_run_resnet50_2cqs_inference(
    mesh_device,
    device_batch_size,
    act_dtype,
    weight_dtype,
    math_fidelity,
    model_location_generator,
):
    if mesh_device.get_num_devices() != 8:
        pytest.skip("Not T3K!")

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
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 803016, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "device_batch_size, act_dtype, weight_dtype, math_fidelity",
    ((16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),),
)
def test_run_resnet50_trace_2cqs_inference(
    mesh_device,
    device_batch_size,
    act_dtype,
    weight_dtype,
    math_fidelity,
    model_location_generator,
):
    if mesh_device.get_num_devices() != 8:
        pytest.skip("Not T3K!")

    run_resnet50_trace_2cqs_inference(
        mesh_device,
        device_batch_size,
        act_dtype,
        weight_dtype,
        math_fidelity,
        model_location_generator,
    )
