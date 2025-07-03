# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.demos.ttnn_resnet.tests.perf_e2e_resnet50 import run_perf_resnet
from models.utility_functions import run_for_wormhole_b0


@run_for_wormhole_b0()
@pytest.mark.model_perf_tg
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "device_batch_size, expected_inference_time, expected_compile_time",
    ((16, 0.0500, 60),),
)
@pytest.mark.parametrize(
    "mesh_device",
    ((8, 4),),
    indirect=True,
)
def test_perf(
    mesh_device,
    device_batch_size,
    expected_inference_time,
    expected_compile_time,
    hf_cat_image_sample_input,
    model_location_generator,
):
    run_perf_resnet(
        device_batch_size,
        expected_inference_time,
        expected_compile_time,
        hf_cat_image_sample_input,
        mesh_device,
        f"resnet50",
        model_location_generator,
    )


@run_for_wormhole_b0()
@pytest.mark.model_perf_tg
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768, "trace_region_size": 1500000}], indirect=True)
@pytest.mark.parametrize(
    "device_batch_size, expected_inference_time, expected_compile_time",
    ((16, 0.0081, 60),),
)
@pytest.mark.parametrize(
    "mesh_device",
    ((8, 4),),
    indirect=True,
)
def test_perf_trace(
    mesh_device,
    device_batch_size,
    expected_inference_time,
    expected_compile_time,
    hf_cat_image_sample_input,
    model_location_generator,
):
    run_perf_resnet(
        device_batch_size,
        expected_inference_time,
        expected_compile_time,
        hf_cat_image_sample_input,
        mesh_device,
        f"resnet50_trace",
        model_location_generator,
    )


@run_for_wormhole_b0()
@pytest.mark.model_perf_tg
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768, "num_command_queues": 2}], indirect=True)
@pytest.mark.parametrize(
    "device_batch_size, expected_inference_time, expected_compile_time",
    ((16, 0.0530, 60),),
)
def test_perf_2cqs(
    mesh_device,
    device_batch_size,
    expected_inference_time,
    expected_compile_time,
    hf_cat_image_sample_input,
    model_location_generator,
):
    run_perf_resnet(
        device_batch_size,
        expected_inference_time,
        expected_compile_time,
        hf_cat_image_sample_input,
        mesh_device,
        f"resnet50_2cqs",
        model_location_generator,
    )


@run_for_wormhole_b0()
@pytest.mark.model_perf_tg
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768, "num_command_queues": 2, "trace_region_size": 1332224}], indirect=True
)
@pytest.mark.parametrize(
    "device_batch_size, expected_inference_time, expected_compile_time",
    ((16, 0.0085, 60),),
)
def test_perf_trace_2cqs(
    mesh_device,
    device_batch_size,
    expected_inference_time,
    expected_compile_time,
    hf_cat_image_sample_input,
    model_location_generator,
):
    run_perf_resnet(
        device_batch_size,
        expected_inference_time,
        expected_compile_time,
        hf_cat_image_sample_input,
        mesh_device,
        f"resnet50_trace_2cqs",
        model_location_generator,
    )
