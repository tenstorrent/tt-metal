# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.common.utility_functions import run_for_wormhole_b0
from models.demos.ttnn_resnet.tests.common.perf_e2e_resnet50 import run_perf_resnet

PERF_EXPECTATIONS = {
    "t3000": {
        "test_perf": {"inference_time": 0.015, "compile_time": 60},
        "test_perf_trace": {"inference_time": 0.0057, "compile_time": 60},
        "test_perf_2cqs": {"inference_time": 0.014, "compile_time": 60},
        "test_perf_trace_2cqs": {"inference_time": 0.007, "compile_time": 60},
    },
    "tg": {
        "test_perf": {"inference_time": 0.0500, "compile_time": 60},
        "test_perf_trace": {"inference_time": 0.0081, "compile_time": 60},
        "test_perf_2cqs": {"inference_time": 0.0530, "compile_time": 60},
        "test_perf_trace_2cqs": {"inference_time": 0.0085, "compile_time": 60},
    },
}


def get_platform_config(mesh_device):
    """Determine platform based on number of devices."""
    num_devices = mesh_device.get_num_devices()
    if num_devices == 8:
        return "t3000"
    elif num_devices == 32:
        return "tg"
    else:
        pytest.skip(f"Unsupported number of devices: {num_devices}. Expected 8 (T3000) or 32 (TG)")


@run_for_wormhole_b0()
@pytest.mark.model_perf_t3000
@pytest.mark.model_perf_tg
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "device_batch_size",
    (16,),
)
def test_perf(
    mesh_device,
    device_batch_size,
    hf_cat_image_sample_input,
    model_location_generator,
):
    platform = get_platform_config(mesh_device)

    expected_inference_time = PERF_EXPECTATIONS[platform]["test_perf"]["inference_time"]
    expected_compile_time = PERF_EXPECTATIONS[platform]["test_perf"]["compile_time"]

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
@pytest.mark.model_perf_t3000
@pytest.mark.model_perf_tg
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768, "trace_region_size": 1500000}], indirect=True)
@pytest.mark.parametrize(
    "device_batch_size",
    (16,),
)
def test_perf_trace(
    mesh_device,
    device_batch_size,
    hf_cat_image_sample_input,
    model_location_generator,
):
    platform = get_platform_config(mesh_device)

    expected_inference_time = PERF_EXPECTATIONS[platform]["test_perf_trace"]["inference_time"]
    expected_compile_time = PERF_EXPECTATIONS[platform]["test_perf_trace"]["compile_time"]

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
@pytest.mark.model_perf_t3000
@pytest.mark.model_perf_tg
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768, "num_command_queues": 2}], indirect=True)
@pytest.mark.parametrize(
    "device_batch_size",
    (16,),
)
def test_perf_2cqs(
    mesh_device,
    device_batch_size,
    hf_cat_image_sample_input,
    model_location_generator,
):
    platform = get_platform_config(mesh_device)

    expected_inference_time = PERF_EXPECTATIONS[platform]["test_perf_2cqs"]["inference_time"]
    expected_compile_time = PERF_EXPECTATIONS[platform]["test_perf_2cqs"]["compile_time"]

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
@pytest.mark.model_perf_t3000
@pytest.mark.model_perf_tg
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768, "num_command_queues": 2, "trace_region_size": 1332224}], indirect=True
)
@pytest.mark.parametrize(
    "device_batch_size",
    (16,),
)
def test_perf_trace_2cqs(
    mesh_device,
    device_batch_size,
    hf_cat_image_sample_input,
    model_location_generator,
):
    platform = get_platform_config(mesh_device)

    expected_inference_time = PERF_EXPECTATIONS[platform]["test_perf_trace_2cqs"]["inference_time"]
    expected_compile_time = PERF_EXPECTATIONS[platform]["test_perf_trace_2cqs"]["compile_time"]

    run_perf_resnet(
        device_batch_size,
        expected_inference_time,
        expected_compile_time,
        hf_cat_image_sample_input,
        mesh_device,
        f"resnet50_trace_2cqs",
        model_location_generator,
    )
