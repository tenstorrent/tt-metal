# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest

from models.demos.ttnn_resnet.tests.perf_e2e_resnet50 import run_perf_resnet
from models.utility_functions import run_for_grayskull


@run_for_grayskull()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, expected_inference_time, expected_compile_time",
    ((20, 0.0090, 20),),
)
def test_perf(
    device,
    batch_size,
    expected_inference_time,
    expected_compile_time,
    hf_cat_image_sample_input,
    model_location_generator,
):
    run_perf_resnet(
        batch_size,
        expected_inference_time,
        expected_compile_time,
        hf_cat_image_sample_input,
        device,
        "resnet50",
        model_location_generator,
    )


@run_for_grayskull()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768, "trace_region_size": 1332224}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, expected_inference_time, expected_compile_time",
    ((20, 0.0068, 20)),
)
def test_perf_trace(
    device,
    batch_size,
    expected_inference_time,
    expected_compile_time,
    hf_cat_image_sample_input,
    model_location_generator,
):
    run_perf_resnet(
        batch_size,
        expected_inference_time,
        expected_compile_time,
        hf_cat_image_sample_input,
        device,
        f"resnet50_trace",
        model_location_generator,
    )


@run_for_grayskull()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768, "num_command_queues": 2}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, expected_inference_time, expected_compile_time",
    ((20, 0.0094, 20),),
)
def test_perf_2cqs(
    device,
    batch_size,
    expected_inference_time,
    expected_compile_time,
    hf_cat_image_sample_input,
    model_location_generator,
):
    run_perf_resnet(
        batch_size,
        expected_inference_time,
        expected_compile_time,
        hf_cat_image_sample_input,
        device,
        "resnet50_2cqs",
        model_location_generator,
    )


@run_for_grayskull()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768, "trace_region_size": 1332224, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size, expected_inference_time, expected_compile_time",
    ((20, 0.0042, 20),),
)
def test_perf_trace_2cqs(
    device,
    batch_size,
    expected_inference_time,
    expected_compile_time,
    hf_cat_image_sample_input,
    model_location_generator,
):
    run_perf_resnet(
        batch_size,
        expected_inference_time,
        expected_compile_time,
        hf_cat_image_sample_input,
        device,
        "resnet50_trace_2cqs",
        model_location_generator,
    )
