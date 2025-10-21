# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.common.utility_functions import run_for_blackhole
from models.demos.ttnn_resnet.tests.common.perf_e2e_resnet50 import run_perf_resnet

# These perf figures were measured on one of the machines in ird -
# as e2e perf depends on the performance of the host machine,
# these figures will be appropriately modified at the time of adding tests to CI


@run_for_blackhole()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, expected_inference_time, expected_compile_time",
    (
        (16, 0.0065, 30),
        (32, 0.0071, 30),
    ),
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


@run_for_blackhole()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768, "trace_region_size": 5554176}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, expected_inference_time, expected_compile_time",
    (
        (16, 0.0024, 30),
        (32, 0.004, 30),
    ),
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


@run_for_blackhole()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768, "num_command_queues": 2}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, expected_inference_time, expected_compile_time",
    (
        (16, 0.06, 30),
        (32, 0.06, 30),
    ),
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


@run_for_blackhole()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768, "num_command_queues": 2, "trace_region_size": 2777088}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size, expected_inference_time, expected_compile_time",
    ((16, 0.0018, 30), (32, 0.0026, 30)),
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
