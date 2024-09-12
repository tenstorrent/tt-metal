# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.demos.ttnn_resnet.tests.multi_device.perf_e2e_resnet50 import run_perf_resnet
from models.utility_functions import run_for_wormhole_b0


@run_for_wormhole_b0()
@pytest.mark.model_perf_tg
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "device_batch_size, enable_async_mode, expected_inference_time, expected_compile_time",
    ((16, True, 0.0365, 60),),
    indirect=["enable_async_mode"],
)
@pytest.mark.parametrize(
    "mesh_device",
    ((8, 4),),
    indirect=True,
)
def test_perf(
    mesh_device,
    use_program_cache,
    device_batch_size,
    expected_inference_time,
    expected_compile_time,
    hf_cat_image_sample_input,
    enable_async_mode,
    model_location_generator,
):
    mode = "async" if enable_async_mode else "sync"
    run_perf_resnet(
        device_batch_size,
        expected_inference_time,
        expected_compile_time,
        hf_cat_image_sample_input,
        mesh_device,
        f"resnet50_{mode}",
        model_location_generator,
    )


@run_for_wormhole_b0()
@pytest.mark.model_perf_tg
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768, "trace_region_size": 1500000}], indirect=True)
@pytest.mark.parametrize(
    "device_batch_size, enable_async_mode, expected_inference_time, expected_compile_time",
    ((16, True, 0.0081, 60),),
    indirect=["enable_async_mode"],
)
@pytest.mark.parametrize(
    "mesh_device",
    ((8, 4),),
    indirect=True,
)
def test_perf_trace(
    mesh_device,
    use_program_cache,
    device_batch_size,
    expected_inference_time,
    expected_compile_time,
    hf_cat_image_sample_input,
    enable_async_mode,
    model_location_generator,
):
    mode = "async" if enable_async_mode else "sync"
    run_perf_resnet(
        device_batch_size,
        expected_inference_time,
        expected_compile_time,
        hf_cat_image_sample_input,
        mesh_device,
        f"resnet50_trace_{mode}",
        model_location_generator,
    )
