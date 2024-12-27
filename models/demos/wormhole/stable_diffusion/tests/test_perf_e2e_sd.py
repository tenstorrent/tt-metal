# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from models.utility_functions import run_for_wormhole_b0
from models.demos.wormhole.stable_diffusion.tests.perf_e2e_sd import run_perf_sd


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, num_inference_steps, expected_inference_time, expected_compile_time",
    ((2, 5, 0.15, 180),),
)
@pytest.mark.parametrize(
    "input_shape",
    ((2, 4, 64, 64),),
)
def test_perf(
    device,  # mesh_device
    use_program_cache,
    batch_size,
    num_inference_steps,
    expected_inference_time,
    expected_compile_time,
    input_shape,
):
    run_perf_sd(
        batch_size,
        num_inference_steps,
        expected_inference_time,
        expected_compile_time,
        input_shape,
        device,  # mesh_device
        "sd",
    )


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768, "trace_region_size": 800768}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, num_inference_steps, enable_async_mode, expected_inference_time, expected_compile_time",
    (
        (2, 5, True, 0.005, 30),
        # (2, 5, False, 0.0046, 30),
    ),
    indirect=["enable_async_mode"],
)
@pytest.mark.parametrize(
    "input_shape",
    ((2, 4, 64, 64),),
)
def test_perf_trace(
    device,
    use_program_cache,
    batch_size,
    num_inference_steps,
    expected_inference_time,
    expected_compile_time,
    input_shape,
    enable_async_mode,
):
    mode = "async" if enable_async_mode else "sync"
    run_perf_sd(
        batch_size,
        num_inference_steps,
        expected_inference_time,
        expected_compile_time,
        input_shape,
        device,
        f"sd_trace_{mode}",
    )


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768, "num_command_queues": 2}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, num_inference_steps, expected_inference_time, expected_compile_time",
    ((2, 5, 0.145, 13.77),),
)
@pytest.mark.parametrize(
    "input_shape",
    ((2, 4, 64, 64),),
)
def test_perf_2cqs(
    device,
    use_program_cache,
    batch_size,
    num_inference_steps,
    expected_inference_time,
    expected_compile_time,
    input_shape,
):
    run_perf_sd(
        batch_size,
        num_inference_steps,
        expected_inference_time,
        expected_compile_time,
        input_shape,
        device,
        "sd_2cqs",
    )


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768, "num_command_queues": 2, "trace_region_size": 12000000}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size, num_inference_steps, expected_inference_time, expected_compile_time",
    ((2, 5, 0.004, 30),),
)
@pytest.mark.parametrize(
    "input_shape",
    ((2, 4, 64, 64),),
)
def test_perf_trace_2cqs(
    device,
    use_program_cache,
    batch_size,
    num_inference_steps,
    expected_inference_time,
    expected_compile_time,
    input_shape,
):
    run_perf_sd(
        batch_size,
        num_inference_steps,
        expected_inference_time,
        expected_compile_time,
        input_shape,
        device,
        "sd_trace_2cqs",
    )
