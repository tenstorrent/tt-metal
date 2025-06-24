# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.demos.vit.tests.perf_e2e_vit import run_perf_vit
from models.utility_functions import run_for_wormhole_b0


@run_for_wormhole_b0()
@pytest.mark.model_perf_t3000
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "device_batch_size, expected_inference_time, expected_compile_time",
    ((8, 0.0120, 60),),
)
def test_perf(
    mesh_device,
    use_program_cache,
    device_batch_size,
    expected_inference_time,
    expected_compile_time,
):
    run_perf_vit(
        device_batch_size,
        expected_inference_time,
        expected_compile_time,
        mesh_device,
        f"vit",
    )


@run_for_wormhole_b0()
@pytest.mark.model_perf_t3000
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768, "trace_region_size": 1500000}], indirect=True)
@pytest.mark.parametrize(
    "device_batch_size, expected_inference_time, expected_compile_time",
    ((16, 0.0070, 60),),
)
def test_perf_trace(
    mesh_device,
    use_program_cache,
    device_batch_size,
    expected_inference_time,
    expected_compile_time,
):
    run_perf_vit(
        device_batch_size,
        expected_inference_time,
        expected_compile_time,
        mesh_device,
        f"vit_trace",
    )


@run_for_wormhole_b0()
@pytest.mark.model_perf_t3000
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768, "num_command_queues": 2}], indirect=True)
@pytest.mark.parametrize(
    "device_batch_size, expected_inference_time, expected_compile_time",
    ((16, 0.0125, 60),),
)
def test_perf_2cqs(
    mesh_device,
    use_program_cache,
    device_batch_size,
    expected_inference_time,
    expected_compile_time,
):
    run_perf_vit(
        device_batch_size,
        expected_inference_time,
        expected_compile_time,
        mesh_device,
        f"vit_2cqs",
    )


@run_for_wormhole_b0()
@pytest.mark.model_perf_t3000
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768, "num_command_queues": 2, "trace_region_size": 1559552}], indirect=True
)
@pytest.mark.parametrize(
    "device_batch_size, expected_inference_time, expected_compile_time",
    ((8, 0.0043, 60),),
)
def test_perf_trace_2cqs(
    mesh_device,
    use_program_cache,
    device_batch_size,
    expected_inference_time,
    expected_compile_time,
):
    run_perf_vit(
        device_batch_size,
        expected_inference_time,
        expected_compile_time,
        mesh_device,
        f"vit_trace_2cqs",
    )
