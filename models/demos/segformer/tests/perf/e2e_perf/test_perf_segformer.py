# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import time
from models.perf.perf_utils import prep_perf_report
from models.utility_functions import run_for_wormhole_b0
from models.demos.segformer.tests.segformer_test_infra import create_test_infra


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype, expected_compile_time, expected_inference_time",
    [
        [1, ttnn.bfloat16, ttnn.bfloat16, 35, 1.5],
    ],
)
def test_perf_segformer(device, batch_size, act_dtype, weight_dtype, expected_compile_time, expected_inference_time):
    test_infra = create_test_infra(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        model_location_generator=None,
    )

    tt_inputs_host, input_mem_config = test_infra.setup_l1_sharded_input(device)

    # JIT Run
    start_jit = time.time()
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    test_infra.run()
    end_jit = time.time()
    test_infra.validate()
    test_infra.dealloc_output()

    # Cache Run
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    test_infra.run()
    test_infra.validate()
    test_infra.dealloc_output()

    # Optimized Run
    start = time.time()
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    test_infra.run()
    end = time.time()
    test_infra.validate()
    test_infra.dealloc_output()

    prep_perf_report(
        model_name="segformer_e2e",
        batch_size=batch_size,
        inference_and_compile_time=end_jit - start_jit,
        inference_time=end - start,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments="bare",
    )
