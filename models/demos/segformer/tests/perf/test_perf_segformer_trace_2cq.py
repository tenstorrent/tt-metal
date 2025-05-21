# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn.torch_tracer

import ttnn
from models.demos.segformer.tests.perf.segformer_test_infra import SegformerTrace2CQ
from models.perf.perf_utils import prep_perf_report
from models.utility_functions import run_for_wormhole_b0


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "num_command_queues": 2, "trace_region_size": 1824800}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype, expected_compile_time, expected_inference_time",
    [
        [1, ttnn.bfloat16, ttnn.bfloat16, 65, 0.0125],
    ],
)
def test_perf_segformer_trace_2cq(
    device, batch_size, act_dtype, weight_dtype, expected_compile_time, expected_inference_time
):
    device.enable_program_cache()

    segformer_t2cq = SegformerTrace2CQ(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        model_location_generator=None,
    )

    compile_finished = segformer_t2cq.compile()
    capture_finished = segformer_t2cq.trace_capture(compile_finished)
    segformer_t2cq.trace_execute(capture_finished)

    segformer_t2cq.validate_outputs()

    prep_perf_report(
        model_name="segformer_e2e",
        batch_size=batch_size,
        inference_and_compile_time=segformer_t2cq.jit_time,
        inference_time=segformer_t2cq.inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments="trace_2cq",
    )

    assert (
        segformer_t2cq.inference_time < expected_inference_time
    ), f"Segformer inference time {segformer_t2cq.inference_time} is too slow, expected {expected_inference_time}"
