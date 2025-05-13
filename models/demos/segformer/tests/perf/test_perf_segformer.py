# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

import models.perf.perf_utils as perf_utils
import ttnn
from models.demos.segformer.tests.perf.segformer_test_infra import SegformerBare
from models.utility_functions import run_for_wormhole_b0


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype, expected_compile_time, expected_inference_time",
    [
        [1, ttnn.bfloat16, ttnn.bfloat16, 65, 0.0277],
    ],
)
def test_perf_segformer(device, batch_size, act_dtype, weight_dtype, expected_compile_time, expected_inference_time):
    device.enable_program_cache()

    segformer = SegformerBare(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        model_location_generator=None,
    )

    segformer.compile()
    segformer.cache()
    segformer.optimized_inference()

    perf_utils.prep_perf_report(
        model_name="segformer_e2e",
        batch_size=batch_size,
        inference_and_compile_time=segformer.jit_time,
        inference_time=segformer.inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments="bare",
    )

    assert (
        segformer.inference_time < expected_inference_time
    ), f"Segformer inference time {segformer.inference_time} is too slow, expected {expected_inference_time}"
