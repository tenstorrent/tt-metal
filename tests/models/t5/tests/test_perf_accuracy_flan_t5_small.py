# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import pytest

from models.t5.tt.t5_for_conditional_generation import (
    flan_t5_small_for_conditional_generation,
)

from tests.models.t5.tests.demo_utils import run_perf_t5


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "model_name,expected_inference_time, expected_compile_time, iteration",
    (
        (
            "google/flan-t5-small",
            0.07,
            6.5,
            50,
        ),
    ),
)
def test_perf_bare_metal(
    model_name, use_program_cache, expected_inference_time, expected_compile_time, iteration, device
):
    run_perf_t5(
        flan_t5_small_for_conditional_generation,
        model_name,
        expected_inference_time,
        expected_compile_time,
        iteration,
        device,
    )


@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "model_name,expected_inference_time, expected_compile_time, iteration",
    (
        (
            "google/flan-t5-small",
            0.12,
            7,
            50,
        ),
    ),
)
def test_perf_virtual_machine(
    model_name, use_program_cache, expected_inference_time, expected_compile_time, iteration, device
):
    run_perf_t5(
        flan_t5_small_for_conditional_generation,
        model_name,
        expected_inference_time,
        expected_compile_time,
        iteration,
        device,
    )
