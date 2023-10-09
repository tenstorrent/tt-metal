import pytest

from models.t5.tt.t5_for_conditional_generation import (
    t5_base_for_conditional_generation,
)

from tests.models.t5.tests.demo_utils import run_perf_t5


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "model_name,expected_inference_time, expected_compile_time, iteration",
    (
        (
            "t5-base",
            0.07,
            6.5,
            39,
        ),
    ),
)
def test_perf_bare_metal(
    model_name, use_program_cache, expected_inference_time, expected_compile_time, iteration, device
):
    run_perf_t5(
        t5_base_for_conditional_generation,
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
            "t5-base",
            0.12,
            7,
            39,
        ),
    ),
)
def test_perf_virtual_machine(
    model_name, use_program_cache, expected_inference_time, expected_compile_time, iteration, device
):
    run_perf_t5(
        t5_base_for_conditional_generation,
        model_name,
        expected_inference_time,
        expected_compile_time,
        iteration,
        device,
    )
