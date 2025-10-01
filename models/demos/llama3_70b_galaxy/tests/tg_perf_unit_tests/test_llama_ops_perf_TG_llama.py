# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import pytest

from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report

from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler


@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    ("op_name", "expected_kernel_duration_4u_us", "expected_kernel_duration_6u_us", "perf_margin"),
    [
        # ("LayerNorm", 12.5, 10.9, 0.05),
        ("ScaledDotProductAttentionDecode", 10.17, 9.07, 0.05),
        # ("PagedUpdateCacheDeviceOperation", 4.5, 3.9, 0.16),
        # ("RotaryEmbeddingLlamaFusedQK", 3.92, 3.58, 0.05),
        # ("Embeddings", 3.8, 3.3, 0.1),
        # ("BinaryDeviceOperation", 2.78, 2.5, 0.05),
    ],
)
def test_llama_tg_ops_perf_device(
    op_name,
    expected_kernel_duration_4u_us,
    expected_kernel_duration_6u_us,
    perf_margin,
    galaxy_type,
):
    batch = 32
    test = "llama-distributed-ln"
    subdir = "llama-unit-tests"
    num_iterations = 3
    expected_kernel_duration_us = (
        expected_kernel_duration_4u_us if galaxy_type == "4U" else expected_kernel_duration_6u_us
    )
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"Llama_TG_{op_name}"

    command = f"pytest models/demos/llama3_70b_galaxy/tests/unit_tests/test_llama_ops.py::test_llama_tg_{op_name}"
    cols = ["DEVICE KERNEL"]
    inference_time_key = "DEVICE KERNEL DURATION [ns]"
    expected_perf_cols = {f"AVG {inference_time_key}": expected_kernel_duration_us * 1e3}

    profiler.start("run")
    profiler.start(step_name)
    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch, op_name, has_signposts=False)
    profiler.end(step_name)
    profiler.end("run")

    # Save the measurement
    measured_avg = post_processed_results[f"AVG {inference_time_key}"]
    measured_max = post_processed_results[f"MAX {inference_time_key}"]
    measured_min = post_processed_results[f"MIN {inference_time_key}"]

    benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-avg", measured_avg)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-max", measured_max)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-min", measured_min)
    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"tg_llama_ops" if galaxy_type != "6U" else "tg_llama_ops_6U",
        ml_model_name="llama70b-tg",
    )
    expected_results = check_device_perf(post_processed_results, perf_margin, expected_perf_cols, assert_on_fail=True)

    prep_device_perf_report(
        model_name=f"llama-tg-{op_name}",
        batch_size=batch,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=test.replace("/", "_"),
    )
