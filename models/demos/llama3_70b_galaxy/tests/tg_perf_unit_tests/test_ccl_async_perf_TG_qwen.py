# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from models.perf.device_perf_utils import run_device_perf_detailed

THRESHOLD = 0.4


@pytest.mark.parametrize(
    "ag_type, warmup_iters, perf_target_us",
    [
        ("binary_mult", 15, 12.54),
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_ag_tg_qwen_perf(
    ag_type,
    warmup_iters,
    perf_target_us,
    galaxy_type,
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"all_gather_{ag_type}"

    subdir = "qwen_ccl_perf"
    if galaxy_type == "6U":
        command = f"pytest tests.ttnn/unit_tests/operations/ccl/test_ccl_async_TG_qwen.py::test_all_gather_6u_qwen -k {ag_type}"
    cols = ["DEVICE KERNEL"]
    op_name = "AllGatherAsync"
    warmup_iters = warmup_iters * 32  # 5 iterations per device

    profiler.start("run")
    profiler.start(step_name)
    results = run_device_perf_detailed(command, subdir, cols, op_name, has_signposts=True, warmup_iters=warmup_iters)
    profiler.end(step_name)
    profiler.end("run")

    # Get the measured performance
    measured_min = results[cols[0]]["MIN"]
    measured_max = results[cols[0]]["MAX"]
    measured_avg = results[cols[0]]["AVG"]
    measured_std = results[cols[0]]["STD"]
    measured_avg_us = measured_avg / 1000

    logger.info(f"Measured performance: {measured_avg_us:.3f} us vs. target: {perf_target_us} us")

    # Save the measurement
    benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-{ag_type}-min", measured_min)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-{ag_type}-max", measured_max)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-{ag_type}-avg", measured_avg)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-{ag_type}-std", measured_std)
    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"tg_qwen_ops",
        ml_model_name="qwen3-32b-tg",
    )

    assert (
        measured_avg_us < perf_target_us + THRESHOLD
    ), f"Performance target not met: {measured_avg_us} us > {perf_target_us} us"


@pytest.mark.parametrize(
    "ar_type, warmup_iters, perf_target_us",
    [
        ("ff2", 15, 19.0),
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_ar_tg_qwen_perf(
    ar_type,
    warmup_iters,
    perf_target_us,
    galaxy_type,
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"all_reduce_{ar_type}"

    subdir = "qwen_ccl_perf"
    command = (
        f"pytest tests/ttnn/unit_tests/operations/ccl/test_ccl_async_TG_qwen.py::test_all_reduce_6U_qwen -k {ar_type}"
    )
    cols = ["DEVICE KERNEL"]
    op_name = "AllReduceAsync"
    warmup_iters = warmup_iters * 32  # 5 iterations per device

    profiler.start("run")
    profiler.start(step_name)
    results = run_device_perf_detailed(command, subdir, cols, op_name, has_signposts=True, warmup_iters=warmup_iters)
    profiler.end(step_name)
    profiler.end("run")

    # Get the measured performance
    measured_min = results[cols[0]]["MIN"]
    measured_max = results[cols[0]]["MAX"]
    measured_avg = results[cols[0]]["AVG"]
    measured_std = results[cols[0]]["STD"]
    measured_avg_us = measured_avg / 1000

    logger.info(f"Measured performance: {measured_avg_us:.3f} us vs. target: {perf_target_us} us")

    # Save the measurement
    benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-{ar_type}-min", measured_min)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-{ar_type}-max", measured_max)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-{ar_type}-avg", measured_avg)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-{ar_type}-std", measured_std)
    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"tg_qwen_ops",
        ml_model_name="qwen3-32b-tg",
    )

    assert (
        measured_avg_us < perf_target_us + THRESHOLD
    ), f"Performance target not met: {measured_avg_us} us > {perf_target_us} us"


@pytest.mark.parametrize(
    "ar_type, warmup_iters, perf_target_us",
    [
        ("rms", 10, 25),
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_rms_perf(
    ar_type,
    warmup_iters,
    perf_target_us,
    galaxy_type,
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"rms_test"

    subdir = "qwen_ccl_perf"
    command = f"pytest tests/ttnn/unit_tests/operations/ccl/test_qwen_fuse_rms_TG.py::test_tg_trace_rms_fuse"
    cols = ["DEVICE KERNEL"]
    op_name = "RMSAllGather"
    warmup_iters = warmup_iters * 32  # 5 iterations per device

    profiler.start("run")
    profiler.start(step_name)
    results = run_device_perf_detailed(command, subdir, cols, op_name, has_signposts=True, warmup_iters=0)
    profiler.end(step_name)
    profiler.end("run")

    # Get the measured performance
    measured_min = results[cols[0]]["MIN"]
    measured_max = results[cols[0]]["MAX"]
    measured_avg = results[cols[0]]["AVG"]
    measured_std = results[cols[0]]["STD"]
    measured_avg_us = measured_avg / 1000

    logger.info(f"Measured performance: {measured_avg_us:.3f} us vs. target: {perf_target_us} us")

    # Save the measurement
    benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-min", measured_min)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-max", measured_max)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-avg", measured_avg)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-std", measured_std)
    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"tg_qwen_ops",
        ml_model_name="qwen3-32b-tg",
    )

    assert measured_avg_us < perf_target_us, f"Performance target not met: {measured_avg_us} us > {perf_target_us} us"


@pytest.mark.parametrize(
    "warmup_iters, perf_target_us",
    [
        (5, 9.7),
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_reduce_scatter_perf(
    warmup_iters,
    perf_target_us,
    galaxy_type,
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"reduce_scatter_perf"

    # All Qwen3-32b reduce scatters have the same configuration as Llama70b, so we just call the same tests
    subdir = "qwen_ccl_perf"
    command = f"pytest tests/ttnn/unit_tests/operations/ccl/test_llama_reduce_scatter_async_TG.py::test_fabric_reduce_scatter_tg_trace"
    cols = ["DEVICE KERNEL"]
    op_name = "LlamaReduceScatterDeviceOperation"

    profiler.start("run")
    profiler.start(step_name)
    results = run_device_perf_detailed(command, subdir, cols, op_name, has_signposts=True, warmup_iters=warmup_iters)
    profiler.end(step_name)
    profiler.end("run")

    # Get the measured performance
    measured_min = results[cols[0]]["MIN"]
    measured_max = results[cols[0]]["MAX"]
    measured_avg = results[cols[0]]["AVG"]
    measured_std = results[cols[0]]["STD"]
    measured_avg_us = measured_avg / 1000

    logger.info(f"Measured performance: {measured_avg_us:.3f} us vs. target: {perf_target_us} us")

    # Save the measurement
    benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-min", measured_min)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-max", measured_max)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-avg", measured_avg)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-std", measured_std)
    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"tg_qwen_ops",
        ml_model_name="qwen3-32b-tg",
    )

    assert measured_avg_us < perf_target_us, f"Performance target not met: {measured_avg_us} us > {perf_target_us} us"


@pytest.mark.parametrize(
    "warmup_iters, perf_target_us",
    [
        (5, 16.6),
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_ag_matmul_tg_qwen_perf(
    warmup_iters,
    perf_target_us,
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"all_gather_matmul"

    subdir = "qwen_ccl_perf"
    command = f"pytest tests/ttnn/unit_tests/operations/ccl/test_ccl_async_TG_qwen.py::test_qwen_all_gather_matmul"
    cols = ["DEVICE KERNEL"]
    op_name = "QwenAllGatherMatmulAsync"
    warmup_iters = warmup_iters * 32  # 5 iterations per device

    profiler.start("run")
    profiler.start(step_name)
    results = run_device_perf_detailed(command, subdir, cols, op_name, has_signposts=True, warmup_iters=warmup_iters)
    profiler.end(step_name)
    profiler.end("run")

    # Get the measured performance
    measured_min = results[cols[0]]["MIN"]
    measured_max = results[cols[0]]["MAX"]
    measured_avg = results[cols[0]]["AVG"]
    measured_std = results[cols[0]]["STD"]
    measured_avg_us = measured_avg / 1000

    logger.info(f"Measured performance: {measured_avg_us:.3f} us vs. target: {perf_target_us} us")

    # Save the measurement
    benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-min", measured_min)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-max", measured_max)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-avg", measured_avg)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-std", measured_std)
    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"tg_qwen_ops",
        ml_model_name="qwen3-32b-tg",
    )

    assert (
        measured_avg_us < perf_target_us + THRESHOLD
    ), f"Performance target not met: {measured_avg_us} us > {perf_target_us} us"
