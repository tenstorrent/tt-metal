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
def test_ag_tg_llama_perf(
    ag_type,
    warmup_iters,
    perf_target_us,
    galaxy_type,
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"all_gather_{ag_type}"

    subdir = "llama_ccl_perf"
    if galaxy_type == "6U":
        command = f"pytest tests/ttnn/unit_tests/operations/ccl/test_ccl_async_TG_llama.py::test_all_gather_6u_llama -k {ag_type}"
    else:
        command = f"pytest tests/ttnn/unit_tests/operations/ccl/test_ccl_async_TG_llama.py::test_all_gather_tg_llama -k {ag_type}"
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
        run_type=f"tg_llama_ops" if galaxy_type != "6U" else "tg_llama_ops_6U",
        ml_model_name="llama70b-tg",
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
def test_ar_tg_llama_perf(
    ar_type,
    warmup_iters,
    perf_target_us,
    galaxy_type,
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"all_reduce_{ar_type}"

    subdir = "llama_ccl_perf"
    if galaxy_type == "6U":
        command = f"pytest tests/ttnn/unit_tests/operations/ccl/test_ccl_async_TG_llama.py::test_all_reduce_6U_llama -k {ar_type}"
    else:
        command = f"pytest tests/ttnn/unit_tests/operations/ccl/test_ccl_async_TG_llama.py::test_all_reduce_tg_llama -k {ar_type}"
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
        run_type=f"tg_llama_ops" if galaxy_type != "6U" else "tg_llama_ops_6U",
        ml_model_name="llama70b-tg",
    )

    assert (
        measured_avg_us < perf_target_us + THRESHOLD
    ), f"Performance target not met: {measured_avg_us} us > {perf_target_us} us"


@pytest.mark.parametrize(
    "ar_type, warmup_iters, perf_target_us",
    [
        ("Matmul_RS", 10, 25),
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_matmul_rs(
    ar_type,
    warmup_iters,
    perf_target_us,
    galaxy_type,
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"matmul_rs_test"

    subdir = "llama_ccl_perf"
    if galaxy_type == "6U":
        command = f"pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_rs_matmul_1d_gather_in0.py::test_6U_matmul_1d_ring_llama_with_rs_perf"
    else:
        command = f"pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_rs_matmul_1d_gather_in0.py::test_tg_matmul_1d_ring_llama_with_rs_perf"
    cols = ["DEVICE KERNEL"]
    op_name = "Matmul_RS"
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
        run_type=f"tg_llama_ops" if galaxy_type != "6U" else "tg_llama_ops_6U",
        ml_model_name="llama70b-tg",
    )

    assert measured_avg_us < perf_target_us, f"Performance target not met: {measured_avg_us} us > {perf_target_us} us"


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

    subdir = "llama_ccl_perf"
    if galaxy_type == "6U":
        command = f"pytest tests/ttnn/unit_tests/operations/ccl/test_minimals.py::test_6u_trace_rms_fuse"
    else:
        command = f"pytest tests/ttnn/unit_tests/operations/ccl/test_minimals.py::test_tg_trace_rms_fuse"
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
        run_type=f"tg_llama_ops" if galaxy_type != "6U" else "tg_llama_ops_6U",
        ml_model_name="llama70b-tg",
    )

    assert measured_avg_us < perf_target_us, f"Performance target not met: {measured_avg_us} us > {perf_target_us} us"


@pytest.mark.skip(reason="Skip test due to issue: https://github.com/tenstorrent/tt-metal/issues/24630")
@pytest.mark.parametrize(
    "warmup_iters, perf_target_us",
    [
        (5, 17),
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_fused_all_gather_concat_perf(
    warmup_iters,
    perf_target_us,
    galaxy_type,
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"all_gather_concat_heads"

    subdir = "llama_ccl_perf"
    if galaxy_type == "6U":
        command = f"pytest tests/ttnn/unit_tests/operations/ccl/test_minimals.py::test_concat_fuse_6u"
    else:
        command = f"pytest tests/ttnn/unit_tests/operations/ccl/test_minimals.py::test_concat_fuse"
    cols = ["DEVICE KERNEL"]
    op_name = "AllGatherConcat"
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
        run_type=f"tg_llama_ops" if galaxy_type != "6U" else "tg_llama_ops_6U",
        ml_model_name="llama70b-tg",
    )

    assert measured_avg_us < perf_target_us, f"Performance target not met: {measured_avg_us} us > {perf_target_us} us"


@pytest.mark.parametrize(
    "warmup_iters, perf_target_us",
    [
        (5, 14.6),
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_fused_all_reduce_create_heads_perf(
    warmup_iters,
    perf_target_us,
    galaxy_type,
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"all_reduce_create_heads"

    subdir = "llama_ccl_perf"
    if galaxy_type == "6U":
        command = f"pytest tests/ttnn/unit_tests/operations/ccl/test_qkv_all_reduce_minimal.py::test_all_reduce_qkv_heads_fuse_perf_6U"
    else:
        command = f"pytest tests/ttnn/unit_tests/operations/ccl/test_qkv_all_reduce_minimal.py::test_all_reduce_qkv_heads_fuse_perf"
    cols = ["DEVICE KERNEL"]
    op_name = "AllReduceCreateQkvHeads"
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
        run_type=f"tg_llama_ops" if galaxy_type != "6U" else "tg_llama_ops_6U",
        ml_model_name="llama70b-tg",
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

    subdir = "llama_ccl_perf"
    if galaxy_type == "6U":
        command = f"pytest tests/ttnn/unit_tests/operations/ccl/test_llama_reduce_scatter_async_6U.py::test_fabric_reduce_scatter_tg_trace_6u"
    else:
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
        run_type=f"tg_llama_ops" if galaxy_type != "6U" else "tg_llama_ops_6U",
        ml_model_name="llama70b-tg",
    )

    assert measured_avg_us < perf_target_us, f"Performance target not met: {measured_avg_us} us > {perf_target_us} us"


@pytest.mark.parametrize(
    "warmup_iters, perf_target_us",
    [
        (5, 9.6),
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_rs_create_heads_perf(
    warmup_iters,
    perf_target_us,
    galaxy_type,
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"rs_create_heads_perf"

    subdir = "llama_ccl_perf"
    if galaxy_type == "6U":
        command = f"pytest tests/ttnn/unit_tests/operations/ccl/test_llama_reduce_scatter_create_heads_async_TG.py::test_rs_create_heads_6u_trace"
    else:
        command = f"pytest tests/ttnn/unit_tests/operations/ccl/test_llama_reduce_scatter_create_heads_async_TG.py::test_rs_create_heads_tg_trace"
    cols = ["DEVICE KERNEL"]
    op_name = "LlamaReduceScatterCreateHeadsDeviceOperation"

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
        run_type=f"tg_llama_ops" if galaxy_type != "6U" else "tg_llama_ops_6U",
        ml_model_name="llama70b-tg",
    )

    assert measured_avg_us < perf_target_us, f"Performance target not met: {measured_avg_us} us > {perf_target_us} us"
