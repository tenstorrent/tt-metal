# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from models.perf.device_perf_utils import run_device_perf_detailed

THRESHOLD = 0.4


# Override the autouse ensure_devices fixture to prevent device initialization
# in the parent process, which would conflict with the child pytest process
@pytest.fixture(autouse=True)
def ensure_devices():
    """Skip device initialization since this test spawns child pytest processes."""


# Override galaxy_type fixture to avoid calling ttnn.cluster.get_cluster_type()
# which might acquire device locks
@pytest.fixture(scope="function")
def galaxy_type():
    """Return galaxy type without initializing devices."""
    import os

    # Use environment variable if set, otherwise default to "4U"
    return os.environ.get("GALAXY_TYPE", "4U")


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
    command = f"pytest tests/ttnn/unit_tests/operations/ccl/test_minimals.py::test_6u_trace_rms_fuse_deepseek"
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


@pytest.mark.parametrize(
    "test_id, perf_target_us",
    [
        ("deepseek_embedding_ag", 20),  # All-gather in embedding: [1,1,32,896] per device
        ("deepseek_rms_norm_ag", 18),  # All-gather in RMS norm stats: [1,1,32,32] per device
        ("deepseek_mla_ag", 25),  # All-gather in MLA: various shapes
        ("deepseek_mlp_rs", 22),  # Reduce-scatter in MLP: [1,1,32,896] per device
        ("deepseek_moe_rs", 28),  # Reduce-scatter in MoE: larger shapes
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_deepseek_ccl_shapes_perf(
    test_id,
    perf_target_us,
    galaxy_type,
):
    """Performance tests for DeepSeek V3 CCL operation shapes."""
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"{test_id}_test"

    subdir = "deepseek_ccl_perf"

    # Map test_id to the actual test function in test_minimals.py
    test_map = {
        "deepseek_embedding_ag": "test_deepseek_embedding_all_gather",
        "deepseek_rms_norm_ag": "test_deepseek_rms_norm_all_gather",
        "deepseek_mla_ag": "test_deepseek_mla_all_gather",
        "deepseek_mlp_rs": "test_deepseek_mlp_reduce_scatter",
        "deepseek_moe_rs": "test_deepseek_moe_reduce_scatter",
    }

    command = f"pytest tests/ttnn/unit_tests/operations/ccl/test_minimals.py::{test_map[test_id]}"
    cols = ["DEVICE KERNEL"]

    # Determine operation name from test_id
    if "_ag" in test_id:
        op_name = "AllGatherAsync"
    elif "_rs" in test_id:
        op_name = "ReduceScatterMinimalAsync"
    else:
        op_name = "CCLOp"

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
        run_type=f"tg_deepseek_ops" if galaxy_type != "6U" else "tg_deepseek_ops_6U",
        ml_model_name="deepseek-v3-tg",
    )

    assert measured_avg_us < perf_target_us, f"Performance target not met: {measured_avg_us} us > {perf_target_us} us"
