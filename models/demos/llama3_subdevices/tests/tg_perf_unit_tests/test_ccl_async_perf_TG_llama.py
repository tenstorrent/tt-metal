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
        ("sdpa_bfp8_linear", 15, 12.9),
        # ("sdpa_bfp8_ring", 15, 12.9),
        ("sdpa_bf16_linear", 15, 12.9),
        # ("sdpa_bf16_ring", 15, 12.9),
        ("binary_mult_bfp8_linear", 15, 12.54),
        # ("binary_mult_bfp8_ring", 15, 12.54),
        ("binary_mult_bf16_linear", 15, 12.54),
        # ("binary_mult_bf16_ring", 15, 12.54),
        ("layernorm_bfp8_linear", 15, 5.4),
        # ("layernorm_bfp8_ring", 15, 5.4),
        ("layernorm_bf16_linear", 15, 5.4),
        # ("layernorm_bf16_ring", 15, 5.4),
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_ag_tg_llama_perf(
    ag_type,
    warmup_iters,
    perf_target_us,
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"all_gather_{ag_type}"

    subdir = "llama_ccl_perf"
    command = (
        f"pytest tests/ttnn/unit_tests/operations/ccl/test_ccl_async_TG_llama.py::test_all_gather_tg_llama -k {ag_type}"
    )
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
    print(f"{ag_type} : Measured performance: {measured_avg_us:.3f} us vs. target: {perf_target_us} us")

    # Save the measurement
    benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-{ag_type}-min", measured_min)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-{ag_type}-max", measured_max)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-{ag_type}-avg", measured_avg)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-{ag_type}-std", measured_std)
    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"tg_llama_ops",
        ml_model_name="llama70b-tg",
    )

    assert (
        measured_avg_us < perf_target_us + THRESHOLD
    ), f"Performance target not met: {measured_avg_us} us > {perf_target_us} us"


@pytest.mark.parametrize(
    "ag_type, warmup_iters, perf_target_us",
    [
        ("4_1_1_32_32_128_1_Linear_bfloat16", 0, 1000),
        ("4_2_1_32_32_128_1_Linear_bfloat16", 0, 1000),
        # ("4_2_1_32_32_128_1_Linear_bfp8_b", 0, 1000),
        ("4_4_1_32_32_128_1_Linear_bfloat16", 0, 1000),
        # ("4_4_1_32_32_128_1_Linear_bfp8_b", 0, 1000),
        # ("8_2_1_32_32_128_1_Linear_bfloat16", 0, 1000),
        # ("8_2_1_32_32_128_1_Linear_bfp8_b", 0, 1000),
        # ("8_4_1_32_32_128_1_Linear_bfloat16", 0, 1000),
        # ("8_4_1_32_32_128_1_Linear_bfp8_b", 0, 1000),
        # ("8_1_8_1_32_1280_0_Linear_bfloat16", 0, 1000),
        # ("8_1_8_1_32_1280_0_Linear_bfp8_b", 0, 1000),
        # ("8_1_8_1_64_1280_0_Linear_bfloat16", 0, 1000),
        # ("8_1_8_1_64_1280_0_Linear_bfp8_b", 0, 1000),
        # ("8_1_8_1_128_1280_0_Linear_bfloat16", 0, 1000),
        # ("8_1_8_1_128_1280_0_Linear_bfp8_b", 0, 1000),
        # ("8_2_8_1_32_1280_0_Linear_bfloat16", 0, 1000),
        # ("8_2_8_1_32_1280_0_Linear_bfp8_b", 0, 1000),
        # ("8_2_8_1_64_1280_0_Linear_bfloat16", 0, 1000),
        # ("8_2_8_1_64_1280_0_Linear_bfp8_b", 0, 1000),
        # ("8_2_8_1_128_1280_0_Linear_bfloat16", 0, 1000),
        # ("8_2_8_1_128_1280_0_Linear_bfp8_b", 0, 1000),
        # ("8_4_8_1_32_1280_0_Linear_bfloat16", 0, 1000),
        # ("8_4_8_1_32_1280_0_Linear_bfp8_b", 0, 1000),
        # ("8_4_8_1_64_1280_0_Linear_bfloat16", 0, 1000),
        # ("8_4_8_1_64_1280_0_Linear_bfp8_b", 0, 1000),
        # ("8_4_8_1_128_1280_0_Linear_bfloat16", 0, 1000),
        # ("8_4_8_1_128_1280_0_Linear_bfp8_b", 0, 1000),
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_ag_tg_test_perf(
    ag_type,
    warmup_iters,
    perf_target_us,
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"all_gather_{ag_type}"

    subdir = "test_ccl_perf"
    command = f"pytest tests/ttnn/unit_tests/operations/ccl/perf/test_ccl_async_perf.py::test_all_gather_async_tg -k {ag_type}"
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
    print(f"{ag_type} : Measured performance: {measured_avg_us:.3f} us vs. target: {perf_target_us} us")

    # Save the measurement
    benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-{ag_type}-min", measured_min)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-{ag_type}-max", measured_max)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-{ag_type}-avg", measured_avg)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-{ag_type}-std", measured_std)
    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"tg_llama_ops",
        ml_model_name="llama70b-tg",
    )

    assert (
        measured_avg_us < perf_target_us + THRESHOLD
    ), f"Performance target not met: {measured_avg_us} us > {perf_target_us} us"


@pytest.mark.parametrize(
    "ar_type, warmup_iters, perf_target_us",
    [
        ("ff2_bfp8_linear", 15, 18.6),
        # ("ff2_bfp8_ring", 15, 18.6),
        ("ff2_bf16_linear", 15, 18.6),
        # ("ff2_bf16_ring", 15, 18.6),
        ("qkv_bfp8_linear", 15, 11.9),
        # ("qkv_bfp8_ring", 15, 11.9),
        ("qkv_bf16_linear", 15, 11.9),
        # ("qkv_bf16_ring", 15, 11.9),
        ("ff1_bfp8_linear", 15, 19.2),
        # ("ff1_bfp8_ring", 15, 19.2),
        ("ff1_bf16_linear", 15, 19.2),
        # ("ff1_bf16_ring", 15, 19.2),
        ("lm_head_bfp8_linear", 15, 61.8),
        # ("lm_head_bfp8_ring", 15, 61.8),
        ("lm_head_bf16_linear", 15, 61.8),
        # ("lm_head_bf16_ring", 15, 61.8),
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_ar_tg_llama_perf(
    ar_type,
    warmup_iters,
    perf_target_us,
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"all_reduce_{ar_type}"

    subdir = "llama_ccl_perf"
    command = (
        f"pytest tests/ttnn/unit_tests/operations/ccl/test_ccl_async_TG_llama.py::test_all_reduce_tg_llama -k {ar_type}"
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
    print(f"{ar_type} : Measured performance: {measured_avg_us:.3f} us vs. target: {perf_target_us} us")

    # Save the measurement
    benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-{ar_type}-min", measured_min)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-{ar_type}-max", measured_max)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-{ar_type}-avg", measured_avg)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-{ar_type}-std", measured_std)
    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"tg_llama_ops",
        ml_model_name="llama70b-tg",
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
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"rms_test"

    subdir = "llama_ccl_perf"
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
        run_type=f"tg_llama_ops",
        ml_model_name="llama70b-tg",
    )

    assert measured_avg_us < perf_target_us, f"Performance target not met: {measured_avg_us} us > {perf_target_us} us"


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
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"all_gather_concat_heads"

    subdir = "llama_ccl_perf"
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
        run_type=f"tg_llama_ops",
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
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"all_reduce_create_heads"

    subdir = "llama_ccl_perf"
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
        run_type=f"tg_llama_ops",
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
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"reduce_scatter_perf"

    subdir = "llama_ccl_perf"
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
        run_type=f"tg_llama_ops",
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
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"rs_create_heads_perf"

    subdir = "llama_ccl_perf"
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
        run_type=f"tg_llama_ops",
        ml_model_name="llama70b-tg",
    )

    assert measured_avg_us < perf_target_us, f"Performance target not met: {measured_avg_us} us > {perf_target_us} us"
