# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
from loguru import logger

from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from models.perf.device_perf_utils import run_device_perf_detailed

os.environ.setdefault("MESH_DEVICE", "TG")

THRESHOLD = 0.5
THRESHOLD_PERCENTAGE = 0.03


@pytest.fixture(autouse=True)
def ensure_devices():
    """Skip device initialization since this test spawns child pytest processes."""


# Override galaxy_type fixture to avoid calling ttnn.cluster.get_cluster_type()
# which might acquire device locks
@pytest.fixture(scope="function")
def galaxy_type():
    """Return galaxy type without initializing devices."""

    # Use environment variable if set, otherwise default to "TG"
    return os.environ.get("GALAXY_TYPE", "TG")


@pytest.mark.parametrize(
    "step_name, warmup_iters, perf_target_us",
    [
        ("wq_kv_a", 10, 31.5),
        ("wq_b", 10, 62.5),
        ("wkv_b1", 10, 39.3),
        ("wkv_b2", 10, 190.7),
        ("wo", 10, 259.56),
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_linear_deepseek_perf(
    step_name,
    warmup_iters,
    perf_target_us,
    galaxy_type,
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()

    subdir = "deepseek_linear_perf"
    command = f"pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_linear_deepseek.py -k {step_name}"
    cols = ["DEVICE KERNEL"]
    op_name = "MatmulDeviceOperation"

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
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-min", measured_min)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-max", measured_max)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-avg", measured_avg)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-std", measured_std)
    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"tg_deepseek_linear",
        ml_model_name="deepseek-v3-tg",
    )

    threshold = max(THRESHOLD, perf_target_us * THRESHOLD_PERCENTAGE)

    assert (
        measured_avg_us < perf_target_us + threshold
    ), f"Performance is worse than target: {measured_avg_us} us > {perf_target_us} us, the threshold was {threshold} us"
    assert (
        measured_avg_us > perf_target_us - threshold
    ), f"Performance is more than {threshold} us better than target, update the target: {measured_avg_us} us < {perf_target_us} us"


@pytest.mark.parametrize(
    "step_name, warmup_iters, perf_target_us",
    [
        ("mla_reduce_wq_kv_a", 10, 7.73),  # Target based on typical reduce performance
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_fast_reduce_wq_kv_a_deepseek_perf(
    step_name,
    warmup_iters,
    perf_target_us,
    galaxy_type,
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()

    subdir = "deepseek_ccl_perf"
    command = f"pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_reduce_deepseek.py"
    cols = ["DEVICE KERNEL"]
    op_name = "FastReduceNCDeviceOperation"

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
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-min", measured_min)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-max", measured_max)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-avg", measured_avg)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-std", measured_std)
    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"tg_deepseek_ccl_ops",
        ml_model_name="deepseek-v3-tg",
    )

    threshold = max(THRESHOLD, perf_target_us * THRESHOLD_PERCENTAGE)

    assert (
        measured_avg_us < perf_target_us + threshold
    ), f"Performance is worse than target: {measured_avg_us} us > {perf_target_us} us, the threshold was {threshold} us"
    assert (
        measured_avg_us > perf_target_us - threshold
    ), f"Performance is more than {threshold} us better than target, update the target: {measured_avg_us} us < {perf_target_us} us"


@pytest.mark.parametrize(
    "step_name, warmup_iters, perf_target_us",
    [
        ("wq_kv_a_ag_decode", 10, 47.91),  # Target based on typical all-gather performance
        ("wo_ag_decode", 10, 50.19),  # Target based on typical all-gather performance
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_ag_tg_deepseek_allgather_perf(
    step_name,
    warmup_iters,
    perf_target_us,
    galaxy_type,
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()

    subdir = "deepseek_ccl_perf"
    command = f"pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_allgather_deepseek.py -k {step_name}"
    cols = ["DEVICE KERNEL"]
    op_name = "AllGatherAsyncDeviceOperation"  # CCL operation name for all-gather
    warmup_iters = warmup_iters * 32  # Multiply by number of devices (32 for TG)

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
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-min", measured_min)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-max", measured_max)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-avg", measured_avg)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-std", measured_std)
    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"tg_deepseek_ccl_ops",
        ml_model_name="deepseek-v3-tg",
    )

    threshold = max(THRESHOLD, perf_target_us * THRESHOLD_PERCENTAGE)

    assert (
        measured_avg_us < perf_target_us + threshold
    ), f"Performance is worse than target: {measured_avg_us} us > {perf_target_us} us, the threshold was {threshold} us"
    assert (
        measured_avg_us > perf_target_us - threshold
    ), f"Performance is more than {threshold} us better than target, update the target: {measured_avg_us} us < {perf_target_us} us"


@pytest.mark.parametrize(
    "step_name, warmup_iters, perf_target_us",
    [
        ("q_slice", 10, 1.47),
        ("kv_nope_slice", 10, 1.26),
        ("kv_rope_slice", 10, 0.95),
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_slice_deepseek_perf(
    step_name,
    warmup_iters,
    perf_target_us,
    galaxy_type,
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()

    subdir = "deepseek_ccl_perf"
    command = f"pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_slice_deepseek.py -k {step_name}"
    cols = ["DEVICE KERNEL"]
    op_name = "SliceDeviceOperation"

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
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-min", measured_min)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-max", measured_max)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-avg", measured_avg)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-std", measured_std)
    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"tg_deepseek_ccl_ops",
        ml_model_name="deepseek-v3-tg",
    )

    threshold = max(THRESHOLD, perf_target_us * THRESHOLD_PERCENTAGE)

    assert (
        measured_avg_us < perf_target_us + threshold
    ), f"Performance is worse than target: {measured_avg_us} us > {perf_target_us} us, the threshold was {threshold} us"
    assert (
        measured_avg_us > perf_target_us - threshold
    ), f"Performance is more than {threshold} us better than target, update the target: {measured_avg_us} us < {perf_target_us} us"


@pytest.mark.parametrize(
    "step_name, warmup_iters, perf_target_us",
    [
        ("q_nope_slice", 10, 2.27),  # 2.29 if run on mesh device
        ("q_rope_slice", 10, 1.84),  # 1.86 if run on mesh device
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_slice_q_rope_nope_deepseek_perf(
    step_name,
    warmup_iters,
    perf_target_us,
    galaxy_type,
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()

    subdir = "deepseek_ccl_perf"
    command = f"pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_slice_q_rope_nope.py -k {step_name}"
    # you can also use mesh device version
    # command = f"pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_slice_q_rope_nope_mesh.py -k {step_name}"
    cols = ["DEVICE KERNEL"]
    op_name = "SliceDeviceOperation"

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
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-min", measured_min)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-max", measured_max)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-avg", measured_avg)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-std", measured_std)
    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"tg_deepseek_ccl_ops",
        ml_model_name="deepseek-v3-tg",
    )

    threshold = max(THRESHOLD, perf_target_us * THRESHOLD_PERCENTAGE)

    assert (
        measured_avg_us < perf_target_us + threshold
    ), f"Performance is worse than target: {measured_avg_us} us > {perf_target_us} us, the threshold was {threshold} us"
    assert (
        measured_avg_us > perf_target_us - threshold
    ), f"Performance is more than {threshold} us better than target, update the target: {measured_avg_us} us < {perf_target_us} us"


@pytest.mark.parametrize(
    "step_name, warmup_iters, perf_target_us",
    [
        ("q_norm", 10, 7.33),
        ("kv_norm", 10, 6.68),
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_rmsnorm_deepseek_perf(
    step_name,
    warmup_iters,
    perf_target_us,
    galaxy_type,
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()

    subdir = "deepseek_ccl_perf"
    command = f"pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_rmsnorm_deepseek.py -k {step_name}"
    cols = ["DEVICE KERNEL"]
    op_name = "LayerNormDeviceOperation"

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
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-min", measured_min)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-max", measured_max)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-avg", measured_avg)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-std", measured_std)
    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"tg_deepseek_ccl_ops",
        ml_model_name="deepseek-v3-tg",
    )

    threshold = max(THRESHOLD, perf_target_us * THRESHOLD_PERCENTAGE)

    assert (
        measured_avg_us < perf_target_us + threshold
    ), f"Performance is worse than target: {measured_avg_us} us > {perf_target_us} us, the threshold was {threshold} us"
    assert (
        measured_avg_us > perf_target_us - threshold
    ), f"Performance is more than {threshold} us better than target, update the target: {measured_avg_us} us < {perf_target_us} us"


@pytest.mark.parametrize(
    "step_name, warmup_iters, perf_target_us, op_name",
    [
        ("kv_nope_to_interleaved", 10, 0.75, "ShardedToInterleavedDeviceOperation"),
        ("kv_rope_reshard", 10, 1.04, "InterleavedToShardedDeviceOperation"),
        ("kv_rope_out_reshard", 10, 1.19, "ShardedToInterleavedDeviceOperation"),
        ("kvpe_reshard", 10, 2.83, "InterleavedToShardedDeviceOperation"),
        ("q_rope_out_reshard", 10, 1.19, "ShardedToInterleavedDeviceOperation"),
        ("flash_mla_reshard", 10, 9.41, "InterleavedToShardedDeviceOperation"),
        ("flash_mla_out_reshard", 10, 3.89, "ShardedToInterleavedDeviceOperation"),
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_to_memory_config_deepseek_perf(
    step_name,
    warmup_iters,
    perf_target_us,
    galaxy_type,
    op_name,
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()

    subdir = "deepseek_ccl_perf"
    command = f"pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_to_memory_config_deepseek.py -k {step_name}"
    cols = ["DEVICE KERNEL"]

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
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-min", measured_min)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-max", measured_max)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-avg", measured_avg)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-std", measured_std)
    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"tg_deepseek_ccl_ops",
        ml_model_name="deepseek-v3-tg",
    )

    threshold = max(THRESHOLD, perf_target_us * THRESHOLD_PERCENTAGE)

    assert (
        measured_avg_us < perf_target_us + threshold
    ), f"Performance is worse than target: {measured_avg_us} us > {perf_target_us} us, the threshold was {threshold} us"
    assert (
        measured_avg_us > perf_target_us - threshold
    ), f"Performance is more than {threshold} us better than target, update the target: {measured_avg_us} us < {perf_target_us} us"


@pytest.mark.parametrize(
    "step_name, warmup_iters, perf_target_us",
    [
        ("kv_rope_permute_pre", 10, 5.52),
        ("kv_rope_permute_post", 10, 1.30),
        ("kvpe_permute", 10, 30.95),
        ("q_nope_permute_pre_linear", 10, 6.23),
        ("q_nope_permute_post_linear", 10, 16.47),
        ("q_rope_permute", 10, 1.62),
        ("attn_out_permute_pre_linear", 10, 31.94),
        ("v_out_permute", 10, 7.61),
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_permute_deepseek_perf(
    step_name,
    warmup_iters,
    perf_target_us,
    galaxy_type,
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()

    subdir = "deepseek_ccl_perf"
    command = f"pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_permute_deepseek.py -k {step_name}"
    cols = ["DEVICE KERNEL"]
    op_name = "TransposeDeviceOperation" if step_name != "q_nope_permute_pre_linear" else "PermuteDeviceOperation"

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
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-min", measured_min)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-max", measured_max)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-avg", measured_avg)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-std", measured_std)
    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"tg_deepseek_ccl_ops",
        ml_model_name="deepseek-v3-tg",
    )

    threshold = max(THRESHOLD, perf_target_us * THRESHOLD_PERCENTAGE)

    assert (
        measured_avg_us < perf_target_us + threshold
    ), f"Performance is worse than target: {measured_avg_us} us > {perf_target_us} us, the threshold was {threshold} us"
    assert (
        measured_avg_us > perf_target_us - threshold
    ), f"Performance is more than {threshold} us better than target, update the target: {measured_avg_us} us < {perf_target_us} us"


@pytest.mark.parametrize(
    "step_name, warmup_iters, perf_target_us",
    [
        ("kv_rope_decode", 10, 4.18),
        ("q_rope_decode", 10, 4.16),
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_rope_deepseek_perf(
    step_name,
    warmup_iters,
    perf_target_us,
    galaxy_type,
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()

    subdir = "deepseek_ccl_perf"
    command = f"pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_rope_deepseek.py -k {step_name}"
    cols = ["DEVICE KERNEL"]
    op_name = "RotaryEmbeddingLlamaDeviceOperation"

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
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-min", measured_min)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-max", measured_max)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-avg", measured_avg)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-std", measured_std)
    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"tg_deepseek_ccl_ops",
        ml_model_name="deepseek-v3-tg",
    )

    threshold = max(THRESHOLD, perf_target_us * THRESHOLD_PERCENTAGE)

    assert (
        measured_avg_us < perf_target_us + threshold
    ), f"Performance is worse than target: {measured_avg_us} us > {perf_target_us} us, the threshold was {threshold} us"
    assert (
        measured_avg_us > perf_target_us - threshold
    ), f"Performance is more than {threshold} us better than target, update the target: {measured_avg_us} us < {perf_target_us} us"


@pytest.mark.parametrize(
    "step_name, warmup_iters, perf_target_us",
    [
        ("kvpe_concat", 10, 1.32),
        ("q_concat", 10, 8.01),
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_concat_deepseek_perf(
    step_name,
    warmup_iters,
    perf_target_us,
    galaxy_type,
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()

    subdir = "deepseek_ccl_perf"
    command = f"pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_concat_deepseek.py -k {step_name}"
    cols = ["DEVICE KERNEL"]
    op_name = "ConcatDeviceOperation"

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
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-min", measured_min)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-max", measured_max)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-avg", measured_avg)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-std", measured_std)
    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"tg_deepseek_ccl_ops",
        ml_model_name="deepseek-v3-tg",
    )

    threshold = max(THRESHOLD, perf_target_us * THRESHOLD_PERCENTAGE)

    assert (
        measured_avg_us < perf_target_us + threshold
    ), f"Performance is worse than target: {measured_avg_us} us > {perf_target_us} us, the threshold was {threshold} us"
    assert (
        measured_avg_us > perf_target_us - threshold
    ), f"Performance is more than {threshold} us better than target, update the target: {measured_avg_us} us < {perf_target_us} us"


@pytest.mark.parametrize(
    "step_name, warmup_iters, perf_target_us",
    [
        ("kvpe_pad", 10, 41.25),  # Target based on typical pad performance
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_pad_deepseek_perf(
    step_name,
    warmup_iters,
    perf_target_us,
    galaxy_type,
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    subdir = "deepseek_pad_perf"
    command = f"pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_pad_deepseek.py::test_deepseek_v3_mla_pad_trace_mode[device_params0-100-10-{step_name}-32]"
    cols = ["DEVICE KERNEL"]
    op_name = "PadDeviceOperation"
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
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-min", measured_min)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-max", measured_max)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-avg", measured_avg)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-std", measured_std)
    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"tg_deepseek_pad",
        ml_model_name="deepseek-v3-tg",
    )

    threshold = max(THRESHOLD, perf_target_us * THRESHOLD_PERCENTAGE)

    assert (
        measured_avg_us < perf_target_us + threshold
    ), f"Performance is worse than target: {measured_avg_us} us > {perf_target_us} us, the threshold was {threshold} us"
    assert (
        measured_avg_us > perf_target_us - threshold
    ), f"Performance is more than {threshold} us better than target, update the target: {measured_avg_us} us < {perf_target_us} us"


@pytest.mark.parametrize(
    "step_name, warmup_iters, perf_target_us",
    [
        ("paged_update_cache_decode", 10, 9.78),  # Target based on existing test baseline
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_paged_update_cache_deepseek_perf(
    step_name,
    warmup_iters,
    perf_target_us,
    galaxy_type,
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()

    subdir = "deepseek_paged_update_cache_perf"
    # Run the existing comprehensive test in trace mode with program cache
    command = f"pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_paged_update_cache_deepseek.py"
    cols = ["DEVICE KERNEL"]
    op_name = "PagedUpdateCacheDeviceOperation"

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
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-min", measured_min)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-max", measured_max)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-avg", measured_avg)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-std", measured_std)
    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"tg_deepseek_paged_update_cache",
        ml_model_name="deepseek-v3-tg",
    )

    threshold = max(THRESHOLD, perf_target_us * THRESHOLD_PERCENTAGE)

    assert (
        measured_avg_us < perf_target_us + threshold
    ), f"Performance is worse than target: {measured_avg_us} us > {perf_target_us} us, the threshold was {threshold} us"
    assert (
        measured_avg_us > perf_target_us - threshold
    ), f"Performance is more than {threshold} us better than target, update the target: {measured_avg_us} us < {perf_target_us} us"


@pytest.mark.parametrize(
    "step_name, warmup_iters, perf_target_us",
    [
        ("q_reshape_decode", 10, 38.74),
        ("v_out_reshape_decode", 10, 174.55),
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_reshape_deepseek_perf(
    step_name,
    warmup_iters,
    perf_target_us,
    galaxy_type,
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()

    subdir = "deepseek_reshape_perf"
    command = f"pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_reshape_deepseek.py -k {step_name}"
    cols = ["DEVICE KERNEL"]
    op_name = "ReshapeViewDeviceOperation"

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
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-min", measured_min)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-max", measured_max)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-avg", measured_avg)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-std", measured_std)
    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"tg_deepseek_reshape",
        ml_model_name="deepseek-v3-tg",
    )

    threshold = max(THRESHOLD, perf_target_us * THRESHOLD_PERCENTAGE)

    assert (
        measured_avg_us < perf_target_us + threshold
    ), f"Performance is worse than target: {measured_avg_us} us > {perf_target_us} us, the threshold was {threshold} us"
    assert (
        measured_avg_us > perf_target_us - threshold
    ), f"Performance is more than {threshold} us better than target, update the target: {measured_avg_us} us < {perf_target_us} us"


@pytest.mark.parametrize(
    "step_name, warmup_iters, perf_target_us",
    [
        ("deepseek_mla_wq_a2a", 10, 52.77),  # Target based on typical all-to-all performance
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_alltoall_tg_deepseek_perf(
    step_name,
    warmup_iters,
    perf_target_us,
    galaxy_type,
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()

    subdir = "deepseek_ccl_perf"
    command = f"pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_alltoall_deepseek.py"
    cols = ["DEVICE KERNEL"]
    op_name = "AllToAllAsyncGenericDeviceOperation"  # CCL operation name for all-to-all
    warmup_iters = warmup_iters * 32  # Multiply by number of devices (32 for TG)

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
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-min", measured_min)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-max", measured_max)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-avg", measured_avg)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-std", measured_std)
    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"tg_deepseek_ccl_ops",
        ml_model_name="deepseek-v3-tg",
    )

    threshold = max(THRESHOLD, perf_target_us * THRESHOLD_PERCENTAGE)

    assert (
        measured_avg_us < perf_target_us + threshold
    ), f"Performance is worse than target: {measured_avg_us} us > {perf_target_us} us, the threshold was {threshold} us"
    assert (
        measured_avg_us > perf_target_us - threshold
    ), f"Performance is more than {threshold} us better than target, update the target: {measured_avg_us} us < {perf_target_us} us"


@pytest.mark.parametrize(
    "step_name, warmup_iters, perf_target_us",
    [
        ("kvpe_mesh_partition", 10, 2.39),
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_mesh_partition_deepseek_perf(
    step_name,
    warmup_iters,
    perf_target_us,
    galaxy_type,
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()

    subdir = "deepseek_ccl_perf"
    command = f"pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_mesh_partition.py -k {step_name}"
    cols = ["DEVICE KERNEL"]
    op_name = "MeshPartitionDeviceOperation"
    warmup_iters = warmup_iters * 32  # Multiply by number of devices (32 for TG)

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
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-min", measured_min)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-max", measured_max)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-avg", measured_avg)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-std", measured_std)
    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"tg_deepseek_ccl_ops",
        ml_model_name="deepseek-v3-tg",
    )

    threshold = max(THRESHOLD, perf_target_us * THRESHOLD_PERCENTAGE)

    assert (
        measured_avg_us < perf_target_us + threshold
    ), f"Performance is worse than target: {measured_avg_us} us > {perf_target_us} us, the threshold was {threshold} us"
    assert (
        measured_avg_us > perf_target_us - threshold
    ), f"Performance is more than {threshold} us better than target, update the target: {measured_avg_us} us < {perf_target_us} us"


@pytest.mark.parametrize(
    "step_name, warmup_iters, perf_target_us",
    [
        ("flash_mla_decode", 10, 110.05),  # Target based on typical flash attention performance
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_flash_mla_deepseek_perf(
    step_name,
    warmup_iters,
    perf_target_us,
    galaxy_type,
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()

    subdir = "deepseek_flash_mla_perf"
    command = f"pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_flash_mla_deepseek.py"
    cols = ["DEVICE KERNEL"]
    op_name = "SdpaDecodeDeviceOperation"

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
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-min", measured_min)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-max", measured_max)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-avg", measured_avg)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-std", measured_std)
    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"tg_deepseek_flash_mla",
        ml_model_name="deepseek-v3-tg",
    )

    threshold = max(THRESHOLD, perf_target_us * THRESHOLD_PERCENTAGE)

    assert (
        measured_avg_us < perf_target_us + threshold
    ), f"Performance is worse than target: {measured_avg_us} us > {perf_target_us} us, the threshold was {threshold} us"
    assert (
        measured_avg_us > perf_target_us - threshold
    ), f"Performance is more than {threshold} us better than target, update the target: {measured_avg_us} us < {perf_target_us} us"


@pytest.mark.parametrize(
    "step_name, warmup_iters, perf_target_us",
    [
        (
            "wq_kv_a_sequence",
            10,
            151.6,
        ),
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_wq_kv_a_sequence_deepseek_perf(
    step_name,
    warmup_iters,
    perf_target_us,
    galaxy_type,
):
    """
    Test performance of the complete wq_kv_a sequence from mla1d.py (lines 1092-1134).

    This test measures the end-to-end performance of the operation sequence:
    1. Linear projection (wq_kv_a): 28.76 µs
    2. All-gather: 106 µs
    3. Fast reduce: 7.73 µs
    4. Slice q: 1.47 µs
    5. Slice kv_nope: 1.26 µs
    6. Slice kv_rope: 0.95 µs

    Total expected: 146.17 µs
    """
    subdir = "fused_op_unit_tests/mla"
    command = f"pytest models/demos/deepseek_v3/tests/{subdir}/test_wq_kv_a_sequence_deepseek.py -k {step_name}"
    cols = ["DEVICE KERNEL"]

    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()

    op_name = ""

    profiler.start("run")
    profiler.start(step_name)
    results = run_device_perf_detailed(
        command, subdir, cols, op_name, has_signposts=True, warmup_iters=warmup_iters, per_op=True
    )
    profiler.end(step_name)
    profiler.end("run")
    # Get the measured performance
    measured_min = 0
    measured_max = 0
    measured_avg = 0
    measured_std = 0
    for op in results.keys():
        if op == "SliceDeviceOperation":
            measured_min += 3 * results[op][cols[0]]["MIN"]
            measured_max += 3 * results[op][cols[0]]["MAX"]
            measured_avg += 3 * results[op][cols[0]]["AVG"]
            measured_std += 3 * results[op][cols[0]]["STD"]
        else:
            measured_min += results[op][cols[0]]["MIN"]
            measured_max += results[op][cols[0]]["MAX"]
            measured_avg += results[op][cols[0]]["AVG"]
            measured_std += results[op][cols[0]]["STD"]
    measured_avg_us = measured_avg / 1000

    logger.info(f"Measured performance: {measured_avg_us:.3f} us vs. target: {perf_target_us} us")

    # Save the measurement
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-min", measured_min)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-max", measured_max)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-avg", measured_avg)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-std", measured_std)
    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"tg_deepseek_wq_kv_a_sequence",
        ml_model_name="deepseek-v3-tg",
    )

    threshold = max(THRESHOLD, perf_target_us * THRESHOLD_PERCENTAGE)

    assert (
        measured_avg_us < perf_target_us + threshold
    ), f"Performance is worse than target: {measured_avg_us} us > {perf_target_us} us, the threshold was {threshold} us"
    assert (
        measured_avg_us > perf_target_us - threshold
    ), f"Performance is more than {threshold} us better than target, update the target: {measured_avg_us} us < {perf_target_us} us"


# Todo: Figure out why this is slower than the sum of its parts by ~9us
@pytest.mark.parametrize(
    "step_name, warmup_iters, perf_target_us",
    [
        (
            "norm_and_rope_sequence",
            10,
            120.72,
        ),  # Sum: q_norm(7.33) + kv_norm(6.68) + to_mem(0.75) + permute(5.52) + reshard(1.04) + rope(4.18) + reshard(1.19) + permute(1.30) + concat(1.32) + pad(41.25) + permute(30.95) + reshard(2.83)
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_norm_and_rope_sequence_deepseek_perf(
    step_name,
    warmup_iters,
    perf_target_us,
    galaxy_type,
):
    subdir = "fused_op_unit_tests/mla"
    command = f"pytest models/demos/deepseek_v3/tests/{subdir}/test_norm_and_rope_sequence_deepseek.py -k {step_name}"
    cols = ["DEVICE KERNEL"]

    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()

    op_name = ""

    profiler.start("run")
    profiler.start(step_name)
    results = run_device_perf_detailed(
        command, subdir, cols, op_name, has_signposts=True, warmup_iters=warmup_iters, per_op=True
    )
    profiler.end(step_name)
    profiler.end("run")
    # Get the measured performance
    measured_min = 0
    measured_max = 0
    measured_avg = 0
    measured_std = 0
    for op in results.keys():
        if op == "TransposeDeviceOperation":
            measured_min += 3 * results[op][cols[0]]["MIN"]
            measured_max += 3 * results[op][cols[0]]["MAX"]
            measured_avg += 3 * results[op][cols[0]]["AVG"]
            measured_std += 3 * results[op][cols[0]]["STD"]
        elif op == "ShardedToInterleavedDeviceOperation":
            measured_min += 2 * results[op][cols[0]]["MIN"]
            measured_max += 2 * results[op][cols[0]]["MAX"]
            measured_avg += 2 * results[op][cols[0]]["AVG"]
            measured_std += 2 * results[op][cols[0]]["STD"]
        else:
            measured_min += results[op][cols[0]]["MIN"]
            measured_max += results[op][cols[0]]["MAX"]
            measured_avg += results[op][cols[0]]["AVG"]
            measured_std += results[op][cols[0]]["STD"]
    measured_avg_us = measured_avg / 1000

    logger.info(f"Measured performance: {measured_avg_us:.3f} us vs. target: {perf_target_us} us")

    # Save the measurement
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-min", measured_min)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-max", measured_max)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-avg", measured_avg)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-std", measured_std)
    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"tg_deepseek_wq_kv_a_sequence",
        ml_model_name="deepseek-v3-tg",
    )

    threshold = max(THRESHOLD, perf_target_us * THRESHOLD_PERCENTAGE)

    assert (
        measured_avg_us < perf_target_us + threshold
    ), f"Performance is worse than target: {measured_avg_us} us > {perf_target_us} us, the threshold was {threshold} us"
    assert (
        measured_avg_us > perf_target_us - threshold
    ), f"Performance is more than {threshold} us better than target, update the target: {measured_avg_us} us < {perf_target_us} us"


@pytest.mark.parametrize(
    "step_name, warmup_iters, perf_target_us",
    [
        (
            "wkv_b2_sequence",
            10,
            185.33,
        ),  # Sum: reshard(3.89) + permute(70.29) + linear(141.68) + permute(7.61)
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_wkv_b2_sequence_deepseek_perf(
    step_name,
    warmup_iters,
    perf_target_us,
    galaxy_type,
):
    """
    Test performance of the complete wkv_b2 sequence from mla1d.py (lines 1331-1348).

    This test measures the end-to-end performance of the operation sequence:
    1. Reshard (flash_mla_out_reshard): 3.89 µs
    2. Permute (attn_out_permute_pre_linear): 31.94 µs
    3. Linear (wkv_b2): 141.68 µs
    4. Permute (v_out_permute): 7.61 µs

    Total expected: 185.12 µs
    """
    subdir = "fused_op_unit_tests/mla"
    command = f"pytest models/demos/deepseek_v3/tests/{subdir}/test_wkv_b2_sequence_deepseek.py -k {step_name}"
    cols = ["DEVICE KERNEL"]

    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()

    op_name = ""

    profiler.start("run")
    profiler.start(step_name)
    results = run_device_perf_detailed(
        command, subdir, cols, op_name, has_signposts=True, warmup_iters=warmup_iters, per_op=True
    )
    profiler.end(step_name)
    profiler.end("run")
    # Get the measured performance
    measured_min = 0
    measured_max = 0
    measured_avg = 0
    measured_std = 0
    for op in results.keys():
        if op == "TransposeDeviceOperation":
            # There are 2 transpose operations (permutes)
            measured_min += 2 * results[op][cols[0]]["MIN"]
            measured_max += 2 * results[op][cols[0]]["MAX"]
            measured_avg += 2 * results[op][cols[0]]["AVG"]
            measured_std += 2 * results[op][cols[0]]["STD"]
        else:
            measured_min += results[op][cols[0]]["MIN"]
            measured_max += results[op][cols[0]]["MAX"]
            measured_avg += results[op][cols[0]]["AVG"]
            measured_std += results[op][cols[0]]["STD"]
    measured_avg_us = measured_avg / 1000

    logger.info(f"Measured performance: {measured_avg_us:.3f} us vs. target: {perf_target_us} us")

    # Save the measurement
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-min", measured_min)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-max", measured_max)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-avg", measured_avg)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-std", measured_std)
    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"tg_deepseek_wkv_b2_sequence",
        ml_model_name="deepseek-v3-tg",
    )

    threshold = max(THRESHOLD, perf_target_us * THRESHOLD_PERCENTAGE)

    assert (
        measured_avg_us < perf_target_us + threshold
    ), f"Performance is worse than target: {measured_avg_us} us > {perf_target_us} us, the threshold was {threshold} us"
    assert (
        measured_avg_us > perf_target_us - threshold
    ), f"Performance is more than {threshold} us better than target, update the target: {measured_avg_us} us < {perf_target_us} us"


@pytest.mark.parametrize(
    "step_name, warmup_iters, perf_target_us",
    [
        (
            "fwd_decode_q_rope_nope",
            10,
            176.96,
        ),  # Sum: linear(57.11) + reshape(38.74) + slice(2.27+1.84) + permute(6.23) + linear(35.62) + permute(16.47) + permute(1.62) + rotary(4.16) + to_mem(1.19) + concat(8.01)
    ],
)
# 33uS slower than sum of unit tests
@pytest.mark.models_device_performance_bare_metal
def test_fwd_decode_q_rope_nope_deepseek_perf(
    step_name,
    warmup_iters,
    perf_target_us,
    galaxy_type,
):
    """
    Test performance of the fwd_decode_q_rope_nope sequence from mla1d.py (lines 1234-1295).

    This test measures the end-to-end performance of the operation sequence:
    1. Linear projection (wq_b): 57.11 µs
    2. Reshape: 38.74 µs
    3. Slice tt_q_nope: 2.27 µs
    4. Slice tt_q_rope: 1.84 µs
    5. Permute tt_q_nope: 6.23 µs
    6. Linear projection (wkv_b1): 35.62 µs
    7. Permute tt_q_nope: 16.47 µs
    8. Permute tt_q_rope: 1.62 µs
    9. Rotary embedding tt_q_rope: 4.16 µs
    10. To memory config: 1.19 µs
    11. Concat tt_q_nope and tt_q_rope: 8.01 µs

    Total expected: 173.26 µs
    """
    subdir = "fused_op_unit_tests/mla"
    command = f"pytest models/demos/deepseek_v3/tests/{subdir}/test_fwd_decode_q_rope_nope_deepseek.py -k {step_name}"
    cols = ["DEVICE KERNEL"]

    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()

    op_name = ""

    profiler.start("run")
    profiler.start(step_name)
    results = run_device_perf_detailed(
        command, subdir, cols, op_name, has_signposts=True, warmup_iters=warmup_iters, per_op=True
    )
    profiler.end(step_name)
    profiler.end("run")
    # Get the measured performance
    measured_min = 0
    measured_max = 0
    measured_avg = 0
    measured_std = 0
    for op in results.keys():
        if op in [
            "SliceDeviceOperation",
            "MatmulDeviceOperation",
            "PermuteDeviceOperation",
        ]:
            measured_min += 2 * results[op][cols[0]]["MIN"]
            measured_max += 2 * results[op][cols[0]]["MAX"]
            measured_avg += 2 * results[op][cols[0]]["AVG"]
            measured_std += 2 * results[op][cols[0]]["STD"]
        else:
            measured_min += 1 * results[op][cols[0]]["MIN"]
            measured_max += 1 * results[op][cols[0]]["MAX"]
            measured_avg += 1 * results[op][cols[0]]["AVG"]
            measured_std += 1 * results[op][cols[0]]["STD"]
    measured_avg_us = measured_avg / 1000

    logger.info(f"Measured performance: {measured_avg_us:.3f} us vs. target: {perf_target_us} us")

    # Save the measurement
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-min", measured_min)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-max", measured_max)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-avg", measured_avg)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-std", measured_std)
    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"tg_deepseek_fwd_decode_q_rope_nope",
        ml_model_name="deepseek-v3-tg",
    )

    threshold = max(THRESHOLD, perf_target_us * THRESHOLD_PERCENTAGE)

    assert (
        measured_avg_us < perf_target_us + threshold
    ), f"Performance is worse than target: {measured_avg_us} us > {perf_target_us} us, the threshold was {threshold} us"
    assert (
        measured_avg_us > perf_target_us - threshold
    ), f"Performance is more than {threshold} us better than target, update the target: {measured_avg_us} us < {perf_target_us} us"
