# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

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
        ("wq_kv_a", 10, 28.81),
        ("wq_b", 10, 57.5),
        ("wkv_b1", 10, 35.63),
        ("wkv_b2", 10, 142),
        ("wo", 10, 259.59),
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
        ("mla_reduce_wq_kv_a", 10, 7.75),  # Target based on typical reduce performance
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
        ("wq_kv_a_ag_decode", 10, 106),  # Target based on typical all-gather performance
        ("wo_ag_decode", 10, 118),  # Target based on typical all-gather performance
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
    op_name = "AllGatherAsync"  # CCL operation name for all-gather
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
        ("q_slice", 10, 1.5),  # Target based on typical reduce performance
        ("kv_nope_slice", 10, 1.3),
        ("kv_rope_slice", 10, 1),
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
        ("q_nope_slice", 10, 2.39),  # 2.43 if run on mesh device
        ("q_rope_slice", 10, 1.88),  # 1.89 if run on mesh device
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
        ("q_norm", 10, 7.34),  # Target based on typical reduce performance
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
        ("kvpe_reshard", 10, 2.8, "InterleavedToShardedDeviceOperation"),
        ("q_rope_out_reshard", 10, 1.18, "ShardedToInterleavedDeviceOperation"),
        ("flash_mla_reshard", 10, 9.34, "InterleavedToShardedDeviceOperation"),
        ("flash_mla_out_reshard", 10, 3.86, "ShardedToInterleavedDeviceOperation"),
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
        ("kv_rope_permute_pre", 10, 7.6),
        ("kv_rope_permute_post", 10, 1.5),
        ("kvpe_permute", 10, 31),
        ("q_nope_permute_pre_linear", 10, 6.20),
        ("q_nope_permute_post_linear", 10, 22.95),
        ("q_rope_permute", 10, 1.65),
        ("attn_out_permute_pre_linear", 10, 71),
        ("v_out_permute", 10, 8),
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
    warmup_iters = warmup_iters  # Multiply by number of devices (32 for TG)

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
        ("kv_rope_decode", 10, 4.5),
        ("q_rope_decode", 10, 4.5),
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
    warmup_iters = warmup_iters  # Multiply by number of devices (32 for TG)

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
        ("kvpe_concat", 10, 1.36),
        ("q_concat", 10, 7.90),
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
    warmup_iters = warmup_iters  # Multiply by number of devices (32 for TG)

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
        ("kvpe_pad", 10, 41.2),  # Target based on typical pad performance
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
        ("paged_update_cache_decode", 10, 10),  # Target based on existing test baseline
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
        ("q_reshape_decode", 10, 38.71),
        ("v_out_reshape_decode", 10, 175),
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
    op_name = "ReshapeDeviceOperation"

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
        ("deepseek_mla_wq_a2a", 10, 383),  # Target based on typical all-to-all performance
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
    op_name = "AllToAllAsync"  # CCL operation name for all-to-all
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
        ("kvpe_mesh_partition", 10, 2.54),
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
        ("flash_mla_decode", 10, 193.4),  # Target based on typical flash attention performance
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
            146.7,
        ),  # Sum: linear(28.81) + all-gather(106) + fast_reduce(7.75) + 3*slices(1.5+1.3+1)
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
    1. Linear projection (wq_kv_a): 28.81 µs
    2. All-gather: 106 µs
    3. Fast reduce: 7.75 µs
    4. Slice q: 1.5 µs
    5. Slice kv_nope: 1.3 µs
    6. Slice kv_rope: 1 µs

    Total expected: 146.36 µs
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
    print(results)
    measured_min = 0
    measured_max = 0
    measured_avg = 0
    measured_std = 0
    for op in results.keys():
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
            120,
        ),  # Sum: q_norm(7.34) + kv_norm(6.68) + to_mem(0.75) + permute(7.6) + reshard(1.04) + rope(4.5) + reshard(1.19) + permute(1.5) + concat(1.36) + pad(45) + permute(31) + reshard(2.8)
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
    print(results)
    measured_min = 0
    measured_max = 0
    measured_avg = 0
    measured_std = 0
    for op in results.keys():
        if op == "MeshDeviceOperationAdapter<ttnn::operations::data_movement::transpose::TransposeDeviceOperation>":
            measured_min += 3 * results[op][cols[0]]["MIN"]
            measured_max += 3 * results[op][cols[0]]["MAX"]
            measured_avg += 3 * results[op][cols[0]]["AVG"]
            measured_std += 3 * results[op][cols[0]]["STD"]
        elif op == "MeshDeviceOperationAdapter<ttnn::operations::data_movement::ShardedToInterleavedDeviceOperation>":
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
            225,
        ),  # Sum: reshard(3.86) + permute(71) + linear(142) + permute(8)
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
    1. Reshard (flash_mla_out_reshard): 3.86 µs
    2. Permute (attn_out_permute_pre_linear): 71 µs
    3. Linear (wkv_b2): 142 µs
    4. Permute (v_out_permute): 8 µs

    Total expected: 224.86 µs
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
    print(results)
    measured_min = 0
    measured_max = 0
    measured_avg = 0
    measured_std = 0
    for op in results.keys():
        if op == "MeshDeviceOperationAdapter<ttnn::operations::data_movement::transpose::TransposeDeviceOperation>":
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
            184.9,
        ),  # Sum: linear(57.5) + reshape(39) + slice(2.43+1.89) + permute(6.20) + linear(35.63) + permute(22.95) + permute(1.65) + rotary(4.56) + to_mem(1.18) + concat(7.90)
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
    1. Linear projection (tt_q): 57.5 µs(measured single) ? 46.72 (measured here)
    2. Reshape: 38.71 µs ? 38.57
    3. Slice tt_q_nope: 2.40 µs ? 2.82
    4. Slice tt_q_rope: 1.87 µs ? 2.82
    5. Permute tt_q_nope: 6.20 µs ? 4.75
    6. Linear projection (tt_q_nope): 35.63 µs ? 46.72
    7. Permute tt_q_nope: 22.95 µs ? 23.56 (transpose)
    8. Permute tt_q_rope: 1.65 µs ? 4.75
    9. Rotary embedding tt_q_rope: 4.56 µs ? 4.56
    10. To memory config: 1.18 µs ? 2.34
    11. Concat tt_q_nope and tt_q_rope: 7.90 µs ? 7.26

    Total expected: 184.9 µs
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
    print(results)
    measured_min = 0
    measured_max = 0
    measured_avg = 0
    measured_std = 0
    for op in results.keys():
        if op in [
            "MeshDeviceOperationAdapter<ttnn::operations::data_movement::slice::SliceDeviceOperation>",
            "MeshDeviceOperationAdapter<ttnn::operations::matmul::MatmulDeviceOperation>",
            "MeshDeviceOperationAdapter<ttnn::operations::data_movement::PermuteDeviceOperation>",
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
