# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger

from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from models.perf.device_perf_utils import run_device_perf_detailed

THRESHOLD = 0.4


@pytest.fixture(autouse=True)
def ensure_devices():
    """Skip device initialization since this test spawns child pytest processes."""


# Override galaxy_type fixture to avoid calling ttnn.cluster.get_cluster_type()
# which might acquire device locks
@pytest.fixture(scope="function")
def galaxy_type():
    """Return galaxy type without initializing devices."""
    import os

    # Use environment variable if set, otherwise default to "TG"
    return os.environ.get("GALAXY_TYPE", "TG")


@pytest.mark.parametrize(
    "step_name, warmup_iters, perf_target_us",
    [
        ("mla_linear_wq_kv_a", 10, 28.81),
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_linear_tg_deepseek_perf(
    step_name,
    warmup_iters,
    perf_target_us,
    galaxy_type,
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()

    subdir = "deepseek_linear_perf"
    command = f"pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_linear_deepseek.py"
    cols = ["DEVICE KERNEL"]
    op_name = "MeshDeviceOperationAdapter<ttnn::prim::MatmulDeviceOperation>"

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

    assert (
        measured_avg_us < perf_target_us + THRESHOLD
    ), f"Performance target not met: {measured_avg_us} us > {perf_target_us} us"


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

    assert (
        measured_avg_us < perf_target_us + THRESHOLD
    ), f"Performance target not met: {measured_avg_us} us > {perf_target_us} us"


@pytest.mark.parametrize(
    "step_name, warmup_iters, perf_target_us",
    [
        ("mla_allgather_wq_kv_a", 10, 106),  # Target based on typical all-gather performance
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
    command = f"pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_allgather_deepseek.py"
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

    assert (
        measured_avg_us < perf_target_us + THRESHOLD
    ), f"Performance target not met: {measured_avg_us} us > {perf_target_us} us"


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

    assert (
        measured_avg_us < perf_target_us + THRESHOLD
    ), f"Performance target not met: {measured_avg_us} us > {perf_target_us} us"


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

    assert (
        measured_avg_us < perf_target_us + THRESHOLD
    ), f"Performance target not met: {measured_avg_us} us > {perf_target_us} us"


@pytest.mark.parametrize(
    "step_name, warmup_iters, perf_target_us, op_name",
    [
        (
            "kv_nope_to_interleaved",
            10,
            0.75,
            "MeshDeviceOperationAdapter<ttnn::prim::ShardedToInterleavedDeviceOperation>",
        ),
        ("kv_rope_reshard", 10, 1.04, "MeshDeviceOperationAdapter<ttnn::prim::InterleavedToShardedDeviceOperation>"),
        (
            "kv_rope_out_reshard",
            10,
            1.19,
            "MeshDeviceOperationAdapter<ttnn::prim::ShardedToInterleavedDeviceOperation>",
        ),
        ("kvpe_reshard", 10, 2.8, "MeshDeviceOperationAdapter<ttnn::prim::InterleavedToShardedDeviceOperation>"),
        ("q_rope_out_reshard", 10, 1.18, "MeshDeviceOperationAdapter<ttnn::prim::ShardedToInterleavedDeviceOperation>"),
        ("flash_mla_reshard", 10, 9.34, "MeshDeviceOperationAdapter<ttnn::prim::InterleavedToShardedDeviceOperation>"),
        (
            "flash_mla_out_reshard",
            10,
            3.86,
            "MeshDeviceOperationAdapter<ttnn::prim::ShardedToInterleavedDeviceOperation>",
        ),
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

    assert (
        measured_avg_us < perf_target_us + THRESHOLD
    ), f"Performance target not met: {measured_avg_us} us > {perf_target_us} us"


@pytest.mark.parametrize(
    "step_name, warmup_iters, perf_target_us",
    [
        ("kv_rope_permute_pre", 10, 7.6),
        ("kv_rope_permute_post", 10, 1.5),
        ("kvpe_permute", 10, 25),
        ("q_nope_permute_pre_linear", 10, 106),
        ("q_nope_permute_post_linear", 10, 23),
        ("q_rope_permute", 10, 1.7),
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

    assert (
        measured_avg_us < perf_target_us + THRESHOLD
    ), f"Performance target not met: {measured_avg_us} us > {perf_target_us} us"


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

    assert (
        measured_avg_us < perf_target_us + THRESHOLD
    ), f"Performance target not met: {measured_avg_us} us > {perf_target_us} us"


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

    assert (
        measured_avg_us < perf_target_us + THRESHOLD
    ), f"Performance target not met: {measured_avg_us} us > {perf_target_us} us"


@pytest.mark.parametrize(
    "step_name, warmup_iters, perf_target_us",
    [
        ("kvpe_pad", 10, 45),  # Target based on typical pad performance
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
    cols = ["DEVICE FW"]
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

    assert (
        measured_avg_us < perf_target_us + THRESHOLD
    ), f"Performance target not met: {measured_avg_us} us > {perf_target_us} us"


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

    assert (
        measured_avg_us < perf_target_us + THRESHOLD
    ), f"Performance target not met: {measured_avg_us} us > {perf_target_us} us"


@pytest.mark.parametrize(
    "step_name, warmup_iters, perf_target_us",
    [
        ("q_reshape_decode", 10, 39),
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

    assert (
        measured_avg_us < perf_target_us + THRESHOLD
    ), f"Performance target not met: {measured_avg_us} us > {perf_target_us} us"
