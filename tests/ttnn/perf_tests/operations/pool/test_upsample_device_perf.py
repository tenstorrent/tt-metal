# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger
from models.perf.benchmarking_utils import BenchmarkProfiler
from tests.ttnn.perf_tests.utils import extract_ops_between_signposts
from tests.ttnn.ttnn_utility_fuction import get_shard_grid_from_num_cores
from tracy import signpost
from tracy.common import clear_profiler_runtime_artifacts
from tracy.process_model_log import get_latest_ops_log_filename, run_device_profiler
from ttnn.device import is_blackhole, is_wormhole_b0


# fmt: off
UPSAMPLE_PERF_CONFIGS = [
    # BLOCK SHARDED - SDXL U-Net decoder upsample (32x32 -> 64x64)
    {
        "test_name": "sdxl_upsample_nearest_bs",
        "batch_size": 2,
        "input_channels": 640,
        "input_height": 32,
        "input_width": 32,
        "scale_factor": (2, 2),
        "mode": "nearest",
        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        "ncores": (8, 8),
        "shard_shape": (256, 80),
        "perf_targets": {"wh": 13.1, "bh_p150": 13.1},
    },
    # HEIGHT SHARDED - Stable Diffusion bilinear upsample (8x8 -> 16x16)
    {
        "test_name": "sd_upsample_bilinear_hs",
        "batch_size": 2,
        "input_channels": 1280,
        "input_height": 8,
        "input_width": 8,
        "scale_factor": (2, 2),
        "mode": "bilinear",
        "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        "ncores": 64,
        "shard_shape": (2, 1280),
        "perf_targets": {"wh": 13.7, "bh_p150": 13.7},
    },
]
# fmt: on

THRESHOLD_PERCENT = 0.05

PERF_TARGETS_WH = {config["test_name"]: config["perf_targets"]["wh"] for config in UPSAMPLE_PERF_CONFIGS}
PERF_TARGETS_BH_P150 = {config["test_name"]: config["perf_targets"]["bh_p150"] for config in UPSAMPLE_PERF_CONFIGS}


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_run_upsample_ops(device):
    """
    Consolidated performance test that runs all upsample configurations in a single process.
    Uses Tracy signposts to mark each configuration's measurement region.
    """
    torch.manual_seed(0)

    if not (is_blackhole() or is_wormhole_b0()):
        pytest.skip("Test is only for Blackhole and Wormhole B0 architectures")

    if device.core_grid.y != 8 and is_wormhole_b0():
        pytest.skip("Needs 8x8 grid for wormhole_b0")

    warmup_iterations = 2

    for config in UPSAMPLE_PERF_CONFIGS:
        test_name = config["test_name"]
        batch_size = config["batch_size"]
        num_channels = config["input_channels"]
        height = config["input_height"]
        width = config["input_width"]

        # Create input tensor in NHWC format
        input_nhwc = torch.randn(batch_size, height, width, num_channels, dtype=torch.bfloat16)

        # Build sharded memory config from pre-computed values
        shard_grid = get_shard_grid_from_num_cores(config["ncores"], device)
        shard_spec = ttnn.ShardSpec(shard_grid, config["shard_shape"], ttnn.ShardOrientation.ROW_MAJOR)
        sharded_mem_config = ttnn.MemoryConfig(config["shard_layout"], ttnn.BufferType.L1, shard_spec)

        # Move to device and apply sharding
        tt_input = ttnn.from_torch(input_nhwc, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
        tt_input = ttnn.to_memory_config(tt_input, memory_config=sharded_mem_config)

        # Setup compute kernel config for bilinear mode
        compute_kernel_config = None
        if config["mode"] == "bilinear":
            compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
            )

        # Warmup iterations
        for _ in range(warmup_iterations):
            tt_output = ttnn.upsample(
                tt_input, config["scale_factor"], mode=config["mode"], compute_kernel_config=compute_kernel_config
            )
            ttnn.deallocate(tt_output)

        # Measured iteration with signposts
        signpost(f"{test_name}-start")
        tt_output = ttnn.upsample(
            tt_input, config["scale_factor"], mode=config["mode"], compute_kernel_config=compute_kernel_config
        )
        signpost(f"{test_name}-end")


@pytest.mark.models_device_performance_bare_metal
def test_upsample_device_perf():
    if is_blackhole():
        perf_targets = PERF_TARGETS_BH_P150
    elif is_wormhole_b0():
        perf_targets = PERF_TARGETS_WH
    else:
        pytest.skip("Test is only for Blackhole and Wormhole B0 architectures")

    profiler = BenchmarkProfiler()
    step_name = "upsample_all_perf"

    subdir = "upsample_perf"
    command = (
        "pytest tests/ttnn/perf_tests/operations/pool/test_upsample_device_perf.py::test_run_upsample_ops --timeout=60"
    )
    op_name = "UpsampleOperation"

    logger.info(f"Command: {command}")

    clear_profiler_runtime_artifacts()

    profiler.start(step_name)
    run_device_profiler(command, subdir, device_analysis_types=["device_kernel_duration"])
    profiler.end(step_name)

    csv_path = get_latest_ops_log_filename(subdir)
    logger.info(f"Parsing results from: {csv_path}")

    results = extract_ops_between_signposts(csv_path, op_name=op_name)

    failures = []
    for test_name, perf_target_us in perf_targets.items():
        if test_name not in results:
            failures.append(f"[{test_name}] No results found in profiling data")
            continue

        durations_ns = results[test_name]
        if len(durations_ns) == 0:
            failures.append(f"[{test_name}] No operations captured between signposts")
            continue

        measured_avg_ns = sum(durations_ns) / len(durations_ns)
        measured_avg_us = measured_avg_ns / 1000
        measured_min_us = min(durations_ns) / 1000
        measured_max_us = max(durations_ns) / 1000

        logger.info(
            f"[{test_name}] Measured: {measured_avg_us:.3f} us (range: {measured_min_us:.3f}-{measured_max_us:.3f}) vs target: {perf_target_us} us"
        )

        threshold_limit = perf_target_us * (1 + THRESHOLD_PERCENT)
        if measured_avg_us >= threshold_limit:
            failures.append(f"[{test_name}] Too slow: {measured_avg_us:.3f} us >= {threshold_limit:.3f} us")

        upper_bound_limit = perf_target_us * (1 - THRESHOLD_PERCENT)
        if measured_avg_us < upper_bound_limit:
            failures.append(f"[{test_name}] Suspiciously fast: {measured_avg_us:.3f} us < {upper_bound_limit:.3f} us")

    if failures:
        pytest.fail(f"Performance targets not met:\n" + "\n".join(failures))
