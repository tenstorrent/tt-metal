# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger
from models.perf.benchmarking_utils import BenchmarkProfiler
from tests.ttnn.perf_tests.utils import extract_ops_between_signposts
from tracy import signpost
from tracy.common import clear_profiler_runtime_artifacts
from tracy.process_model_log import get_latest_ops_log_filename, run_device_profiler
from ttnn.device import is_blackhole, is_wormhole_b0

# OFT model configuration (from models/experimental/oft/tt/tt_oft.py)
# - Input: 159x159 integral image with 64 channels
# - Grid: Split into 18 slices, each slice (1, 1408, 78, 2) -> reshaped to (1, 1, 1408, 468) with K=78
# - Uses HEIGHT_SHARDED output with batch_output_channels=True
# - Output width = channels * K = 64 * 78 = 4992
# fmt: off
GRID_SAMPLE_PERF_CONFIGS = [
    # OFT Integral Image Sampling - with precomputed grid (host preparation)
    # Measured on WH: 1117.7us
    {
        "test_name": "oft_integral_precomputed",
        "input_height": 159,
        "input_width": 159,
        "input_channels": 64,
        "grid_height": 1408,       # One slice of OFT grid (25344 / 18 slices)
        "grid_width": 78,          # Number of grid points per row (K = 78)
        "mode": "bilinear",
        "align_corners": False,
        "use_precomputed_grid": True,
        "batch_output_channels": True,
        "in_dtype": ttnn.bfloat16,
        "grid_dtype": ttnn.bfloat16,
        "shard_height": 32,
        "shard_width": 4992,       # channels * K = 64 * 78
        "core_grid": (8, 8),
        "perf_targets": {"wh": 1174, "bh_p150": 845},
    },
    # OFT Integral Image Sampling - without precomputed grid (no host preparation)
    # Measured on WH: 5307.9us
    {
        "test_name": "oft_integral_standard",
        "input_height": 159,
        "input_width": 159,
        "input_channels": 64,
        "grid_height": 1408,       # One slice of OFT grid
        "grid_width": 78,          # Number of grid points per row (K = 78)
        "mode": "bilinear",
        "align_corners": False,
        "use_precomputed_grid": False,
        "batch_output_channels": True,
        "in_dtype": ttnn.bfloat16,
        "grid_dtype": ttnn.bfloat16,
        "shard_height": 32,
        "shard_width": 4992,       # channels * K = 64 * 78
        "core_grid": (8, 8),
        "perf_targets": {"wh": 5573, "bh_p150": 1605},
    },
]
# fmt: on

# Applied to both upper and lower bounds
THRESHOLD_PERCENT = 0.05

# Build performance targets per architecture (in us) from global config
PERF_TARGETS_WH = {config["test_name"]: config["perf_targets"]["wh"] for config in GRID_SAMPLE_PERF_CONFIGS}
PERF_TARGETS_BH_P150 = {config["test_name"]: config["perf_targets"]["bh_p150"] for config in GRID_SAMPLE_PERF_CONFIGS}


def test_run_grid_sample_ops(device):
    """
    Consolidated performance test that runs all grid_sample configurations in a single process.
    Uses Tracy signposts to mark each configuration's measurement region.
    """
    torch.manual_seed(0)

    if not (is_blackhole() or is_wormhole_b0()):
        pytest.skip("Test is only for Blackhole and Wormhole B0 architectures")

    if device.core_grid.y != 8 and is_wormhole_b0():
        pytest.skip("Needs 8x8 grid for wormhole_b0")

    warmup_iterations = 2

    for config in GRID_SAMPLE_PERF_CONFIGS:
        test_name = config["test_name"]

        # Create input tensor (NHWC format for grid_sample)
        input_shape = (1, config["input_height"], config["input_width"], config["input_channels"])
        torch_input = torch.randn(input_shape, dtype=torch.bfloat16)

        # Create grid tensor with OFT-style reshaping for batch_output_channels
        # Grid starts as (1, grid_height, grid_width, 2), then reshaped to pack K coordinate sets
        grid_shape = (1, config["grid_height"], config["grid_width"], 2)
        torch_grid = torch.rand(grid_shape, dtype=torch.float32) * 2.0 - 1.0

        if config["use_precomputed_grid"]:
            # Prepare the grid on host: (1, H, W, 2) -> (1, H, W, 6)
            ttnn_grid_host = ttnn.from_torch(torch_grid, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32)
            input_shape_nhwc = [1, config["input_height"], config["input_width"], config["input_channels"]]
            prepared_grid = ttnn.prepare_grid_sample_grid(
                ttnn_grid_host,
                input_shape_nhwc,
                padding_mode="zeros",
                align_corners=config["align_corners"],
                output_dtype=config["grid_dtype"],
            )
            # Reshape to pack grid_width coordinate sets into last dimension: (1, H, W, 6) -> (1, 1, H, W*6)
            # This gives K = grid_width (78 in OFT)
            prepared_grid = ttnn.reshape(prepared_grid, [1, 1, config["grid_height"], config["grid_width"] * 6])
            tt_grid = ttnn.to_device(prepared_grid, device)
        else:
            # Standard grid: reshape to pack coordinate sets: (1, H, W, 2) -> (1, 1, H, W*2)
            torch_grid_reshaped = torch_grid.reshape(1, 1, config["grid_height"], config["grid_width"] * 2)
            tt_grid = ttnn.from_torch(
                torch_grid_reshaped.to(torch.bfloat16)
                if config["grid_dtype"] == ttnn.bfloat16
                else torch_grid_reshaped,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                dtype=config["grid_dtype"],
            )

        # Create input tensor on device
        tt_input = ttnn.from_torch(torch_input, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

        # Create HEIGHT_SHARDED memory config (matching OFT model)
        core_grid = ttnn.CoreGrid(x=config["core_grid"][0], y=config["core_grid"][1])
        memory_config = ttnn.create_sharded_memory_config(
            (config["shard_height"], config["shard_width"]),
            core_grid,
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        # Warmup iterations (without signposts)
        for _ in range(warmup_iterations):
            tt_output = ttnn.grid_sample(
                tt_input,
                tt_grid,
                mode=config["mode"],
                align_corners=config["align_corners"],
                use_precomputed_grid=config["use_precomputed_grid"],
                batch_output_channels=config["batch_output_channels"],
                memory_config=memory_config,
            )
            ttnn.deallocate(tt_output)

        # Measured iteration with signposts
        signpost(f"{test_name}-start")
        tt_output_tensor_on_device = ttnn.grid_sample(
            tt_input,
            tt_grid,
            mode=config["mode"],
            align_corners=config["align_corners"],
            use_precomputed_grid=config["use_precomputed_grid"],
            batch_output_channels=config["batch_output_channels"],
            memory_config=memory_config,
        )
        signpost(f"{test_name}-end")

        # Deallocate tensors
        ttnn.deallocate(tt_output_tensor_on_device)
        ttnn.deallocate(tt_input)
        ttnn.deallocate(tt_grid)


@pytest.mark.models_device_performance_bare_metal
def test_grid_sample_device_perf():
    if is_blackhole():
        perf_targets = PERF_TARGETS_BH_P150
    elif is_wormhole_b0():
        perf_targets = PERF_TARGETS_WH
    else:
        pytest.skip("Test is only for Blackhole and Wormhole B0 architectures")

    profiler = BenchmarkProfiler()
    step_name = "grid_sample_all_perf"

    subdir = "grid_sample_perf"
    command = "pytest tests/ttnn/perf_tests/operations/pool/test_grid_sample_device_perf.py::test_run_grid_sample_ops --timeout=60"
    # Grid sample operations are named GridSampleOperation in the profiler output
    op_name = "GridSampleOperation"

    logger.info(f"Command: {command}")

    clear_profiler_runtime_artifacts()

    # Run the consolidated test with profiling
    profiler.start(step_name)
    run_device_profiler(command, subdir, device_analysis_types=["device_kernel_duration"])
    profiler.end(step_name)

    csv_path = get_latest_ops_log_filename(subdir)
    logger.info(f"Parsing results from: {csv_path}")
    results = extract_ops_between_signposts(csv_path, op_name=op_name)

    # Validate results for each test
    failures = []
    for test_name, perf_target_us in perf_targets.items():
        if test_name not in results:
            failures.append(f"[{test_name}] No results found in profiling data")
            continue

        durations_ns = results[test_name]
        if len(durations_ns) == 0:
            failures.append(f"[{test_name}] No operations captured between signposts")
            continue

        # Calculate statistics
        measured_avg_ns = sum(durations_ns) / len(durations_ns)
        measured_min_ns = min(durations_ns)
        measured_max_ns = max(durations_ns)

        measured_avg_us = measured_avg_ns / 1000
        measured_min_us = measured_min_ns / 1000
        measured_max_us = measured_max_ns / 1000

        logger.info(f"[{test_name}] Measured performance: {measured_avg_us:.3f} us vs. target: {perf_target_us} us")
        logger.info(f"[{test_name}] Performance range: {measured_min_us:.3f} - {measured_max_us:.3f} us")
        logger.info(f"[{test_name}] Number of samples: {len(durations_ns)}")

        # Check against target with percentage threshold (lower bound - too slow)
        threshold_limit = perf_target_us * (1 + THRESHOLD_PERCENT)
        if measured_avg_us >= threshold_limit:
            failures.append(
                f"[{test_name}] Performance target not met: {measured_avg_us:.3f} us >= {threshold_limit:.3f} us (target {perf_target_us} us + {THRESHOLD_PERCENT*100}%)"
            )

        # Check upper bound - too fast (potential measurement issue)
        upper_bound_limit = perf_target_us * (1 - THRESHOLD_PERCENT)
        if measured_avg_us < upper_bound_limit:
            failures.append(
                f"[{test_name}] Performance suspiciously fast: {measured_avg_us:.3f} us < {upper_bound_limit:.3f} us (target {perf_target_us} us - {THRESHOLD_PERCENT*100}%)"
            )

    # Assert all tests passed
    if failures:
        failure_msg = "\n".join(failures)
        pytest.fail(f"Performance targets not met:\n{failure_msg}")
