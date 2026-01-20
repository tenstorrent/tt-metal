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

# Global pool configuration array with performance targets (in us)
# Measured on WH: resnet50_maxpool_hs=124.7us (batch=16), vgg16_maxpool_hs=31.4us,
#                 gemma3_avgpool_bs=17.614us, mobilenetv2_avgpool_ws=8.378us (batch=10)
# Added ~5% margin for performance targets
# fmt: off
POOL_PERF_CONFIGS = [
    # HEIGHT SHARDED - ResNet50 stem maxpool (batch=16, matches actual ResNet50 inference)
    {
        "test_name": "resnet50_maxpool_hs",
        "pool_type": "max",
        "batch_size": 16,
        "input_channels": 64,
        "input_height": 112,
        "input_width": 112,
        "kernel_h": 3,
        "kernel_w": 3,
        "stride_h": 2,
        "stride_w": 2,
        "pad_h": 1,
        "pad_w": 1,
        "dilation_h": 1,
        "dilation_w": 1,
        "ceil_mode": False,
        "in_dtype": ttnn.bfloat16,
        "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        "perf_targets": {"wh": 131, "bh_p150": 131},
    },
    # HEIGHT SHARDED - VGG16 first maxpool (224x224 -> 112x112)
    {
        "test_name": "vgg16_maxpool_hs",
        "pool_type": "max",
        "batch_size": 1,
        "input_channels": 64,
        "input_height": 224,
        "input_width": 224,
        "kernel_h": 2,
        "kernel_w": 2,
        "stride_h": 2,
        "stride_w": 2,
        "pad_h": 0,
        "pad_w": 0,
        "dilation_h": 1,
        "dilation_w": 1,
        "ceil_mode": False,
        "in_dtype": ttnn.bfloat16,
        "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        "perf_targets": {"wh": 32.7, "bh_p150": 32.7},
    },
    # BLOCK SHARDED - Gemma3 avgpool for vision token compression
    {
        "test_name": "gemma3_avgpool_bs",
        "pool_type": "avg",
        "batch_size": 1,
        "input_channels": 1152,
        "input_height": 64,
        "input_width": 64,
        "kernel_h": 4,
        "kernel_w": 4,
        "stride_h": 4,
        "stride_w": 4,
        "pad_h": 0,
        "pad_w": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None,
        "in_dtype": ttnn.bfloat16,
        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        "perf_targets": {"wh": 18, "bh_p150": 18},
    },
    # WIDTH SHARDED - MobileNetV2 global avgpool (batch=10, matches actual MobileNetV2 inference)
    {
        "test_name": "mobilenetv2_avgpool_ws",
        "pool_type": "avg",
        "batch_size": 10,
        "input_channels": 1280,
        "input_height": 7,
        "input_width": 7,
        "kernel_h": 7,
        "kernel_w": 7,
        "stride_h": 1,
        "stride_w": 1,
        "pad_h": 0,
        "pad_w": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None,
        "in_dtype": ttnn.bfloat8_b,
        "shard_layout": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        "perf_targets": {"wh": 8.77, "bh_p150": 8.77},
    },
]
# fmt: on

# Applied to both upper and lower bounds
THRESHOLD_PERCENT = 0.05

# Build performance targets per architecture (in us) from global config
PERF_TARGETS_WH = {config["test_name"]: config["perf_targets"]["wh"] for config in POOL_PERF_CONFIGS}
PERF_TARGETS_BH_P150 = {config["test_name"]: config["perf_targets"]["bh_p150"] for config in POOL_PERF_CONFIGS}


def test_run_pool2d_ops(device):
    """
    Consolidated performance test that runs all pool configurations in a single process.
    Uses Tracy signposts to mark each configuration's measurement region.
    """
    torch.manual_seed(0)

    if not (is_blackhole() or is_wormhole_b0()):
        pytest.skip("Test is only for Blackhole and Wormhole B0 architectures")

    if device.core_grid.y != 8 and is_wormhole_b0():
        pytest.skip("Needs 8x8 grid for wormhole_b0")

    warmup_iterations = 2

    for config in POOL_PERF_CONFIGS:
        test_name = config["test_name"]
        pool_type = config["pool_type"]

        # Create input tensor directly on device
        tt_input_tensor = ttnn.empty(
            (1, 1, config["input_height"] * config["input_width"] * config["batch_size"], config["input_channels"]),
            dtype=config["in_dtype"],
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        # Warmup iterations (without signposts)
        for _ in range(warmup_iterations):
            if pool_type == "max":
                tt_output = ttnn.max_pool2d(
                    input_tensor=tt_input_tensor,
                    batch_size=config["batch_size"],
                    input_h=config["input_height"],
                    input_w=config["input_width"],
                    channels=config["input_channels"],
                    kernel_size=[config["kernel_h"], config["kernel_w"]],
                    stride=[config["stride_h"], config["stride_w"]],
                    padding=[config["pad_h"], config["pad_h"], config["pad_w"], config["pad_w"]],
                    dilation=[config["dilation_h"], config["dilation_w"]],
                    ceil_mode=config["ceil_mode"],
                    applied_shard_scheme=config["shard_layout"],
                    deallocate_input=False,
                    config_tensor_in_dram=True,
                )
            else:  # avg pool
                tt_output = ttnn.avg_pool2d(
                    input_tensor=tt_input_tensor,
                    batch_size=config["batch_size"],
                    input_h=config["input_height"],
                    input_w=config["input_width"],
                    channels=config["input_channels"],
                    kernel_size=[config["kernel_h"], config["kernel_w"]],
                    stride=[config["stride_h"], config["stride_w"]],
                    padding=[config["pad_h"], config["pad_h"], config["pad_w"], config["pad_w"]],
                    applied_shard_scheme=config["shard_layout"],
                    deallocate_input=False,
                    config_tensor_in_dram=True,
                )
            ttnn.deallocate(tt_output)

        # Measured iteration with signposts
        signpost(f"{test_name}-start")
        if pool_type == "max":
            tt_output_tensor_on_device = ttnn.max_pool2d(
                input_tensor=tt_input_tensor,
                batch_size=config["batch_size"],
                input_h=config["input_height"],
                input_w=config["input_width"],
                channels=config["input_channels"],
                kernel_size=[config["kernel_h"], config["kernel_w"]],
                stride=[config["stride_h"], config["stride_w"]],
                padding=[config["pad_h"], config["pad_h"], config["pad_w"], config["pad_w"]],
                dilation=[config["dilation_h"], config["dilation_w"]],
                ceil_mode=config["ceil_mode"],
                applied_shard_scheme=config["shard_layout"],
                deallocate_input=False,
                config_tensor_in_dram=True,
            )
        else:  # avg pool
            tt_output_tensor_on_device = ttnn.avg_pool2d(
                input_tensor=tt_input_tensor,
                batch_size=config["batch_size"],
                input_h=config["input_height"],
                input_w=config["input_width"],
                channels=config["input_channels"],
                kernel_size=[config["kernel_h"], config["kernel_w"]],
                stride=[config["stride_h"], config["stride_w"]],
                padding=[config["pad_h"], config["pad_h"], config["pad_w"], config["pad_w"]],
                applied_shard_scheme=config["shard_layout"],
                deallocate_input=False,
                config_tensor_in_dram=True,
            )
        signpost(f"{test_name}-end")

        # Deallocate output tensor without reading it
        ttnn.deallocate(tt_output_tensor_on_device)


@pytest.mark.models_device_performance_bare_metal
def test_pool2d_device_perf():
    if is_blackhole():
        perf_targets = PERF_TARGETS_BH_P150
    elif is_wormhole_b0():
        perf_targets = PERF_TARGETS_WH
    else:
        pytest.skip("Test is only for Blackhole and Wormhole B0 architectures")

    profiler = BenchmarkProfiler()
    step_name = "pool_all_perf"

    subdir = "pool_perf"
    command = (
        "pytest tests/ttnn/perf_tests/operations/pool/test_pool2d_device_perf.py::test_run_pool2d_ops --timeout=60"
    )
    # Pool operations are named Pool2D in the profiler output
    op_name = "Pool2D"

    logger.info(f"Command: {command}")

    # Clear any previous profiler artifacts
    clear_profiler_runtime_artifacts()

    # Run the consolidated test with profiling
    profiler.start(step_name)
    run_device_profiler(command, subdir, device_analysis_types=["device_kernel_duration"])
    profiler.end(step_name)

    # Get the latest CSV file
    csv_path = get_latest_ops_log_filename(subdir)
    logger.info(f"Parsing results from: {csv_path}")

    # Extract ops between signposts
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
