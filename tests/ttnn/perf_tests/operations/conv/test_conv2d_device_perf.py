# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest
import torch
import ttnn
from loguru import logger
from models.perf.benchmarking_utils import BenchmarkProfiler
from tracy import signpost
from tracy.common import clear_profiler_runtime_artifacts
from tracy.process_model_log import get_latest_ops_log_filename, run_device_profiler
from ttnn.device import is_blackhole, is_wormhole_b0

# Global conv configuration array with performance targets (in us)
# fmt: off
CONV_PERF_CONFIGS = [
    # HEIGHT SHARDED, SDXL VAE - compute bound
    {
        "test_name": "vae_sdxl_hs",
        "batch_size": 1, "input_channels": 4, "output_channels": 512,
        "input_height": 128, "input_width": 128,
        "kernel_h": 3, "kernel_w": 3, "stride_h": 1, "stride_w": 1,
        "pad_h": 1, "pad_w": 1, "groups": 1,
        "weights_dtype": ttnn.bfloat8_b, "output_dtype": ttnn.bfloat16,
        "math_fidelity": ttnn.MathFidelity.LoFi,
        "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        "act_block_h_override": 0, "act_block_w_div": 1,
        "enable_activation_reuse": False, "enable_act_double_buffer": False,
        "enable_weights_double_buffer": False, "fp32_accum": False,
        "perf_targets": {"wh": 67, "bh_p150": 25},
    },
    # HEIGHT SHARDED, Unet - dm bound (activation reuse)
    {
        "test_name": "unet_hs",
        "batch_size": 1, "input_channels": 4, "output_channels": 16,
        "input_height": 1056, "input_width": 160,
        "kernel_h": 3, "kernel_w": 3, "stride_h": 1, "stride_w": 1,
        "pad_h": 1, "pad_w": 1, "groups": 1,
        "weights_dtype": ttnn.bfloat8_b, "output_dtype": ttnn.bfloat8_b,
        "math_fidelity": ttnn.MathFidelity.LoFi,
        "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        "act_block_h_override": 0, "act_block_w_div": 1,
        "enable_activation_reuse": True, "enable_act_double_buffer": True,
        "enable_weights_double_buffer": False, "fp32_accum": False,
        "perf_targets": {"wh": 87, "bh_p150": 36},
    },
    # BLOCK SHARDED, SDXL
    {
        "test_name": "sdxl_bs",
        "batch_size": 1, "input_channels": 640, "output_channels": 640,
        "input_height": 64, "input_width": 64,
        "kernel_h": 3, "kernel_w": 3, "stride_h": 2, "stride_w": 2,
        "pad_h": 1, "pad_w": 1, "groups": 1,
        "weights_dtype": ttnn.bfloat8_b, "output_dtype": ttnn.bfloat16,
        "math_fidelity": ttnn.MathFidelity.HiFi2,
        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        "act_block_h_override": 0, "act_block_w_div": 1,
        "enable_activation_reuse": False, "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True, "fp32_accum": True,
        "perf_targets": {"wh": 115, "bh_p150": 53},
    },
    # WIDTH SHARDED, Segformer
    {
        "test_name": "segformer_ws",
        "batch_size": 1, "input_channels": 576, "output_channels": 576,
        "input_height": 8, "input_width": 8,
        "kernel_h": 3, "kernel_w": 3, "stride_h": 1, "stride_w": 1,
        "pad_h": 0, "pad_w": 0, "groups": 576,
        "weights_dtype": ttnn.bfloat16, "output_dtype": ttnn.bfloat16,
        "math_fidelity": ttnn.MathFidelity.LoFi,
        "shard_layout": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        "act_block_h_override": 0, "act_block_w_div": 1,
        "enable_activation_reuse": False, "enable_act_double_buffer": False,
        "enable_weights_double_buffer": False, "fp32_accum": False,
        "perf_targets": {"wh": 74, "bh_p150": 37},
    },
]
# fmt: on

# Applied to both upper and lower bounds
THRESHOLD_PERCENT = 0.05

# Build performance targets per architecture (in us) from global config
PERF_TARGETS_WH = {config["test_name"]: config["perf_targets"]["wh"] for config in CONV_PERF_CONFIGS}
PERF_TARGETS_BH_P150 = {config["test_name"]: config["perf_targets"]["bh_p150"] for config in CONV_PERF_CONFIGS}


def extract_ops_between_signposts(csv_path, op_name="Conv2dDeviceOperation"):
    """
    Extract operation durations between signpost pairs from a Tracy CSV file.

    Parses a Tracy profiler CSV file and extracts operation durations that occur between
    "test-name-start" and "test-name-end" signpost markers. The test name is extracted
    from the signpost OP CODE.

    Args:
        csv_path (str): Path to the Tracy profiler CSV file.
        op_name (str, optional): Name of the operation to extract (default: "Conv2dDeviceOperation").

    Returns:
        dict: Dictionary mapping test names to lists of durations in nanoseconds.
              Format: {test_name: [duration_ns, ...]}

    Example:
        results = extract_ops_between_signposts("ops_perf_results_2025_01_15.csv")
        # Returns: {"test_conv1": [12345.0, 12456.0], "test_conv2": [23456.0]}
    """
    df = pd.read_csv(csv_path)
    results = {}
    current_region = None

    for _, row in df.iterrows():
        if row["OP TYPE"] == "signpost":
            op_code = row["OP CODE"]
            if op_code.endswith("-start"):
                current_region = op_code[:-6]  # Remove "-start" suffix to get test_name
            elif op_code.endswith("-end"):
                current_region = None
        elif current_region and row["OP CODE"] == op_name:
            if current_region not in results:
                results[current_region] = []
            results[current_region].append(float(row["DEVICE KERNEL DURATION [ns]"]))

    return results


def test_run_conv2d_ops(device):
    """
    Consolidated performance test that runs all conv configurations in a single process.
    Uses Tracy signposts to mark each configuration's measurement region.
    """
    torch.manual_seed(0)

    if not (is_blackhole() or is_wormhole_b0()):
        pytest.skip("Test is only for Blackhole and Wormhole B0 architectures")

    if device.core_grid.y != 8 and is_wormhole_b0():
        pytest.skip("Needs 8x8 grid for wormhole_b0")

    warmup_iterations = 2

    for config in CONV_PERF_CONFIGS:
        test_name = config["test_name"]

        # Create input tensor directly on device
        tt_input_tensor = ttnn.empty(
            (1, 1, config["input_height"] * config["input_width"] * config["batch_size"], config["input_channels"]),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        # Create weights
        torch_weight_tensor = torch.randn(
            (
                config["output_channels"],
                config["input_channels"] // config["groups"],
                config["kernel_h"],
                config["kernel_w"],
            ),
            dtype=torch.bfloat16,
        )
        tt_weight_tensor = ttnn.from_torch(
            torch_weight_tensor,
            config["weights_dtype"] if config["weights_dtype"] != ttnn.bfloat8_b else ttnn.float32,
        )

        # Create bias
        torch_bias_tensor = torch.randn((1, 1, 1, config["output_channels"]), dtype=torch.bfloat16)
        tt_bias_tensor = ttnn.from_torch(
            torch_bias_tensor,
            config["weights_dtype"] if config["weights_dtype"] != ttnn.bfloat8_b else ttnn.float32,
        )

        # Configure Conv2d
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=config["weights_dtype"],
            shard_layout=config["shard_layout"],
            deallocate_activation=False,
            transpose_shards=False,
            reshard_if_not_optimal=False,
            override_sharding_config=False,
            enable_act_double_buffer=config["enable_act_double_buffer"],
            enable_weights_double_buffer=config["enable_weights_double_buffer"],
            act_block_h_override=config["act_block_h_override"],
            act_block_w_div=config["act_block_w_div"],
            enable_activation_reuse=config["enable_activation_reuse"],
            config_tensors_in_dram=True,
        )

        compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=config["math_fidelity"],
            math_approx_mode=True,
            fp32_dest_acc_en=config["fp32_accum"],
            packer_l1_acc=False,
        )

        # Warmup iterations (without signposts)
        for _ in range(warmup_iterations):
            tt_output = ttnn.conv2d(
                input_tensor=tt_input_tensor,
                weight_tensor=tt_weight_tensor,
                device=device,
                in_channels=config["input_channels"],
                out_channels=config["output_channels"],
                bias_tensor=tt_bias_tensor,
                kernel_size=(config["kernel_h"], config["kernel_w"]),
                stride=(config["stride_h"], config["stride_w"]),
                padding=(config["pad_h"], config["pad_w"]),
                batch_size=config["batch_size"],
                input_height=config["input_height"],
                input_width=config["input_width"],
                conv_config=conv_config,
                compute_config=compute_config,
                groups=config["groups"],
                dtype=config["output_dtype"],
                slice_config=ttnn.Conv2dL1FullSliceConfig,
            )
            ttnn.deallocate(tt_output)

        # Measured iteration with signposts
        signpost(f"{test_name}-start")
        tt_output_tensor_on_device = ttnn.conv2d(
            input_tensor=tt_input_tensor,
            weight_tensor=tt_weight_tensor,
            device=device,
            in_channels=config["input_channels"],
            out_channels=config["output_channels"],
            bias_tensor=tt_bias_tensor,
            kernel_size=(config["kernel_h"], config["kernel_w"]),
            stride=(config["stride_h"], config["stride_w"]),
            padding=(config["pad_h"], config["pad_w"]),
            batch_size=config["batch_size"],
            input_height=config["input_height"],
            input_width=config["input_width"],
            conv_config=conv_config,
            compute_config=compute_config,
            groups=config["groups"],
            dtype=config["output_dtype"],
            slice_config=ttnn.Conv2dL1FullSliceConfig,
        )
        signpost(f"{test_name}-end")

        # Deallocate output tensor without reading it
        ttnn.deallocate(tt_output_tensor_on_device)


@pytest.mark.models_device_performance_bare_metal
def test_conv2d_device_perf():
    if is_blackhole():
        perf_targets = PERF_TARGETS_BH_P150
    elif is_wormhole_b0():
        perf_targets = PERF_TARGETS_WH
    else:
        pytest.skip("Test is only for Blackhole and Wormhole B0 architectures")

    profiler = BenchmarkProfiler()
    step_name = "conv_all_perf"

    subdir = "conv_perf"
    command = "pytest tests/ttnn/perf_tests/operations/conv/test_conv2d_device_perf.py::test_run_conv2d_ops"
    op_name = "Conv2dDeviceOperation"

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
