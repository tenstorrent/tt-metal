# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.perf.device_perf_utils import run_device_perf
from ttnn.device import is_blackhole, is_wormhole_b0

# Performance threshold tolerance in us
THRESHOLD = 0.1

# Performance targets per architecture (in us)
PERF_TARGETS_WH = {
    "vae_sdxl_hs": 70,
    "unet_conv_group_4_act_reuse": 90,
    "sdxl_bs": 150,
    "segformer_ws": 100,
}

# Blackhole targets initialized with Wormhole values
PERF_TARGETS_BH = {
    "vae_sdxl_hs": 70,
    "unet_conv_group_4_act_reuse": 90,
    "sdxl_bs": 150,
    "segformer_ws": 100,
}


@pytest.mark.parametrize(
    "test_name, test_function, filter_args, description",
    [
        # HEIGHT SHARDED - compute bound
        (
            "vae_sdxl_hs",
            "test_conv2d_vae_sdxl",
            "-k 'vae_dec_4x512_128x128_hs_none and no_auto_slice'",
            "VAE SDXL height sharded convolution",
        ),
        # HEIGHT SHARDED - dm bound
        (
            "unet_conv_group_4_act_reuse",
            "test_conv2d_activation_reuse_unet_conv_group_4",
            "-k 'act_reuse_on'",
            "UNet grouped convolution with activation reuse (16→64 channels, 1056x160, 3x3 kernel, groups=4, height sharded, 63 cores)",
        ),
        # BLOCK SHARDED
        (
            "sdxl_bs",
            "test_conv2d_sdxl",
            "-k 'bs_640x640_64x64_s2'",
            "SDXL block sharded convolution",
        ),
        # WIDTH SHARDED
        (
            "segformer_ws",
            "test_conv_for_segformer_512x512",
            "-k 'segformer_576x576_8x8_ws and no_auto_shard'",
            "Segformer width sharded convolution (576x576, 8x8 kernel, width sharded, auto_shard=False)",
        ),
    ],
    ids=["vae_sdxl_hs", "unet_conv_group_4_act_reuse", "sdxl_bs", "segformer_ws"],
)
@pytest.mark.models_device_performance_bare_metal
def test_conv_perf(
    test_name,
    test_function,
    filter_args,
    description,
):
    if is_blackhole():
        perf_target_us = PERF_TARGETS_BH[test_name]
    elif is_wormhole_b0():
        perf_target_us = PERF_TARGETS_WH[test_name]
    else:
        pytest.skip("Test is only for Blackhole and Wormhole B0 architectures")

    profiler = BenchmarkProfiler()
    step_name = f"conv_{test_name}_perf"
    warmup_iterations = 3

    subdir = "conv_perf"
    command = f"pytest tests/ttnn/nightly/unit_tests/operations/conv/test_conv2d.py::{test_function} " f"{filter_args}"
    columns = ["DEVICE KERNEL"]
    op_name = "Conv2dDeviceOperation"

    logger.info(f"Command: {command}")

    profiler.start(step_name)
    results = run_device_perf(
        command, subdir, warmup_iterations, columns, batch_size=1, op_name=op_name, has_signposts=False
    )
    profiler.end(step_name)

    duration_col = columns[0] + " DURATION [ns]"
    measured_min = results[f"MIN {duration_col}"]
    measured_max = results[f"MAX {duration_col}"]
    measured_avg = results[f"AVG {duration_col}"]
    measured_avg_us = measured_avg / 1000

    logger.info(f"[{test_name}] Measured performance: {measured_avg_us:.3f} us vs. target: {perf_target_us} us")
    logger.info(f"[{test_name}] Performance range: {measured_min/1000:.3f} - {measured_max/1000:.3f} us")

    assert (
        measured_avg_us < perf_target_us + THRESHOLD
    ), f"[{test_name}] Performance target not met: {measured_avg_us:.3f} us > {perf_target_us} us"
