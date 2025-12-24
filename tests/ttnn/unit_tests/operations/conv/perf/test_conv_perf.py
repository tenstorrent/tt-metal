# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.perf.device_perf_utils import run_device_perf
from ttnn.device import is_blackhole, is_wormhole_b0

# Performance threshold tolerance in us
THRESHOLD = 1

# Performance targets per architecture (in us)
PERF_TARGETS_WH = {
    "vae_sdxl_hs": 67,
    "unet_hs": 71,
    "sdxl_bs": 149,
    "segformer_ws": 74,
}
PERF_TARGETS_BH_P150 = {
    "vae_sdxl_hs": 26,
    "unet_hs": 54,
    "sdxl_bs": 62,
    "segformer_ws": 38,
}


@pytest.mark.parametrize(
    "test_name, test_function, filter_args",
    [
        # HEIGHT SHARDED, SDXL - compute bound
        (
            "vae_sdxl_hs",
            "test_conv2d_vae_sdxl",
            "-k 'vae_dec_4x512_128x128_hs_none and no_auto_slice'",
        ),
        # HEIGHT SHARDED, Unet - dm bound
        (
            "unet_hs",
            "test_conv2d_activation_reuse_unet_conv_group_4",
            "-k 'act_reuse_on'",
        ),
        # BLOCK SHARDED, SDXL
        (
            "sdxl_bs",
            "test_conv2d_sdxl",
            "-k 'bs_640x640_64x64_s2'",
        ),
        # WIDTH SHARDED, Segformer
        (
            "segformer_ws",
            "test_conv_for_segformer_512x512",
            "-k 'segformer_576x576_8x8_ws and no_auto_shard'",
        ),
    ],
    ids=["vae_sdxl_hs", "unet_hs", "sdxl_bs", "segformer_ws"],
)
@pytest.mark.models_device_performance_bare_metal
def test_conv_perf(test_name, test_function, filter_args):
    if is_blackhole():
        perf_target_us = PERF_TARGETS_BH_P150[test_name]
    elif is_wormhole_b0():
        perf_target_us = PERF_TARGETS_WH[test_name]
    else:
        pytest.skip("Test is only for Blackhole and Wormhole B0 architectures")

    profiler = BenchmarkProfiler()
    step_name = f"conv_{test_name}_perf"
    warmup_iterations = 2

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
