# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import math
import pandas as pd
import pytest
import ttnn
from loguru import logger

from tracy.process_model_log import (
    get_latest_ops_log_filename,
    run_device_profiler,
)

from tests.ttnn.nightly.unit_tests.operations.experimental.test_mla_wqkv_ab import SHAPE2TIME, LARGE_N_TILES_PER_CORE


@pytest.mark.parametrize(
    "M, K, N, L, pos",
    SHAPE2TIME.keys(),
)
@pytest.mark.parametrize("check_accuracy", [True], ids=["check_accuracy_True"])
@pytest.mark.parametrize("dump_outputs", [False], ids=["dump_outputs_False"])
def test_mla_wqkv_ab_performance(M, K, N, L, pos, check_accuracy, dump_outputs):
    command = (
        "pytest "
        "tests/ttnn/nightly/unit_tests/operations/experimental/test_mla_wqkv_ab.py::"
        f"test_mla_wqkv_ab[dump_outputs_{dump_outputs}-check_accuracy_{check_accuracy}-M={M}-K={K}-N={N}-L={L}-pos={pos}-dispatch_row]"
    )
    run_device_profiler(command, "ttnn_mla_wqkv_ab_performance", device_analysis_types=["device_kernel_duration"])

    r = post_process_ops_log("ttnn_mla_wqkv_ab_performance", float_columns=["DEVICE KERNEL DURATION [ns]"])
    duration_us = r["DEVICE KERNEL DURATION [ns]"].sum() / 1000.0
    logger.info(f"Duration per layer: {duration_us / L} us")
    logger.warning(f"Total Duration: {duration_us} us")

    bytes_per_tile = 1024 + 64  # bfloat8_b
    num_cores = 12

    Kt, Nt = math.ceil(K / ttnn.TILE_SIZE), math.ceil(N / ttnn.TILE_SIZE)
    padded_w_tiles_per_core = Kt * LARGE_N_TILES_PER_CORE
    total_bytes_transferred = L * num_cores * padded_w_tiles_per_core * bytes_per_tile
    realized_bandwidth = int(total_bytes_transferred / (duration_us * 1000))
    logger.warning(f"Realized Bandwidth: {realized_bandwidth} GB/s")

    useful_w_tiles = L * Kt * Nt
    useful_bytes_used = useful_w_tiles * bytes_per_tile
    useful_bandwidth = int(useful_bytes_used / (duration_us * 1000))
    logger.warning(f"Useful Bandwidth: {useful_bandwidth} GB/s")

    assert (
        duration_us < SHAPE2TIME[(M, K, N, L, pos)]
    ), f"Performance {duration_us} us is greater than expected {SHAPE2TIME[(M, K, N, L, pos)]} us"


def post_process_ops_log(output_logs_subdir: str, float_columns: list[str]):
    filename = get_latest_ops_log_filename(output_logs_subdir)

    df = pd.read_csv(filename)
    return {col: df[col].astype(float).to_numpy() for col in float_columns}
