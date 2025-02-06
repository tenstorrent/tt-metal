# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys

from loguru import logger
import pytest
import csv
from tt_metal.tools.profiler.process_device_log import import_log_run_stats
import tt_metal.tools.profiler.device_post_proc_config as device_post_proc_config
from tests.tt_metal.microbenchmarks.ethernet.test_ethernet_link_write_worker_with_transaction_id_common import (
    run_erisc_write_worker,
)

from models.utility_functions import is_grayskull

from tt_metal.tools.profiler.common import PROFILER_LOGS_DIR, PROFILER_DEVICE_SIDE_LOG

profiler_log_path = PROFILER_LOGS_DIR / PROFILER_DEVICE_SIDE_LOG

FILE_NAME = PROFILER_LOGS_DIR / "test_ethernet_link_write_worker_bandwidth.csv"

if os.path.exists(FILE_NAME):
    os.remove(FILE_NAME)


# uni-direction test for eth-sender <---> eth-receiver ---> worker
@pytest.mark.skipif(is_grayskull(), reason="Unsupported on GS")
@pytest.mark.parametrize("sample_count", [256])
@pytest.mark.parametrize("channel_count", [16])
@pytest.mark.parametrize("num_directions", [1])
@pytest.mark.parametrize("test_latency", [0])
@pytest.mark.parametrize("enable_worker", [1])
@pytest.mark.parametrize(
    "sample_size_expected_bw",
    [(16, 0.21), (128, 1.72), (256, 3.44), (512, 6.89), (1024, 11.73), (2048, 11.83), (4096, 12.04), (8192, 12.07)],
)
def test_erisc_write_worker_bw_uni_dir(
    sample_count, sample_size_expected_latency, channel_count, num_directions, enable_worker, test_latency
):
    run_erisc_write_worker(
        sample_count,
        sample_size_expected_latency,
        channel_count,
        num_directions,
        test_latency,
        enable_worker,
        FILE_NAME,
    )


# bi-direction test for eth-sender <---> eth-receiver ---> worker
@pytest.mark.skipif(is_grayskull(), reason="Unsupported on GS")
@pytest.mark.parametrize("sample_count", [1000])
@pytest.mark.parametrize("channel_count", [16])
@pytest.mark.parametrize("num_directions", [2])
@pytest.mark.parametrize("test_latency", [0])
@pytest.mark.parametrize("enable_worker", [1])
@pytest.mark.parametrize(
    "sample_size_expected_latency",
    [(16, 0.13), (128, 1.03), (256, 2.08), (512, 4.15), (1024, 8.31), (2048, 11.40), (4096, 11.82)],
)
def test_erisc_write_worker_bw_bi_dir(
    sample_count, sample_size_expected_latency, channel_count, num_directions, test_latency, enable_worker
):
    run_erisc_write_worker(
        sample_count,
        sample_size_expected_latency,
        channel_count,
        num_directions,
        test_latency,
        enable_worker,
        FILE_NAME,
    )


# uni-direction test for eth-sender <---> eth-receiver
@pytest.mark.skipif(is_grayskull(), reason="Unsupported on GS")
@pytest.mark.parametrize("sample_count", [256])
@pytest.mark.parametrize("channel_count", [16])
@pytest.mark.parametrize("num_directions", [1])
@pytest.mark.parametrize("test_latency", [0])
@pytest.mark.parametrize("enable_worker", [0])
@pytest.mark.parametrize(
    "sample_size_expected_latency",
    [(16, 0.28), (128, 2.25), (256, 4.39), (512, 8.35), (1024, 11.74), (2048, 11.84), (4096, 12.04), (8192, 12.07)],
)
def test_erisc_bw_uni_dir(
    sample_count, sample_size_expected_latency, channel_count, num_directions, test_latency, enable_worker
):
    run_erisc_write_worker(
        sample_count,
        sample_size_expected_latency,
        channel_count,
        num_directions,
        test_latency,
        enable_worker,
        FILE_NAME,
    )


# bi-direction test for eth-sender <---> eth-receiver
@pytest.mark.skipif(is_grayskull(), reason="Unsupported on GS")
@pytest.mark.parametrize("sample_count", [1000])
@pytest.mark.parametrize("channel_count", [16])
@pytest.mark.parametrize("num_directions", [2])
@pytest.mark.parametrize("test_latency", [0])
@pytest.mark.parametrize("enable_worker", [0])
@pytest.mark.parametrize(
    "sample_size_expected_latency",
    [(16, 0.19), (128, 1.59), (256, 3.19), (512, 6.39), (1024, 10.9), (2048, 11.4), (4096, 11.82)],
)
def test_erisc_bw_bi_dir(
    sample_count, sample_size_expected_latency, channel_count, num_directions, test_latency, enable_worker
):
    run_erisc_write_worker(
        sample_count,
        sample_size_expected_latency,
        channel_count,
        num_directions,
        test_latency,
        enable_worker,
        FILE_NAME,
    )
