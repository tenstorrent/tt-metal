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

FILE_NAME = PROFILER_LOGS_DIR / "test_ethernet_link_write_worker_latency.csv"

if os.path.exists(FILE_NAME):
    os.remove(FILE_NAME)


# uni-direction test for eth-sender <---> eth-receiver ---> worker
@pytest.mark.skipif(is_grayskull(), reason="Unsupported on GS")
@pytest.mark.parametrize("sample_count", [1])
@pytest.mark.parametrize("channel_count", [16])
@pytest.mark.parametrize("num_directions", [1])
@pytest.mark.parametrize("test_latency", [1])
@pytest.mark.parametrize("enable_worker", [1])
@pytest.mark.parametrize(
    "sample_size_expected_latency",
    [(16, 97.2), (128, 97.2), (256, 98.0), (512, 98.0), (1024, 99.0), (2048, 173.0), (4096, 340.0), (8192, 678.5)],
)
def test_erisc_write_worker_latency_uni_dir(
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


# bi-direction test for eth-sender <---> eth-receiver ---> worker
@pytest.mark.skipif(is_grayskull(), reason="Unsupported on GS")
@pytest.mark.parametrize("sample_count", [1])
@pytest.mark.parametrize("channel_count", [16])
@pytest.mark.parametrize("num_directions", [2])
@pytest.mark.parametrize("test_latency", [1])
@pytest.mark.parametrize("enable_worker", [1])
@pytest.mark.parametrize(
    "sample_size_expected_latency",
    [(16, 148.0), (128, 148.0), (256, 148.7), (512, 148.8), (1024, 149.2), (2048, 178.2), (4096, 344.2)],
)
def test_erisc_write_worker_latency_bi_dir(
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
