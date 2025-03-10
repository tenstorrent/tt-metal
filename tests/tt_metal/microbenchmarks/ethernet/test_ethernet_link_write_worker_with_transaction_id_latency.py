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
    profile_results,
)

from models.utility_functions import is_grayskull

from tt_metal.tools.profiler.common import PROFILER_LOGS_DIR, PROFILER_DEVICE_SIDE_LOG

profiler_log_path = PROFILER_LOGS_DIR / PROFILER_DEVICE_SIDE_LOG

FILE_NAME = PROFILER_LOGS_DIR / "test_ethernet_link_write_worker_latency.csv"

if os.path.exists(FILE_NAME):
    os.remove(FILE_NAME)


def run_erisc_write_worker_latency(
    benchmark_type, sample_count, sample_size_expected_latency, channel_count, disable_trid, file_name
):
    os.system(f"rm -rf {os.environ['TT_METAL_HOME']}/generated/profiler/.logs/profile_log_device.csv")

    test_latency = 1
    sample_size = sample_size_expected_latency[0]
    sample_size_expected_latency = sample_size_expected_latency[1]
    diff = sample_size_expected_latency * 0.1
    expected_latency_lower_bound = sample_size_expected_latency - diff
    expected_latency_upper_bound = sample_size_expected_latency + diff

    ARCH_NAME = os.getenv("ARCH_NAME")
    cmd = f"TT_METAL_DEVICE_PROFILER=1 \
            {os.environ['TT_METAL_HOME']}/build/test/tt_metal/perf_microbenchmark/ethernet/test_ethernet_write_worker_latency_no_edm_{ARCH_NAME} \
                {benchmark_type} \
                {sample_count} \
                {sample_size} \
                {channel_count} \
                {test_latency} \
                {disable_trid} "
    rc = os.system(cmd)
    if rc != 0:
        logger.info("Error in running the test")
        assert False

    main_loop_latency = profile_results(
        sample_size, sample_count, channel_count, benchmark_type, test_latency, file_name
    )
    logger.info(f"sender_loop_latency {main_loop_latency}")

    assert expected_latency_lower_bound <= main_loop_latency <= expected_latency_upper_bound


# uni-direction test for eth-sender <---> eth-receiver
@pytest.mark.skipif(is_grayskull(), reason="Unsupported on GS")
@pytest.mark.parametrize("sample_count", [1])
@pytest.mark.parametrize("channel_count", [16])
@pytest.mark.parametrize(
    "sample_size_expected_latency",
    [
        (16, 894.0),
        (128, 911.0),
        (256, 966.0),
        (512, 984.0),
        (1024, 1245.0),
        (2048, 1479.0),
        (4096, 1803.0),
        (8192, 2451.0),
    ],
)
def test_erisc_latency_uni_dir(sample_count, sample_size_expected_latency, channel_count):
    benchmark_type_id = 0
    disable_trid = 0  # don't care in this case
    run_erisc_write_worker_latency(
        benchmark_type_id,
        sample_count,
        sample_size_expected_latency,
        channel_count,
        disable_trid,
        FILE_NAME,
    )


# uni-direction test for eth-sender <---> eth-receiver ---> worker
@pytest.mark.skipif(is_grayskull(), reason="Unsupported on GS")
@pytest.mark.parametrize("sample_count", [1])
@pytest.mark.parametrize("channel_count", [16])
@pytest.mark.parametrize("disable_trid", [0])
@pytest.mark.parametrize(
    "sample_size_expected_latency",
    [
        (16, 984.0),
        (128, 1002.0),
        (256, 1019.0),
        (512, 1074.0),
        (1024, 1335.0),
        (2048, 1609.0),
        (4096, 2018.0),
        (8192, 2811.0),
    ],
)
def test_erisc_write_worker_latency_uni_dir(sample_count, sample_size_expected_latency, channel_count, disable_trid):
    benchmark_type_id = 2
    run_erisc_write_worker_latency(
        benchmark_type_id,
        sample_count,
        sample_size_expected_latency,
        channel_count,
        disable_trid,
        FILE_NAME,
    )
