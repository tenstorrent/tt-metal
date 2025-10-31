# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import itertools
import csv
import os
from datetime import datetime
from loguru import logger
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from models.perf.device_perf_utils import run_device_perf_detailed
from models.perf.device_perf_utils import run_device_perf_detailed2

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = f"/tmp/all_send_recv_results_{timestamp}.csv"


@pytest.mark.parametrize("warmup_iters", [0])
@pytest.mark.models_device_performance_bare_metal
def test_all_send_recv_llama_sweep(warmup_iters):
    profiler = BenchmarkProfiler()
    cols = ["DEVICE KERNEL"]
    op_name1 = "SendAsync"
    op_name2 = "RecvAsync"

    logger.info(f"CSV file: {csv_path}")

    subdir = f"llama_ccl_perf_send_recv/{timestamp}"

    cmd = "pytest tests/ttnn/unit_tests/operations/ccl/test_send_recv_async.py::test_send_recv_llama"

    try:
        results1, results2 = run_device_perf_detailed2(
            cmd, subdir, cols, op_name1, op_name2, has_signposts=True, warmup_iters=warmup_iters
        )
        avg_us1 = results1[cols[0]]["MAX"] / 1000
        avg_us2 = results2[cols[0]]["MAX"] / 1000

    except Exception as e:
        logger.error(f"FAILED: {e}")
        avg_us1 = "FAIL"
        avg_us2 = "FAIL"

    logger.info(f"avg_us1 -> {avg_us1}")
    logger.info(f"avg_us2 -> {avg_us2}")
