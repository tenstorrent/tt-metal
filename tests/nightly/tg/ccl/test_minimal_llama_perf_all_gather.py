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

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = f"/tmp/all_gather_results_{timestamp}.csv"


def append_csv_row(row):
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["num_links", "num_workers", "topology", "llama_case", "avg_us"])

        writer.writerow(row)


# num_links_ids = ["1link", "2link", "4link"]
num_links_ids = ["1link", "4link"]
# num_workers_ids = ["2worker", "4worker"]
num_workers_ids = ["1worker", "2worker", "4worker"]
topology_ids = ["1D_Ring"]
# llama_ids = ["llama_1", "llama_2", "llama_3", "llama_4", "llama_5", "llama_6"]
llama_ids = ["llama_1", "llama_2", "llama_3", "llama_4", "llama_6"]
llama_ids = ["llama_5"]


@pytest.mark.parametrize("warmup_iters", [3])
@pytest.mark.models_device_performance_bare_metal
def test_all_gather_llama_sweep(warmup_iters):
    profiler = BenchmarkProfiler()
    cols = ["DEVICE KERNEL"]
    op_name = "AllGatherAsync"

    logger.info(f"CSV file: {csv_path}")

    for link, worker, topo, llama_case in itertools.product(num_links_ids, num_workers_ids, topology_ids, llama_ids):
        case_filter = f"{link} and {worker} and {topo} and {llama_case}"
        subdir = f"llama_ccl_perf/{link}_{worker}_{topo}_{llama_case}"

        cmd = (
            "pytest ./tests/nightly/tg/ccl/test_minimal_all_gather_async.py::test_all_gather_llama "
            f'-k "{case_filter}"'
        )

        logger.info(f"Running {case_filter}")

        try:
            results = run_device_perf_detailed(
                cmd, subdir, cols, op_name, has_signposts=True, warmup_iters=warmup_iters
            )
            avg_us = results[cols[0]]["MAX"] / 1000

        except Exception as e:
            logger.error(f"FAILED {case_filter}: {e}")
            avg_us = "FAIL"

        append_csv_row([link, worker, topo, llama_case, avg_us])
        logger.info(f"{case_filter} -> {avg_us}")
