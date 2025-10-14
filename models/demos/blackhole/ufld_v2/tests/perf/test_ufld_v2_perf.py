# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
from loguru import logger

from models.common.utility_functions import run_for_blackhole
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf


@run_for_blackhole()
@pytest.mark.parametrize(
    "batch_size, expected_perf,test",
    [
        [1, 341, "UFLD-v2"],
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_ufld_v2(batch_size, expected_perf, test):
    subdir = "ttnn_UFLD_v2"
    num_iterations = 1
    margin = 0.03

    command = f"pytest models/demos/blackhole/ufld_v2/tests/pcc/test_ttnn_ufld_v2.py::test_ufld_v2_model"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols, assert_on_fail=True)

    logger.info(f"{expected_results}")

    prep_device_perf_report(
        model_name=f"ttnn_ufld_v2{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=test.replace("/", "_"),
    )
