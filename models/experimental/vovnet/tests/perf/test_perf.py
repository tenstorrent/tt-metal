# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report


@pytest.mark.parametrize(
    "batch_size, expected_perf",
    [
        [1, 159.00],
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_vovnet(batch_size, expected_perf):
    subdir = "ttnn_vovnet"
    num_iterations = 1
    margin = 0.03

    command = f"pytest models/experimental/vovnet/tests/pcc/test_tt_vovnet.py::test_vovnet_model_inference"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols, assert_on_fail=False)

    logger.info(f"{expected_results}")

    prep_device_perf_report(
        model_name=f"ttnn_vovnet{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )
