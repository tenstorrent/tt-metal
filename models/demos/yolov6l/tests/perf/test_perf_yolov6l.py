# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
from loguru import logger

from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf
from models.utility_functions import is_wormhole_b0, run_for_wormhole_b0


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "batch_size, expected_perf",
    [
        [1, 110.3],
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_yolov6l(batch_size, expected_perf):
    subdir = "ttnn_yolov6l"
    num_iterations = 1
    margin = 0.05
    expected_perf = expected_perf if is_wormhole_b0() else 0

    command = f"pytest models/demos/yolov6l/tests/pcc/test_ttnn_yolov6l.py::test_yolov6l"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols, assert_on_fail=True)

    logger.info(f"{expected_results}")

    prep_device_perf_report(
        model_name=f"ttnn_yolov6l{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )
