# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
from loguru import logger

from models.common.utility_functions import run_for_wormhole_b0
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "batch_size",
    [1],
)
@pytest.mark.parametrize(
    "model_task",
    [
        "segment",  # To run the test for instance segmentation
        "detect",  # To run the test for Object Detection
    ],
    ids=["segment", "detect"],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_yolov9c(model_task, batch_size):
    subdir = "ttnn_yolov9c"
    num_iterations = 1
    margin = 0.03
    enable_segment = model_task == "segment"
    expected_perf = 55.1 if enable_segment else 90

    command = (
        f"pytest models/demos/yolov9c/tests/pcc/test_ttnn_yolov9c.py::test_yolov9c[device_params0-segment-True]"
        if model_task == "segment"
        else f"pytest models/demos/yolov9c/tests/pcc/test_ttnn_yolov9c.py::test_yolov9c[device_params0-detect-True]"
    )
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols, assert_on_fail=True)

    logger.info(f"{expected_results}")

    prep_device_perf_report(
        model_name=f"ttnn_functional_yolov9c{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )
