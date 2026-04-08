# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
from loguru import logger

from models.common.utility_functions import is_wormhole_b0
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf


@pytest.mark.parametrize(
    "name_suffix,batch_size,resolution,expected_perf,test_selector,op_support_count",
    [
        [
            "b1_640",
            1,
            640,
            0.0,
            "test_yolov8l and 640 and not dp_batch8 and not test_yolov8l_640 and not test_yolov8l_1280",
        ],
        [
            "b1_1280",
            1,
            1280,
            0.0,
            "models/demos/yolov8l/tests/pcc/test_yolov8l.py::test_yolov8l[1280-l1_1280_for_all_res-True]",
            12000,
        ],
        ["b8_640", 8, 640, 0.0, "test_yolov8l_dp_batch8 and 640", 12000],
        ["b8_1280", 8, 1280, 0.0, "test_yolov8l_dp_batch8 and 1280 and not 640", 12000],
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_yolov8l(name_suffix, batch_size, resolution, expected_perf, test_selector, op_support_count):
    subdir = "ttnn_yolov8l"
    num_iterations = 1
    margin = 0.03
    expected_perf = expected_perf if is_wormhole_b0() else 0
    if test_selector.endswith(".py") or "::" in test_selector:
        command = f"pytest {test_selector}"
    else:
        command = "pytest models/demos/yolov8l/tests/pcc/test_yolov8l.py " f'-k "{test_selector}"'
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(
        command,
        subdir,
        num_iterations,
        cols,
        batch_size,
        op_support_count=op_support_count,
    )
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)

    logger.info(f"{expected_results}")

    prep_device_perf_report(
        model_name=f"ttnn_yolov8l_{name_suffix}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=f"resolution={resolution}",
    )
