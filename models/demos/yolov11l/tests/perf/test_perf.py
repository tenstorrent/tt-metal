# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
from loguru import logger

from models.common.utility_functions import is_wormhole_b0
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf


@pytest.mark.parametrize(
    "name_suffix,batch_size,resolution,expected_perf,test_selector,op_support_count",
    [
        ["b1_640", 1, 640, 42.0, "test_yolov11 and resolution1 and not dp_batch", 6000],
        ["b1_1280", 1, 1280, 10.8, "test_yolov11 and resolution0 and not dp_batch", 6000],
        ["b2_640", 2, 640, 42.0, "dp_batch2 and n300_1x2 and 640", 6000],
        ["b2_1280", 2, 1280, 10.8, "dp_batch2 and n300_1x2 and 1280", 6000],
        ["b4_640", 4, 640, 42.0, "dp_batch4 and wh_1x4 and 640", 6000],
        ["b4_1280", 4, 1280, 10.8, "dp_batch4 and wh_1x4 and 1280", 6000],
        ["b8_640", 8, 640, 42.0, "dp_batch8 and t3k_1x8 and 640", 6000],
        ["b8_1280", 8, 1280, 10.8, "dp_batch8 and t3k_1x8 and 1280", 6000],
    ],
    ids=[
        "b1_640",
        "b1_1280",
        "b2_640",
        "b2_1280",
        "b4_640",
        "b4_1280",
        "b8_640",
        "b8_1280",
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_yolov11(
    name_suffix, batch_size, resolution, expected_perf, test_selector, op_support_count
):
    subdir = "ttnn_yolov11l"
    num_iterations = 1
    margin = 0.05
    expected_perf = expected_perf if is_wormhole_b0() else 0

    command = "pytest models/demos/yolov11l/tests/pcc/test_ttnn_yolov11.py " f'-k "{test_selector}"'

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
        model_name=f"ttnn_yolov11l_{name_suffix}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=f"resolution={resolution}",
    )
