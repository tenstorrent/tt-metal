# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
from loguru import logger

from models.common.utility_functions import is_wormhole_b0
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf


@pytest.mark.parametrize(
    "name_suffix,batch_size,resolution,expected_perf,test_filter",
    [
        ["b1_640", 1, 640, 27.8, "test_yolov11 and resolution1 and not dp_batch8"],
        ["b1_1280", 1, 1280, 8.3, "test_yolov11 and resolution0 and not dp_batch8"],
        ["b8_640", 8, 640, 27.8, "test_yolov11_dp_batch8 and 640"],
        ["b8_1280", 8, 1280, 8.3, "test_yolov11_dp_batch8 and 1280"],
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_yolov11(name_suffix, batch_size, resolution, expected_perf, test_filter):
    subdir = "ttnn_yolov11l"
    num_iterations = 1
    margin = 0.03
    expected_perf = expected_perf if is_wormhole_b0() else 0

    command = "pytest models/demos/yolov11l/tests/pcc/test_ttnn_yolov11.py " f'-k "{test_filter}"'

    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(
        command,
        subdir,
        num_iterations,
        cols,
        batch_size,
        op_support_count=6000,
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
