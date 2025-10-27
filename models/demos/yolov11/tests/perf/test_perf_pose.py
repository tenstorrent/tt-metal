# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Performance test for YOLO11 Pose Estimation

Tests device-level performance metrics for pose estimation model.
Measures kernel execution time and samples per second.
"""

import pytest
from loguru import logger

from models.common.utility_functions import is_wormhole_b0
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf


@pytest.mark.parametrize(
    "batch_size, expected_perf",
    [
        [1, 276],  # Expected samples/s (adjust based on actual measurements)
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_yolov11_pose(batch_size, expected_perf):
    """
    Test device performance for YOLO11 Pose model

    Args:
        batch_size: Batch size for inference
        expected_perf: Expected performance (samples/second)
    """
    subdir = "ttnn_yolov11_pose"
    num_iterations = 1
    margin = 0.03  # 3% margin for performance variance
    expected_perf = expected_perf if is_wormhole_b0() else 0

    # Run the PCC test to measure performance
    command = f"pytest models/demos/yolov11/tests/pcc/test_ttnn_yolov11_pose_model.py::test_yolov11_pose_model"

    # Columns to extract from device profiling
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    # Key metric to track
    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    # Run performance measurement
    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)

    logger.info(f"Performance results: {expected_results}")

    # Generate performance report
    prep_device_perf_report(
        model_name=f"ttnn_yolov11_pose_batch{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="YOLO11 Pose Estimation - 17 keypoints per person",
    )
