# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
from loguru import logger

from models.common.utility_functions import is_blackhole, is_wormhole_b0
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf

# @pytest.mark.parametrize(
#     "batch_size, expected_perf",
#     [
#         [1, 236.95],
#     ],
# )


def _expected_device_kernel_samples_per_s():
    """AVG DEVICE KERNEL SAMPLES/S baselines differ by architecture (see yolov8l README platforms)."""
    if is_wormhole_b0():
        return 236.95
    if is_blackhole():
        # Blackhole (e.g. Quiet Box / T3020) — lower than Wormhole n150 baseline above.
        return 178.5
    pytest.skip("YOLOv8l device perf baseline not defined for this architecture")


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_yolov8l(batch_size):
    expected_perf = _expected_device_kernel_samples_per_s()
    subdir = "ttnn_yolov8l"
    num_iterations = 1
    margin = 0.05
    command = f"pytest models/demos/yolov8l/tests/pcc/test_yolov8l.py::test_yolov8l_640"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols, assert_on_fail=True)

    logger.info(f"{expected_results}")

    prep_device_perf_report(
        model_name=f"ttnn_yolov8l{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )
