# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Performance tests for RF-DETR Medium.
"""

import pytest

from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf


@pytest.mark.parametrize(
    "batch_size, model_name, expected_perf",
    [
        (1, "rfdetr_medium", 26.1),  # device kernel: ~38.3ms → ~26.1 samples/s
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_rfdetr_medium(batch_size, model_name, expected_perf):
    subdir = model_name
    num_iterations = 1
    margin = 0.03

    command = "pytest models/experimental/rfdetr_medium/tests/pcc/test_ttnn_rfdetr_e2e.py" "::test_rfdetr_medium_e2e -v"

    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]
    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)
    prep_device_perf_report(
        model_name=f"ttnn_rfdetr_medium_{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="RF-DETR Medium detection, initial bringup",
    )
