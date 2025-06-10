# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

import models.perf.device_perf_utils as perf_utils


@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "batch_size, expected_perf",
    [
        [1, 102.0],
    ],
)
def test_perf_device_bare_metal(batch_size, expected_perf):
    subdir = "segformer"
    num_iterations = 1
    margin = 0.05

    command = f"pytest tests/ttnn/integration_tests/segformer/test_segformer_for_semantic_segmentation.py"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = perf_utils.run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = perf_utils.check_device_perf(
        post_processed_results, margin, expected_perf_cols, assert_on_fail=True
    )
    perf_utils.prep_device_perf_report(
        model_name=f"segformer_for_semantic_segmentation",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=f"{num_iterations}_iterations",
    )
