# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report


@pytest.mark.parametrize(
    "batch_size, test, expected_perf",
    [
        [1, "BFLOAT16-L1-falcon_7b-layers_32-prefill_seq256", 3.666],
        [32, "BFLOAT16-L1-falcon_7b-layers_32-decode_batch32", 139],
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal(batch_size, test, expected_perf):
    subdir = f"ttnn_falcon7b"
    num_iterations = 3
    margin = 0.03
    command = f"pytest models/demos/ttnn_falcon7b/tests/test_metal_falcon.py::test_perf_bare_metal[{test}]"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)
    prep_device_perf_report(
        model_name=f"falcon7b_batch_size{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=test,
    )
