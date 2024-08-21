# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report


@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "batch_size, test, expected_perf",
    [
        [20, "False-LoFi-activations_BFLOAT8_B-weights_BFLOAT8_B-batch_20-device_params0", 7700],
    ],
)
def test_perf_device_bare_metal(batch_size, test, expected_perf):
    subdir = "resnet50"
    num_iterations = 4
    margin = 0.03
    command = f"pytest models/demos/resnet/tests/test_metal_resnet50_performant.py::test_run_resnet50_inference[{test}]"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]
    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)
    prep_device_perf_report(
        model_name=f"resnet50_batch_size{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=test,
    )
