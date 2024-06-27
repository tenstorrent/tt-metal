# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from models.utility_functions import skip_for_wormhole_b0
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report


@skip_for_wormhole_b0()
@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "batch_size, test, expected_perf",
    [
        [8, "3-198-8-facebook/deit-base-distilled-patch16-224", 374],
    ],
)
def test_perf_device_bare_metal(batch_size, test, expected_perf):
    subdir = "ttnn_optimized_sharded_deit"
    num_iterations = 1
    margin = 0.03
    # command = f"pytest models/experimental/functional_roberta/demo/demo.py::test_demo_squadv2[{test}]"
    command = f"pytest models/experimental/functional_deit/demo/demo_with_teacher.py::test_demo[{test}]"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)
    prep_device_perf_report(
        model_name=f"ttnn_optimized_sharded_deit_{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=test.replace("/", "_"),
    )
