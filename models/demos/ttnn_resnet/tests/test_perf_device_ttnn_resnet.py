# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report
from models.utility_functions import run_for_wormhole_b0
from models.utility_functions import (
    is_grayskull,
    is_wormhole_b0,
)


@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "batch_size, test, expected_perf",
    [
        [20, "20-act_dtype0-weight_dtype0-math_fidelity0-device_params0", 6600],
        [16, "16-act_dtype1-weight_dtype1-math_fidelity1-device_params1", 5020],
    ],
)
def test_perf_device_bare_metal(batch_size, test, expected_perf):
    if is_grayskull():
        if batch_size == 16:
            pytest.skip("Skipping batch size 16 for Grayskull")

    if is_wormhole_b0():
        if batch_size == 20:
            pytest.skip("Skipping batch size 20 for Wormhole B0 due to fitting issue")

    subdir = "resnet50"
    num_iterations = 4
    margin = 0.03
    command = (
        f"pytest models/demos/ttnn_resnet/tests/test_ttnn_resnet50_performant.py::test_run_resnet50_inference[{test}]"
    )
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]
    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size, True)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)
    prep_device_perf_report(
        model_name=f"ttnn_resnet50_batch_size{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=test,
    )
