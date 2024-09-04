# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report
from models.utility_functions import run_for_wormhole_b0, run_for_grayskull


@run_for_grayskull()
@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "batch_size, test, expected_perf",
    [
        [20, "20-act_dtype0-weight_dtype0-math_fidelity0-device_params0", 6600],
    ],
)
def test_perf_device_bare_metal_gs(batch_size, test, expected_perf):
    subdir = "resnet50"
    num_iterations = 4
    margin = 0.03
    command = f"pytest models/demos/ttnn_resnet/tests/test_ttnn_resnet50_performant.py::test_run_resnet50_inference_gs[{test}]"
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


@run_for_wormhole_b0()
@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "batch_size, test, expected_perf",
    [
        [16, "16-act_dtype0-weight_dtype0-math_fidelity0-device_params0", 5020],
    ],
)
def test_perf_device_bare_metal_wh(batch_size, test, expected_perf):
    subdir = "resnet50"
    num_iterations = 4
    margin = 0.03
    command = f"pytest models/demos/ttnn_resnet/tests/test_ttnn_resnet50_performant.py::test_run_resnet50_inference_wh[{test}]"
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
