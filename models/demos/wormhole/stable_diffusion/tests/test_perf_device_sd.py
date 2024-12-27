# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from models.utility_functions import run_for_wormhole_b0
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report


@run_for_wormhole_b0()
@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "batch_size, test, expected_perf",
    [
        [2, "input_shape0-2-5-device_params0", 16.65],
    ],
)
def test_perf_device(batch_size, test, expected_perf):
    subdir = f"ttnn_sd_{batch_size}"
    margin = 0.03
    num_iterations = 1

    expected_perf = 16.65

    command = (
        f"pytest models/demos/wormhole/stable_diffusion_dp/tests/test_sd_performant.py::test_run_sd_inference[{test}]"
    )

    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]
    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}
    model_name = f"ttnn_sd_{batch_size}"

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols, assert_on_fail=True)
    prep_device_perf_report(
        model_name=model_name,
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=model_name.replace("/", "_"),
    )
