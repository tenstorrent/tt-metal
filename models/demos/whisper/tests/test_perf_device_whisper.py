# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from models.utility_functions import is_grayskull
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report


@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize("batch_size", [8])
def test_perf_device_bare_metal(batch_size, reset_seeds):
    subdir = "ttnn_whisper_optimized_"
    margin = 0.03
    num_iterations = 1

    expected_perf = 13.38 if is_grayskull() else 35.07
    command = (
        f"pytest tests/ttnn/integration_tests/whisper/test_ttnn_optimized_functional_whisper.py::test_ttnn_whisper"
    )
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]
    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}
    model_name = f"ttnn_optimized_whisper_{batch_size}"

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols, assert_on_fail=True)
    prep_device_perf_report(
        model_name=model_name,
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=model_name.replace("/", "_"),
    )
