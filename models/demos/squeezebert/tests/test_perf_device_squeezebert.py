# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf
from models.utility_functions import is_grayskull


@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "batch_size, test",
    [
        [8, "sequence_size=384-batch_size=8-model_name=squeezebert/squeezebert-uncased"],
    ],
)
def test_perf_device_bare_metal(batch_size, test):
    subdir = "ttnn_squeezebert"
    num_iterations = 1
    margin = 0.03
    expected_perf = 102.7 if is_grayskull() else 299.7

    command = f"pytest tests/ttnn/integration_tests/squeezebert/test_ttnn_squeezebert.py::test_squeezebert_for_question_answering"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols, assert_on_fail=True)
    prep_device_perf_report(
        model_name=f"ttnn_squeezebert_{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=test.replace("/", "_"),
    )
