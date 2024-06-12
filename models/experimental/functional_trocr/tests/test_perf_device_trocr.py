# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report


@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "batch_size, test, expected_perf",
    [
        [1, "microsoft/trocr-base-handwritten", 0.16],
    ],
)
def test_perf_device_bare_metal(batch_size, test, expected_perf):
    subdir = "functional_trocr"
    num_iterations = 1
    margin = 0.03
    command = f'pytest --disable-warnings --input-path="models/sample_data/iam_ocr_image.jpg" models/experimental/functional_trocr/demo/ttnn_trocr_demo.py::test_trocr_demo[{test}]'
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]
    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)

    prep_device_perf_report(
        model_name=f"functional_trocr_{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=test.replace("/", "_"),
    )
