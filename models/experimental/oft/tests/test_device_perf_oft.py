# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report


def run_device_perf_tests(
    command, expected_device_perf_cycles_per_iteration, subdir, model_name, num_iterations=1, batch_size=1, margin=0.015
):
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL DURATION [ns]"
    post_processed_results = run_device_perf(
        command, subdir=subdir, num_iterations=num_iterations, cols=cols, batch_size=batch_size
    )
    expected_perf_cols = {inference_time_key: expected_device_perf_cycles_per_iteration}
    expected_results = check_device_perf(
        post_processed_results, margin=margin, expected_perf_cols=expected_perf_cols, assert_on_fail=True
    )
    prep_device_perf_report(
        model_name=model_name,
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )


@pytest.mark.parametrize(
    "command, expected_device_perf_cycles_per_iteration, subdir, model_name, num_iterations, batch_size, margin",
    [
        (
            "pytest models/experimental/oft/tests/pcc/test_oftnet.py::test_oftnet -k bfp16_use_device_oft",
            253_123_435,
            "oft_oftnet",
            "oft_oftnet",
            1,
            1,
            0.015,
        ),
        (
            "pytest models/experimental/oft/tests/pcc/test_encoder.py::test_decode",
            4_764_102,
            "oft_decoder",
            "oft_decoder",
            1,
            1,
            0.015,
        ),
        (
            "pytest models/experimental/oft/demo/demo.py::test_demo_inference",
            256_114_255,
            "oft_full_demo",
            "oft_full_demo",
            1,
            1,
            0.015,
        ),
    ],
    ids=[
        "device_perf_oft_oftnet",
        "device_perf_oft_decoder",
        "device_perf_oft_full",
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_device_perf_oft(
    command, expected_device_perf_cycles_per_iteration, subdir, model_name, num_iterations, batch_size, margin
):
    run_device_perf_tests(
        command,
        expected_device_perf_cycles_per_iteration,
        subdir=subdir,
        model_name=model_name,
        num_iterations=num_iterations,
        batch_size=batch_size,
        margin=margin,
    )
