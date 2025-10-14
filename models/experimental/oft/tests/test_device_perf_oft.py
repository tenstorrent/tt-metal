# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report

OFT_DEVICE_TEST_TOTAL_ITERATIONS = 3


@pytest.mark.models_device_performance_bare_metal
def test_device_perf_oft_oftnet():
    expected_device_perf_cycles_per_iteration = 342_772_374
    command = f"pytest models/experimental/oft/tests/test_oftnet.py::test_oftnet -k bfp16_use_device_oft"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    batch_size = 1

    inference_time_key = "AVG DEVICE KERNEL DURATION [ns]"
    post_processed_results = run_device_perf(
        command, subdir="oft", num_iterations=OFT_DEVICE_TEST_TOTAL_ITERATIONS, cols=cols, batch_size=batch_size
    )
    expected_perf_cols = {inference_time_key: expected_device_perf_cycles_per_iteration}
    expected_results = check_device_perf(
        post_processed_results, margin=0.015, expected_perf_cols=expected_perf_cols, assert_on_fail=True
    )
    prep_device_perf_report(
        model_name=f"oft_oftnet",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )


@pytest.mark.models_device_performance_bare_metal
def test_device_perf_oft_decoder():
    expected_device_perf_cycles_per_iteration = 4_767_804
    command = f"pytest models/experimental/oft/tests/test_encoder.py::test_decode"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    batch_size = 1

    inference_time_key = "AVG DEVICE KERNEL DURATION [ns]"
    post_processed_results = run_device_perf(
        command, subdir="oft", num_iterations=OFT_DEVICE_TEST_TOTAL_ITERATIONS, cols=cols, batch_size=batch_size
    )
    expected_perf_cols = {inference_time_key: expected_device_perf_cycles_per_iteration}
    expected_results = check_device_perf(
        post_processed_results, margin=0.015, expected_perf_cols=expected_perf_cols, assert_on_fail=True
    )
    prep_device_perf_report(
        model_name=f"oft_decode",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )


@pytest.mark.models_device_performance_bare_metal
def test_device_perf_oft_full():
    expected_device_perf_cycles_per_iteration = 345_592_041
    command = f"pytest models/experimental/oft/demo/demo.py::test_demo_inference"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    batch_size = 1

    inference_time_key = "AVG DEVICE KERNEL DURATION [ns]"
    post_processed_results = run_device_perf(
        command, subdir="oft", num_iterations=OFT_DEVICE_TEST_TOTAL_ITERATIONS, cols=cols, batch_size=batch_size
    )
    expected_perf_cols = {inference_time_key: expected_device_perf_cycles_per_iteration}
    expected_results = check_device_perf(
        post_processed_results, margin=0.015, expected_perf_cols=expected_perf_cols, assert_on_fail=True
    )
    prep_device_perf_report(
        model_name=f"oft_full_demo",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )
