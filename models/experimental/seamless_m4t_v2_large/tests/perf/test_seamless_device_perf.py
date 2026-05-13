# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Seamless M4T v2 Large — device performance test.

Measures raw device kernel execution time using the device profiler. Runs the T2TT PCC test with
profiler enabled and reports device throughput (samples/s).

Usage::

    pytest models/experimental/seamless_m4t_v2_large/tests/perf/test_seamless_device_perf.py \\
        -v -m models_device_performance_bare_metal
"""

import pytest

from models.perf.device_perf_utils import prep_device_perf_report, run_device_perf


@pytest.mark.timeout(3600)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_seamless():
    """
    Device performance test for Seamless M4T v2 Large (T2TT path).

    Runs the full T2TT PCC forward with device profiler to measure raw kernel execution time.
    Inner pytest uses ``--timeout=0`` so the PCC test's own timeout governs the workload.
    """
    batch_size = 1
    subdir = "ttnn_seamless_m4t_v2_large_t2tt"
    num_iterations = 1
    command = (
        "pytest --timeout=0 "
        "models/experimental/seamless_m4t_v2_large/tests/pcc/test_seamless_m4t_v2_model.py"
        "::test_t2tt -sv"
    )
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]
    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    actual_perf = post_processed_results.get(inference_time_key, 0)
    print(f"\n{'='*60}")
    print("Seamless M4T v2 Large Device Performance (T2TT PCC forward)")
    print(f"{'='*60}")
    print(f"  Measured:  {actual_perf:.2f} samples/s")
    print(f"{'='*60}\n")

    prep_device_perf_report(
        model_name=f"ttnn_seamless_m4t_v2_large_batch{batch_size}_t2tt",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results={},
        comments="seamless_m4t_v2_large_t2tt",
    )
