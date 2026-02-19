# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Swin-L Backbone Device Performance Test.

Measures raw device kernel execution time using the device profiler.
Runs the backbone PCC test with profiler enabled and reports device FPS.

Usage:
    pytest models/experimental/swin_l/tests/perf/test_swin_l_device_perf.py -v -m models_device_performance_bare_metal
"""

import pytest

from models.perf.device_perf_utils import prep_device_perf_report, run_device_perf


@pytest.mark.timeout(600)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_swin_l():
    """
    Device performance test for the Swin-L backbone.

    Runs the backbone PCC test with device profiler to measure raw kernel execution time.
    Input: 800x1333 (DINO-5scale standard).
    """
    batch_size = 1
    subdir = "ttnn_swin_l_backbone_800x1333"
    num_iterations = 1

    command = "pytest models/experimental/swin_l/tests/pcc/test_ttnn_backbone.py" "::test_ttnn_swin_l_backbone_e2e -sv"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    actual_perf = post_processed_results.get(inference_time_key, 0)

    print(f"\n{'='*60}")
    print("Swin-L Backbone Device Performance (800x1333)")
    print(f"{'='*60}")
    print(f"  Measured:  {actual_perf:.2f} samples/s")
    print(f"{'='*60}\n")

    prep_device_perf_report(
        model_name=f"ttnn_swin_l_backbone_batch{batch_size}_800x1333",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results={},
        comments="swin_l_backbone_800x1333",
    )
