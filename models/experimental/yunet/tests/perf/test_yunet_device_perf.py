# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
YUNet Device Performance Test.

This test measures the raw device kernel execution time using the device profiler.
It runs the model without trace (per-op dispatch) and reports device FPS.

Usage:
    # Default 640x640
    pytest models/experimental/yunet/tests/perf/test_yunet_device_perf.py -v -m models_device_performance_bare_metal

    # Run with 320x320
    pytest models/experimental/yunet/tests/perf/test_yunet_device_perf.py -v -m models_device_performance_bare_metal --input-size 320
"""

import pytest

from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf


# Expected performance for each input size (samples/s)
EXPECTED_PERF_BLACKHOLE = {
    320: 1150,
    640: 450,
}

EXPECTED_PERF_WORMHOLE = {
    320: 480,
    640: 170,
}


def get_expected_perf(input_h):
    """Get expected perf based on device architecture."""
    import ttnn

    is_wormhole = "wormhole_b0" in ttnn.get_arch_name()

    if is_wormhole:
        return EXPECTED_PERF_WORMHOLE.get(input_h, 150)
    else:
        return EXPECTED_PERF_BLACKHOLE.get(input_h, 300)


@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_yunet(input_size):
    """
    Device performance test for YUNet.

    Runs the PCC test with device profiler enabled to measure raw kernel execution time.

    Args:
        input_size: Tuple of (height, width) from conftest.py --input-size option
    """
    input_h, input_w = input_size
    expected_perf = get_expected_perf(input_h)
    batch_size = 1

    subdir = f"ttnn_YUNet_{input_h}x{input_w}"
    num_iterations = 1
    margin = 0.05  # 5% margin to allow 1100-1200 for 320, 400-500 for 640

    # Run the PCC test with correct input size
    command = f"pytest models/experimental/yunet/tests/pcc/test_pcc.py::test_yunet_pcc --input-size {input_h}"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols, assert_on_fail=True)

    # Log actual measured performance
    actual_perf = post_processed_results.get(inference_time_key, 0)
    print(f"\n{'='*50}")
    print(f"YUNet Device Performance ({input_h}x{input_w})")
    print(f"{'='*50}")
    print(f"  Measured:  {actual_perf:.1f} samples/s")
    print(f"  Expected:  {expected_perf} samples/s (±{margin*100:.0f}%)")
    print(f"{'='*50}\n")

    prep_device_perf_report(
        model_name=f"ttnn_yunet_batch{batch_size}_{input_h}x{input_w}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=f"YUNet_{input_h}x{input_w}",
    )
