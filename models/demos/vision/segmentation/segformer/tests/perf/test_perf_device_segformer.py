# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import models.perf.device_perf_utils as perf_utils
from models.common.utility_functions import is_blackhole

# Arch-conditional target — the linear_config_1024 / GEGLU subblock-volume fix landed in
# 47df299ef68 ("model configs: kernel-A/B-verified subblock-volume fixes (single-chip
# WH)") roughly doubled segformer device-kernel throughput on Blackhole only. WH stays at
# the prior baseline of 211 samples/s. BH was Tracy A/B-verified at 391.71 samples/s on
# p100a; the prior 211 target tripped BH's "performance suspiciously fast" upper-bound
# assertion (measured 391.99). Single parametrize at module load — is_blackhole() is
# evaluated once when the test module imports.
_EXPECTED_PERF = 392 if is_blackhole() else 211


@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "batch_size, expected_perf",
    [
        [1, _EXPECTED_PERF],
    ],
)
def test_perf_device_segformer_segmentation(batch_size, expected_perf, model_location_generator):
    subdir = "segformer"
    num_iterations = 1
    margin = 0.05

    command = f"pytest models/demos/vision/segmentation/segformer/tests/pcc/test_segformer_for_semantic_segmentation.py::test_segformer_for_semantic_segmentation"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = perf_utils.run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = perf_utils.check_device_perf(
        post_processed_results, margin, expected_perf_cols, assert_on_fail=True
    )
    perf_utils.prep_device_perf_report(
        model_name=f"segformer_for_semantic_segmentation",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=f"{num_iterations}_iterations",
    )
