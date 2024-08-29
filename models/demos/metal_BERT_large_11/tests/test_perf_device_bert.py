# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report
from models.utility_functions import skip_for_grayskull, is_wormhole_b0, is_blackhole


def run_bert_perf(batch_size, test, expected_perf):
    subdir = "bert"
    num_iterations = 4
    margin = 0.03
    command = f"pytest models/demos/metal_BERT_large_11/tests/test_bert.py::test_bert[{test}]"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)
    prep_device_perf_report(
        model_name=f"bert11",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=test,
    )


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Incorrect device metrics for wormhole b0")
@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "batch_size, test, expected_perf",
    [
        # [9, "BERT_LARGE-batch_9-MIXED_PRECISION_BATCH9", 70],
        [8, "BERT_LARGE-batch_8-MIXED_PRECISION_BATCH8", 170],
        # [7, "BERT_LARGE-batch_7-BFLOAT8_B-SHARDED", 240],
        [12, "BERT_LARGE-batch_12-BFLOAT8_B-SHARDED", 400],
    ],
)
def test_perf_device_bare_metal(batch_size, test, expected_perf):
    run_bert_perf(batch_size, test, expected_perf)


@pytest.mark.skip("#7525: Hangs non-deterministically on device perf")
@skip_for_grayskull("Incorrect device metrics for grayskull")
@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "batch_size, test, expected_perf",
    [
        [7, "BERT_LARGE-batch_8-BFLOAT8_B-SHARDED", 280],
        [8, "BERT_LARGE-batch_8-BFLOAT8_B-SHARDED", 340],
    ],
)
def test_perf_device_bare_metal_wh(batch_size, test, expected_perf):
    run_bert_perf(batch_size, test, expected_perf)
