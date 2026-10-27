# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device-kernel throughput assertion for b8s256.

Not on the tier-3 cron - tier 3 is exempt from the device-perf pipeline - so
this runs manually, or automatically if the model is ever promoted."""

import pytest
from loguru import logger

from models.common.utility_functions import is_wormhole_b0, run_for_wormhole_b0
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "batch_size, seq_len, expected_perf, test",
    [
        [8, 256, 374.3, "modernbert"],
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_modernbert(batch_size, seq_len, expected_perf, test):
    subdir = "ttnn_modernbert_model"
    num_iterations = 1
    margin = 0.03
    expected_perf = expected_perf if is_wormhole_b0() else 0

    # The profiler captures every op in the process, so the target has to be a
    # single forward pass rather than a timing loop. test_modernbert_profile.py
    # exists for exactly this.
    command = (
        "pytest models/experimental/modernbert/tests/test_modernbert_profile.py"
        f"::test_modernbert_single_forward[{seq_len}-{batch_size}]"
    )
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols, assert_on_fail=True)

    logger.info(f"{expected_results}")

    prep_device_perf_report(
        model_name=f"ttnn_modernbert{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=f"{test}_seq{seq_len}",
    )
