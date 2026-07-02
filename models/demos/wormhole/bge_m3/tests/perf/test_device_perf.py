# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest

from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf


@pytest.mark.parametrize(
    "batch_size, model_name, expected_perf",
    [
        (1, "bge_m3", 50.0),
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_bge_m3(batch_size, model_name, expected_perf):
    subdir = model_name
    num_iterations = 1
    margin = 0.04

    command = f"pytest models/demos/wormhole/bge_m3/tests/pcc/test_model.py"

    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    # PCC pytest has no Tracy signposts; True raises IndexError in post_process_ops_log.
    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size, has_signposts=False)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)
    prep_device_perf_report(
        model_name=f"ttnn_functional_{model_name}_{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )
