# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import pytest

from tracy.process_model_log import run_device_profiler
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf
import models.perf.device_perf_utils


def _run_device_profiler_op_support_count(*args, **kwargs):
    if "op_support_count" not in kwargs:
        kwargs["op_support_count"] = 10000
    return run_device_profiler(*args, **kwargs)


models.perf.device_perf_utils.run_device_profiler = _run_device_profiler_op_support_count


@pytest.mark.parametrize(
    "batch_size, model_name, expected_perf",
    [
        (1, "bevformerv2", 0.105),  # Expected performance: ~0.33 samples/s (measured from device kernel duration)
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_bevformerv2(batch_size, model_name, expected_perf):
    subdir = model_name
    num_iterations = 1
    margin = 0.04

    command = f"pytest models/experimental/BEVFormerV2/tests/pcc/test_bevformer_v2.py::test_bevformerv2"

    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size, has_signposts=True)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)
    prep_device_perf_report(
        model_name=f"ttnn_functional_{model_name}_{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )
