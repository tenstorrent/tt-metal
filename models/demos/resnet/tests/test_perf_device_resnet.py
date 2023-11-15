# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import pytest
from loguru import logger

from tt_metal.tools.profiler.common import clear_profiler_runtime_artifacts

from tt_metal.tools.profiler.process_model_log import (
    post_process_ops_log,
    run_device_profiler,
    get_samples_per_s,
)


@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "batch_size, test, expected_perf",
    [
        [8, "HiFi4-activations_BFLOAT16-weights_BFLOAT16-batch_8", 2080],
        # [8, "HiFi2-activations_BFLOAT8_B-weights_BFLOAT8_B-batch_8", 2575],
        [16, "HiFi2-activations_BFLOAT8_B-weights_BFLOAT8_B-batch_16", 3140],
        [8, "LoFi-activations_BFLOAT8_B-weights_BFLOAT8_B-batch_8", 2675],
        [16, "LoFi-activations_BFLOAT8_B-weights_BFLOAT8_B-batch_16", 3355],
    ],
)
def test_perf_device_bare_metal(batch_size, test, expected_perf):
    subdir = "resnet50"
    num_iterations = 4
    margin = 0.03
    command = f"pytest models/demos/resnet/tests/test_metal_resnet50.py::test_run_resnet50_inference[{test}]"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]
    duration_cols = [col + " DURATION [ns]" for col in cols]
    samples_cols = [col + " SAMPLES/S" for col in cols]

    clear_profiler_runtime_artifacts()

    results = {}
    for d_col in duration_cols:
        results[f"AVG {d_col}"] = 0
        results[f"MIN {d_col}"] = float("inf")
        results[f"MAX {d_col}"] = -float("inf")

    for _ in range(num_iterations):
        run_device_profiler(command, subdir)
        r = post_process_ops_log(subdir, duration_cols)
        for d_col in duration_cols:
            results[f"AVG {d_col}"] += r[d_col]
            results[f"MIN {d_col}"] = min(results[f"MIN {d_col}"], r[d_col])
            results[f"MAX {d_col}"] = max(results[f"MAX {d_col}"], r[d_col])

    post_processed_results = {}
    for s_col, d_col in zip(samples_cols, duration_cols):
        post_processed_results[f"AVG {s_col}"] = get_samples_per_s(results[f"AVG {d_col}"] / num_iterations, batch_size)
        post_processed_results[f"MIN {s_col}"] = get_samples_per_s(results[f"MAX {d_col}"], batch_size)
        post_processed_results[f"MAX {s_col}"] = get_samples_per_s(results[f"MIN {d_col}"], batch_size)

    logger.warning("This script does not currently assert for perf regressions, and prints info only")
    logger.info(
        f"\nTest: {command}"
        f"\nPerformance statistics over {num_iterations} iterations"
        f"\n{json.dumps(post_processed_results, indent=4)}"
    )

    lower_threshold = (1 - margin) * expected_perf
    upper_threshold = (1 + margin) * expected_perf
    passing = lower_threshold <= post_processed_results["AVG DEVICE KERNEL SAMPLES/S"] <= upper_threshold
    if not passing:
        logger.error(
            f"Average device kernel duration {post_processed_results['AVG DEVICE KERNEL SAMPLES/S']} is outside of expected range ({lower_threshold}, {upper_threshold})"
        )
