# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import pytest
from loguru import logger

from tt_metal.tools.profiler.process_model_log import (
    clear_logs,
    post_process_ops_log,
    run_device_profiler,
    get_samples_per_s,
)


@pytest.mark.models_device_performance_virtual_machine
@pytest.mark.parametrize(
    "batch_size, test, expected_perf",
    [
        [8, "HiFi4-activations_BFLOAT16-weights_BFLOAT16-batch_8", 1750],
        [8, "LoFi-activations_BFLOAT16-weights_BFLOAT16-batch_8", 1900],
        [8, "LoFi-activations_BFLOAT8_B-weights_BFLOAT8_B-batch_8", 2140],
        [16, "LoFi-activations_BFLOAT8_B-weights_BFLOAT8_B-batch_16", 2560],
    ],
)
def test_perf_device_virtual_machine(batch_size, test, expected_perf):
    subdir = "resnet50"
    num_iterations = 5
    margin = 0.05
    command = f"pytest models/demos/resnet/tests/test_metal_resnet50.py::test_run_resnet50_inference[{test}]"
    cols = ["DEVICE FW DURATION [ns]", "DEVICE KERNEL DURATION [ns]", "DEVICE BRISC KERNEL DURATION [ns]"]

    clear_logs(subdir)

    results = {}
    for col in cols:
        results[f"AVG {col}"] = 0
        results[f"MIN {col}"] = float("inf")
        results[f"MAX {col}"] = -float("inf")

    for _ in range(num_iterations):
        run_device_profiler(command, "resnet50")
        r = post_process_ops_log(subdir, cols)
        for col in cols:
            results[f"AVG {col}"] += r[col]
            results[f"MIN {col}"] = min(results[f"MIN {col}"], r[col])
            results[f"MAX {col}"] = max(results[f"MAX {col}"], r[col])

    for col in cols:
        results[f"AVG {col}"] = get_samples_per_s(results[f"AVG {col}"] / num_iterations, batch_size)
        results[f"MIN {col}"] = get_samples_per_s(results[f"MIN {col}"], batch_size)
        results[f"MAX {col}"] = get_samples_per_s(results[f"MAX {col}"], batch_size)

    logger.warning("This script does not currently assert for perf regressions, and prints info only")
    logger.info(
        f"\nTest: {command}"
        f"\nPerformance statistics over {num_iterations} iterations"
        f"\n{json.dumps(results, indent=4)}"
    )

    lower_threshold = (1 - margin) * expected_perf
    upper_threshold = (1 + margin) * expected_perf
    passing = lower_threshold <= results["AVG DEVICE KERNEL DURATION [ns]"] <= upper_threshold
    if not passing:
        logger.error(
            f"Average device kernel duration{results['AVG DEVICE KERNEL DURATION [ns]']} is outside of expected range ({lower_threshold}, {upper_threshold})"
        )
