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


@pytest.mark.models_device_performance_virtual_machine
@pytest.mark.parametrize(
    "batch_size, test, expected_perf",
    [
        [8, "HiFi4-activations_BFLOAT16-weights_BFLOAT16-batch_8", 1550],
        [8, "LoFi-activations_BFLOAT16-weights_BFLOAT16-batch_8", 1670],
        [8, "LoFi-activations_BFLOAT8_B-weights_BFLOAT8_B-batch_8", 1820],
    ],
)
def test_perf_device_virtual_machine(batch_size, test, expected_perf):
    subdir = "resnet50"
    num_iterations = 5
    margin = 0.05
    command = f"pytest models/demos/resnet/tests/test_metal_resnet50.py::test_run_resnet50_inference[{test}]"
    cols = ["DEVICE FW DURATION [ns]", "DEVICE KERNEL DURATION [ns]", "DEVICE BRISC KERNEL DURATION [ns]"]

    clear_profiler_runtime_artifacts()

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

    logger.info(
        f"\nTest: {command}"
        f"\nPerformance statistics over {num_iterations} iterations"
        f"\n{json.dumps(results, indent=4)}"
    )

    # lower_threshold = (1 - margin) * expected_perf
    # upper_threshold = (1 + margin) * expected_perf
    # assert lower_threshold <= results["AVG DEVICE KERNEL DURATION [ns]"] <= upper_threshold
