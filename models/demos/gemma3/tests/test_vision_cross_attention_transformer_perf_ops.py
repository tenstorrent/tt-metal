# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import json

import pandas as pd
from loguru import logger
from tracy.process_model_log import get_latest_ops_log_filename

import pytest
from models.demos.llama3_70b_galaxy.tests.test_prefill_device_perf import (
    average_per_instance_dict,
    build_duration_dict,
    build_duration_per_instance_dict,
    merge_device_rows,
)
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.perf.device_perf_utils import run_device_perf


def compare_with_target(kernel_duration_per_instance_averaged_dict, perf_targets, margins):
    passing = True
    for op_index, op_code_with_id in enumerate(kernel_duration_per_instance_averaged_dict.keys()):
        if op_code_with_id in perf_targets:
            avg_kernel_duration = kernel_duration_per_instance_averaged_dict[op_code_with_id]

            # Verify kernel duration is within tolerance
            upper_limit = perf_targets[op_code_with_id] + margins[op_code_with_id] * perf_targets[op_code_with_id]
            lower_limit = perf_targets[op_code_with_id] - margins[op_code_with_id] * perf_targets[op_code_with_id]

            if avg_kernel_duration > upper_limit:
                passing = False
                logger.info(
                    f"{op_code_with_id} kernel: {avg_kernel_duration} ns is larger than target "
                    f"({perf_targets[op_code_with_id]}) ns, difference: "
                    f"{abs(avg_kernel_duration - upper_limit)} ns, margin: "
                    f"{margins[op_code_with_id]}, "
                    f"relative margin to pass would be: "
                    f"{abs(perf_targets[op_code_with_id] - avg_kernel_duration) / perf_targets[op_code_with_id]}"
                )
            elif avg_kernel_duration < lower_limit:
                passing = False
                logger.info(
                    f"{op_code_with_id} kernel: {avg_kernel_duration} ns is smaller than target "
                    f"({perf_targets[op_code_with_id]}) ns, difference: "
                    f"{abs(lower_limit - avg_kernel_duration)} ns, margin: "
                    f"{margins[op_code_with_id]}, "
                    f"relative margin to pass would be: "
                    f"{abs(perf_targets[op_code_with_id] - avg_kernel_duration) / perf_targets[op_code_with_id]}"
                )
        else:
            passing = False
            logger.info(f"Warning: {op_code_with_id} not found in perf_targets")

    assert passing, "One or more ops did not meet performance targets. Check logs for details."


@pytest.mark.models_device_performance_bare_metal
def test_op_to_op_perf_gemma_vision():
    profiler = BenchmarkProfiler()
    batch_size = 1
    subdir = f"ttnn_gemma_cross_attention_perf"
    num_iterations = 1
    command = f"pytest models/demos/gemma3/tests/test_vision_cross_attention_transformer.py::test_gemma_vision"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]
    profiler.start("run")
    profiler.start("PROFILING OP TO OP")
    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size, has_signposts=False)
    profiler.end("PROFILING OP TO OP")
    profiler.end("run")

    filename = get_latest_ops_log_filename(subdir)
    df = pd.read_csv(filename)
    df = df[df["OP TYPE"].isin(["tt_dnn_device"])]
    df = merge_device_rows(df)

    ops_raw_dict = df[["OP CODE", "DEVICE KERNEL DURATION [ns]"]].to_dict(orient="records")
    kernel_duration_dict = build_duration_dict(ops_raw_dict, "DEVICE KERNEL DURATION [ns]")
    kernel_duration_per_instance_dict = build_duration_per_instance_dict(kernel_duration_dict, 1)

    # Average over all iterations of each op instance (in this specific case it is the same)
    kernel_duration_per_instance_averaged_dict = average_per_instance_dict(kernel_duration_per_instance_dict)

    expected_perf_cols = {}
    margins = {}
    with open(
        f"models/demos/gemma3/tests/perf_targets/targets_test_perf_vision_cross_attention_op_to_op.json", "r"
    ) as f:
        expected_perf_cols = json.load(f)
    with open(
        f"models/demos/gemma3/tests/perf_targets/targets_margins_test_perf_vision_cross_attention_op_to_op.json", "r"
    ) as f:
        margins = json.load(f)
    compare_with_target(kernel_duration_per_instance_averaged_dict, expected_perf_cols, margins)
