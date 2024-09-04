# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import pytest

from models.utility_functions import run_for_wormhole_b0
from tt_metal.tools.profiler.process_model_log import post_process_ops_log, run_device_profiler

MATMUL_VARIANTS = ["matmul1d_regular", "matmul1d_transposed", "matmul2d", "matmul_no_mcast"]


# for all matmul variants that can enable stagger, profile op with stagger enabled and disabled,
# and make sure the execution time of the op is longer with stagger enabled
@run_for_wormhole_b0()
@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "matmul_variant",
    MATMUL_VARIANTS,
)
def test_matmul_stagger(matmul_variant):
    duration_column = "DEVICE FW DURATION [ns]"
    output_logs_subdir = "test_matmul"

    command = f"pytest tests/device_perf_tests/matmul_stagger/test_run_8x7_matmul.py -k {matmul_variant}"

    os.environ["TT_ENABLE_MATMUL_STAGGER"] = "1"
    output_logs_subdir_with_stagger = output_logs_subdir + "_with_stagger"
    run_device_profiler(command, output_logs_subdir_with_stagger)
    results_with_stagger = post_process_ops_log(output_logs_subdir_with_stagger, [duration_column])

    del os.environ["TT_ENABLE_MATMUL_STAGGER"]
    run_device_profiler(command, output_logs_subdir)
    results_without_stagger = post_process_ops_log(output_logs_subdir, [duration_column])

    assert (
        results_without_stagger[duration_column] < results_with_stagger[duration_column]
    ), f"There should be a visible perf drop when running {matmul_variant} with stagger"
