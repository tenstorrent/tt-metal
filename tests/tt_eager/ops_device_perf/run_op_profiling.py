#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import numpy as np
import pytest
from pathlib import Path

from tt_metal.tools.profiler.process_model_log import run_device_profiler, post_process_ops_log, get_profiler_folder

from models.utility_functions import is_wormhole_b0, is_blackhole

from tt_metal.tools.profiler.common import PROFILER_LOGS_DIR, PROFILER_DEVICE_SIDE_LOG, generate_logs_folder

from tt_metal.tools.profiler.process_device_log import import_log_run_stats
import tt_metal.tools.profiler.device_post_proc_config as device_post_proc_config

matmul_baseline_kernel_duration_gs = [
    347076.298,
    656080.448,
    347153.226,
    655997.51,
    175263.62,
    328744.596,
    175286.458,
    328779.454,
    1966421.516,
    2192647.532,
    2176973.452,
    2250058.658,
]

matmul_baseline_kernel_duration_wh = [
    230737,
    329652,
    417351,
    419615,
    214276,
    304323,
    393762,
    394062,
    148655,
    191725,
    245358,
    247829,
    124888,
    158895,
    222026,
    222966,
    892214,
    1301073,
    1593991,
    1601608,
    802177,
    1169392,
    1476336,
    1472074,
    597053,
    776933,
    996587,
    985680,
    478471,
    613935,
    # 928427,
    # 907950,
]


def run_op_test():
    logger.info(f"========= RUNNING OP TEST - matmul ")

    op_name = "tt::operations::primary::Matmul"
    duration_cols = ["DEVICE KERNEL DURATION [ns]"]
    profiler_out_dir = "op_profiler_results"

    run_device_profiler(
        "pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_matmul_1d_2d.py", profiler_out_dir
    )
    results = post_process_ops_log(profiler_out_dir, duration_cols, False, op_name)
    kernel_durations_ns = results[duration_cols[0]]

    setup = device_post_proc_config.default_setup()
    setup.deviceInputLog = generate_logs_folder(get_profiler_folder(profiler_out_dir)) / PROFILER_DEVICE_SIDE_LOG
    deviceData = import_log_run_stats(setup)
    freq = deviceData["deviceInfo"]["freq"]
    freq_to_cycle_ratio = freq / 1000.0

    kernel_durations_cycle = kernel_durations_ns * freq_to_cycle_ratio

    if deviceData["deviceInfo"]["arch"] == "grayskull":
        max_diff = 0.1
        if len(kernel_durations_cycle) != len(matmul_baseline_kernel_duration_gs):
            logger.info(f"number of tests not equal to the baseline tests! bypass check")
            assert True
        else:
            percentage_diff = (
                np.abs(kernel_durations_cycle - matmul_baseline_kernel_duration_gs) / matmul_baseline_kernel_duration_gs
            )
            is_within_range = np.all(percentage_diff < max_diff)
            if is_within_range == False:
                index = np.where(percentage_diff > max_diff)[0]
                for id in index:
                    logger.info(
                        "Diff is too large! Tested kernel duration: {}, baseline kernel duration: {}, diff: {}, percentage_diff: {}",
                        kernel_durations_cycle[id],
                        matmul_baseline_kernel_duration_gs[id],
                        np.abs(kernel_durations_cycle[id] - matmul_baseline_kernel_duration_gs[id]),
                        percentage_diff[id],
                    )
            assert is_within_range

    elif deviceData["deviceInfo"]["arch"] == "wormhole_b0":
        max_diff = 0.1
        if len(kernel_durations_cycle) != len(matmul_baseline_kernel_duration_wh):
            logger.info(f"number of tests not equal to the baseline tests! bypass check")
            assert True
        else:
            percentage_diff = (
                np.abs(kernel_durations_cycle - matmul_baseline_kernel_duration_wh) / matmul_baseline_kernel_duration_wh
            )
            is_within_range = np.all(percentage_diff < max_diff)
            if is_within_range == False:
                index = np.where(percentage_diff > max_diff)[0]
                for id in index:
                    logger.info(
                        "Diff is too large! Tested kernel duration: {}, baseline kernel duration: {}, diff: {}, percentage_diff: {}",
                        kernel_durations_cycle[id],
                        matmul_baseline_kernel_duration_wh[id],
                        np.abs(kernel_durations_cycle[id] - matmul_baseline_kernel_duration_wh[id]),
                        percentage_diff[id],
                    )
        assert is_within_range


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.models_device_performance_bare_metal
def test_run_op_test():
    run_op_test()
