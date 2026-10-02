# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os

from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf

# The profiled ResNet run is its own pytest session, so without an explicit --timeout it inherits the 300 s default
# from pytest.ini rather than the budget the pipeline yaml gives this test (pytest_timeout: 540 s on wh_n150, 240 s on
# bh_p150b_civ2). The inner session gets that budget minus what the outer test still needs once the inner run returns
# (profiler post-processing and the perf report: 23-58 s on the 2026-09-26..28 scheduled runs), so a slow inner run
# fails with its own traceback instead of being killed from outside, and is not cut off at 300 s while the outer test
# still has time left.
POST_PROCESSING_MARGIN_S = 60


def inner_pytest_timeout_s(config, margin_s=POST_PROCESSING_MARGIN_S):
    """Timeout for the profiled pytest session: the outer session's effective pytest-timeout minus margin_s.

    Resolves the outer value the way pytest-timeout does (--timeout, then PYTEST_TIMEOUT, then the ini value).
    Returns None when no outer timeout applies or it is too short to split, leaving the inner session's own default.
    """
    outer = config.getoption("timeout", default=None)
    if outer is None:
        outer = os.environ.get("PYTEST_TIMEOUT")
    if outer is None:
        try:
            outer = config.getini("timeout")
        except ValueError:
            outer = None
    outer = float(outer) if outer else 0.0
    if outer <= margin_s:
        return None
    return int(outer - margin_s)


def run_perf_device(batch_size, test, command, expected_perf, inner_timeout_s=None):
    subdir = "resnet50"
    num_iterations = 1
    margin = 0.03
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]
    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    if inner_timeout_s is not None:
        command = f"{command} --timeout {inner_timeout_s}"

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size, has_signposts=True)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)
    prep_device_perf_report(
        model_name=f"ttnn_resnet50_batch_size{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=test,
    )
