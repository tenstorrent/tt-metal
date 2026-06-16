# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Dump device perf report for Devstral-2 single-layer prefill+decode profiling.

Runs the 1-layer prefill+decode profile workload under Tracy (``run_device_perf``) and
writes ``device_perf_*.csv`` plus a partial benchmark JSON via ``prep_device_perf_report``.
No golden perf assertion — this test is for collecting / inspecting reports.

The profile workload matches ``tests/perf/test_profile_single_layer_prefill_decode.py``
(partial Hub weights, prefill 128, decode at 128).
Each measured iteration profiles **both** prefill and decode inside the signpost window.

Run::

    pytest models/experimental/devstral2_123B_instruct/tests/perf/test_device_perf_single_layer_prefill_decode.py \\
        -v -m models_device_performance_bare_metal

After the run, analyze ``generated/profiler/devstral2_123B_instruct_L1_prefill_decode/reports/*/ops_perf_results_*.csv``::

    tt-perf-report <ops_perf_results.csv> --start-signpost start --end-signpost stop
"""

from __future__ import annotations

import pytest
from loguru import logger

from models.perf.device_perf_utils import prep_device_perf_report, run_device_perf

MODEL_NAME = "devstral2_123B_instruct_L1_prefill128_decode1"
SUBDIR = "devstral2_123B_instruct_L1_prefill_decode"
PROFILE_TEST = (
    "models/experimental/devstral2_123B_instruct/tests/perf/test_profile_single_layer_prefill_decode.py"
    "::test_profile_single_layer_prefill_decode"
)
COLS = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]


@pytest.mark.timeout(0)
@pytest.mark.models_device_performance_bare_metal
def test_device_perf_single_layer_prefill_decode():
    """Capture Tracy device perf for 1-layer prefill+decode and dump CSV/JSON report."""
    command = f"pytest {PROFILE_TEST} -v"
    num_iterations = 1
    batch_size = 1

    post_processed_results = run_device_perf(
        command,
        subdir=SUBDIR,
        num_iterations=num_iterations,
        cols=COLS,
        batch_size=batch_size,
        has_signposts=True,
    )

    logger.info(f"Device perf results for {MODEL_NAME}:\n{post_processed_results}")

    prep_device_perf_report(
        model_name=MODEL_NAME,
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results={},
        comments="prefill128_decode1_partial_weights",
    )
