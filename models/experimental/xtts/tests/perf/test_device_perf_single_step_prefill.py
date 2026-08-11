# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Dump a device perf report for the XTTS-v2 single-step GPT PREFILL workload.

Spawns ``test_profile_single_step_prefill`` under Tracy and writes
``device_perf_*.csv`` plus a partial benchmark JSON via ``prep_device_perf_report``.
No golden perf assertion — this is for collecting and inspecting reports, not gating
(the gate lives in ``test_e2e_perf.py``).

This script intentionally avoids importing ``ttnn`` so it does not compete for the UMD
device lock with the pytest child it spawns.

Run::

    python models/experimental/xtts/tests/perf/test_device_perf_single_step_prefill.py

Then analyze the CSV::

    tt-perf-report <ops_perf_results.csv> --start-signpost start --end-signpost stop

Note the report's "Total %" column ranks HOST wall time, not device work — read the
Device Time column (or the stacked report) when comparing op costs.
"""

from __future__ import annotations

import os

TEXT_LEN = 96  # keep in sync with the inner workload
MAX_NEW_TOKENS = 240


def _inner_command() -> str:
    profile_test = (
        "models/experimental/xtts/tests/perf/test_profile_single_step_prefill.py::test_profile_single_step_prefill"
    )
    # Match the inner test's @pytest.mark.timeout(1800).
    return f"pytest --timeout=1800 {profile_test} -sv"


def main() -> int:
    from loguru import logger
    from tracy.common import clear_profiler_runtime_artifacts
    from tracy.process_model_log import get_samples_per_s, post_process_ops_log, run_device_profiler

    from models.perf.device_perf_utils import prep_device_perf_report

    num_layers = os.environ.get("XTTS_PERF_NUM_LAYERS", "30")
    model_name = f"xtts_gpt_prefill_L{num_layers}_text{TEXT_LEN}"
    subdir = "xtts_gpt_single_step_prefill"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]
    # Headroom for weight load / warmup before ReadDeviceProfiler clears the buffer.
    op_support_count = 50000

    batch_size = 1
    duration_cols = [col + " DURATION [ns]" for col in cols]
    samples_cols = [col + " SAMPLES/S" for col in cols]

    clear_profiler_runtime_artifacts()
    run_device_profiler(
        _inner_command(),
        subdir,
        check_test_return_code=False,
        device_analysis_types=["device_kernel_duration"],
        op_support_count=op_support_count,
    )

    raw = post_process_ops_log(subdir, duration_cols, has_signposts=True)
    post_processed_results = {}
    for s_col, d_col in zip(samples_cols, duration_cols):
        ns = raw[d_col]
        post_processed_results[f"AVG {s_col}"] = get_samples_per_s(ns, batch_size)
        post_processed_results[f"MIN {s_col}"] = get_samples_per_s(ns, batch_size)
        post_processed_results[f"MAX {s_col}"] = get_samples_per_s(ns, batch_size)
        post_processed_results[f"AVG {d_col}"] = ns
        post_processed_results[f"MIN {d_col}"] = ns
        post_processed_results[f"MAX {d_col}"] = ns

    logger.info(f"Device perf results for {model_name}:\n{post_processed_results}")

    prep_device_perf_report(
        model_name=model_name,
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results={},
        comments=f"single_prefill_text{TEXT_LEN}_maxtok{MAX_NEW_TOKENS}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
