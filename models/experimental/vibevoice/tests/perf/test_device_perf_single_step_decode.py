# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Dump device perf report for VibeVoice single eager speech-frame profiling.

Runs the single-frame profile workload under Tracy and writes ``device_perf_*.csv``
plus a partial benchmark JSON via ``prep_device_perf_report``.  No golden perf
assertion — for collecting / inspecting reports.

This script intentionally avoids importing ``ttnn`` so it does not compete for the
UMD device lock with another pytest parent.

Run::

    python models/experimental/vibevoice/tests/perf/test_device_perf_single_step_decode.py

After the run, analyze the CSV::

    tt-perf-report <ops_perf_results.csv> --start-signpost start --end-signpost stop
"""

from __future__ import annotations

import os


def _inner_command() -> str:
    profile_test = (
        "models/experimental/vibevoice/tests/perf/"
        "test_profile_single_step_decode.py::test_profile_single_step_decode"
    )
    return f"pytest --timeout=3600 {profile_test} -sv"


def main() -> int:
    from loguru import logger
    from tracy.common import clear_profiler_runtime_artifacts
    from tracy.process_model_log import get_samples_per_s, post_process_ops_log, run_device_profiler

    from models.perf.device_perf_utils import prep_device_perf_report

    model_name = "vibevoice_eager_speech_frame1"
    subdir = "vibevoice_lm_single_step_decode"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]
    op_support_count = 100000

    command = _inner_command()
    batch_size = 1
    duration_cols = [col + " DURATION [ns]" for col in cols]
    samples_cols = [col + " SAMPLES/S" for col in cols]

    clear_profiler_runtime_artifacts()
    # The workload returns right after the profiled frame, so device profiler data has to
    # be flushed mid-run.  This is the env var ``python -m tracy --dump-device-data-mid-run``
    # sets; the tracy subprocess and the pytest workload inherit it.  Restored afterwards so
    # it does not leak into the rest of the process.
    prev_mid_run_dump = os.environ.get("TT_METAL_PROFILER_MID_RUN_DUMP")
    os.environ["TT_METAL_PROFILER_MID_RUN_DUMP"] = "1"
    try:
        run_device_profiler(
            command,
            subdir,
            check_test_return_code=False,
            device_analysis_types=["device_kernel_duration"],
            op_support_count=op_support_count,
        )
    finally:
        if prev_mid_run_dump is None:
            os.environ.pop("TT_METAL_PROFILER_MID_RUN_DUMP", None)
        else:
            os.environ["TT_METAL_PROFILER_MID_RUN_DUMP"] = prev_mid_run_dump

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
        comments="single_eager_speech_frame1",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
