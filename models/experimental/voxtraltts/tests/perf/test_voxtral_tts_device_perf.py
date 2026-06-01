# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest

from models.common.utility_functions import run_for_blackhole
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf

# TT-only workload (no CPU reference / no PCC) bracketed by start/stop signposts,
# so the device-perf report reflects only the TT pipeline's steady-state forward.
_PERF_TEST = (
    "models/experimental/voxtraltts/tests/perf/test_voxtral_tts_perf_run.py" "::test_voxtral_tts_device_perf_run"
)
# Device profiler op-marker buffer (TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT). Kept high:
# the missing-op assert in tools/tracy/process_ops_logs.py is commented out locally, so a
# too-low value would SILENTLY drop real ops and skew the report. Sizing the buffer above
# the workload's op count is the safe choice; it does not affect report-gen time (that scales
# with total logged ops, reduced instead via fewer generate_steps + no warm-up).
_OP_SUPPORT_COUNT = 50000


@run_for_blackhole("Voxtral TTS device perf is targeted for P150/Blackhole")
@pytest.mark.parametrize(
    "batch_size, model_name, expected_perf",
    [
        (1, "voxtral_tts_e2e", 1.0),
    ],
)
# Generous timeout: this wraps a profiled subprocess whose tracy report post-processing
# (import ops, append device data, generate OPs CSV) is pure-Python and scales with the
# logged op count, which is dominated by per-token prefill over the prompt. The inner
# pytest already runs with --timeout=0; only this outer timeout can kill the run.
@pytest.mark.timeout(14400)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_voxtral_tts(batch_size, model_name, expected_perf):
    subdir = model_name
    num_iterations = 1
    margin = 0.04

    command = f"pytest --timeout=0 {_PERF_TEST} -sv"

    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(
        command,
        subdir,
        num_iterations,
        cols,
        batch_size,
        has_signposts=True,
        op_support_count=_OP_SUPPORT_COUNT,
    )
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)
    prep_device_perf_report(
        model_name=f"ttnn_functional_{model_name}_{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="voxtral_tts_p150_e2e_pcc",
    )
