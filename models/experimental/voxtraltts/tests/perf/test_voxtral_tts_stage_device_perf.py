# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-stage device-perf wrappers for Voxtral TTS (text prefill / decode, acoustic, audio decode).

Each parametrized case profiles ONE stage in its own subprocess (via the matching
``test_voxtral_tts_stage_perf_run[<stage>]`` workload), so the profiler log stays small
and report generation is fast — unlike the full e2e perf wrapper. Use these to see which
stage dominates device time and to iterate on a single stage during optimization.

The complete-pipeline perf test (``test_voxtral_tts_device_perf.py``) is unchanged and
remains the source for end-to-end numbers.
"""

import pytest

from models.common.utility_functions import run_for_blackhole
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf

_WORKLOAD = (
    "models/experimental/voxtraltts/tests/perf/test_voxtral_tts_stage_perf_run.py" "::test_voxtral_tts_stage_perf_run"
)
# See test_voxtral_tts_device_perf.py: kept high because the missing-op assert in
# tools/tracy/process_ops_logs.py is commented out locally; a too-low value would silently
# drop real ops. Per-stage logs are small, so this is comfortably oversized.
_OP_SUPPORT_COUNT = 50000

# expected_perf is a placeholder (1.0) per stage — set real targets once a clean baseline
# AVG DEVICE KERNEL SAMPLES/S is captured for each stage.
_STAGE_CASES = [
    ("text_prefill", 1.0),
    ("text_decode", 1.0),
    ("acoustic_forward", 1.0),
    ("audio_decode", 1.0),
]


@run_for_blackhole("Voxtral TTS device perf is targeted for P150/Blackhole")
@pytest.mark.parametrize("stage, expected_perf", _STAGE_CASES)
@pytest.mark.parametrize("batch_size, model_name", [(1, "voxtral_tts")])
@pytest.mark.timeout(7200)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_voxtral_tts_stage(batch_size, model_name, stage, expected_perf):
    subdir = f"{model_name}_{stage}"
    num_iterations = 1
    margin = 0.04

    # Select the stage via -k (the workload has two parametrize axes, so the node id is
    # "[device_params0-<stage>]"; -k matches the stage substring regardless of axis order).
    command = f'pytest --timeout=0 {_WORKLOAD} -sv -k "{stage}"'

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
        model_name=f"ttnn_functional_{model_name}_{stage}_{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=f"voxtral_tts_p150_stage_{stage}",
    )
