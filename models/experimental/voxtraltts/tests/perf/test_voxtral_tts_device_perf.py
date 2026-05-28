# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest

from models.common.utility_functions import run_for_blackhole
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf

_PERF_TEST = "models/experimental/voxtraltts/tests/perf/test_voxtral_tts_perf_inference.py"
# Model load dispatches ~68k device ops; use headroom above that so load-time
# profiler overflow is minimized before ReadDeviceProfiler() clears the buffer.
_OP_SUPPORT_COUNT = 75000


@run_for_blackhole("Voxtral TTS device perf is targeted for P150/Blackhole")
@pytest.mark.parametrize(
    "batch_size, model_name, expected_perf",
    [
        (1, "voxtral_tts_e2e", 0.43),
    ],
)
@pytest.mark.timeout(3600)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_voxtral_tts(batch_size, model_name, expected_perf):
    subdir = ""
    num_iterations = 1
    margin = 0.04

    # Inner run is under Tracy (-p); avoid -sv so model load does not flood the terminal.
    command = f"pytest --timeout=0 {_PERF_TEST} -q"

    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(
        command,
        subdir,
        num_iterations,
        cols,
        batch_size,
        device_analysis_types=[],
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
