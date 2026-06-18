# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device-perf wrapper for the single text-decoder-layer prefill+decode workload.

Runs ``test_profile_single_layer_prefill_decode`` in its own subprocess under the device
profiler, then post-processes the op log into a perf report. Use this to track one layer's
prefill+decode device cost in isolation; the workload file is the profile target, this is
the report generator (mirrors test_voxtral_tts_stage_device_perf.py).
"""

import pytest

from models.common.utility_functions import run_for_blackhole
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf

_WORKLOAD = (
    "models/experimental/voxtraltts/tests/perf/test_profile_single_layer_prefill_decode.py"
    "::test_profile_single_layer_prefill_decode"
)
# Single-layer log is small, but kept high for the same reason as the stage wrapper: the
# missing-op assert in tools/tracy/process_ops_logs.py is commented out locally, so a
# too-low value would silently drop real ops.
_OP_SUPPORT_COUNT = 50000


@run_for_blackhole("Voxtral TTS device perf is targeted for P150/Blackhole")
@pytest.mark.parametrize("batch_size, model_name", [(1, "voxtral_tts")])
@pytest.mark.timeout(3600)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_voxtral_single_layer(batch_size, model_name):
    subdir = f"{model_name}_single_layer"
    num_iterations = 1
    margin = 0.04

    command = f"pytest --timeout=0 {_WORKLOAD} -sv"

    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    # Baseline captured on P150 (single text decoder layer, prefill 128 + decode 1).
    expected_perf_cols = {inference_time_key: 443.0}

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
        model_name=f"ttnn_functional_{model_name}_single_layer_{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=f"voxtral_tts_p150_single_layer_prefill_decode_{batch_size}",
    )
