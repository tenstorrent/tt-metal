# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Voxtral TTS — device performance test for P150 (Blackhole).

Measures raw device kernel execution time using the device profiler. Runs the
complete E2E PCC test (``test_ttnn_voxtral_tts_e2e_pcc``) with profiler enabled
and reports device throughput (samples/s).

Usage::

    pytest models/experimental/voxtraltts/tests/perf/test_voxtral_tts_device_perf.py \\
        -v -m models_device_performance_bare_metal
"""

import pytest

from models.common.utility_functions import run_for_blackhole
from models.perf.device_perf_utils import prep_device_perf_report, run_device_perf

_PCC_TEST = "models/experimental/voxtraltts/tests/pcc/test_ttnn_voxtral_tts_e2e.py"
_PCC_TARGET = "test_ttnn_voxtral_tts_e2e_pcc"
_DEVICE_COLS = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]
_INFERENCE_TIME_KEY = "AVG DEVICE KERNEL SAMPLES/S"
_BATCH_SIZE = 1

# Full pipeline PCC (text + acoustic loop + waveform) exceeds Tracy's default buffer.
_OP_SUPPORT_COUNT = 50000


@run_for_blackhole("Voxtral TTS device perf is targeted for P150/Blackhole")
@pytest.mark.timeout(3600)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_voxtral_tts():
    """
    Device performance for the complete Voxtral TTS E2E PCC path.

    Inner pytest uses ``--timeout=0`` so the PCC test's own timeout governs the workload.
    """
    subdir = "ttnn_voxtral_tts_e2e_pcc"
    num_iterations = 1
    command = f"pytest --timeout=0 {_PCC_TEST}::{_PCC_TARGET} -sv"

    post_processed_results = run_device_perf(
        command,
        subdir,
        num_iterations,
        _DEVICE_COLS,
        _BATCH_SIZE,
        op_support_count=_OP_SUPPORT_COUNT,
    )
    actual_perf = post_processed_results.get(_INFERENCE_TIME_KEY, 0)
    kernel_ns = post_processed_results.get("AVG DEVICE KERNEL DURATION [ns]", 0)
    print(f"\n{'=' * 60}")
    print("Voxtral TTS Device Performance (complete E2E PCC)")
    print(f"{'=' * 60}")
    print(f"  Measured: {actual_perf:.2f} samples/s ({kernel_ns / 1e6:.2f} ms kernel)")
    print(f"{'=' * 60}\n")

    prep_device_perf_report(
        model_name=f"ttnn_voxtral_tts_batch{_BATCH_SIZE}_e2e_pcc",
        batch_size=_BATCH_SIZE,
        post_processed_results=post_processed_results,
        expected_results={},
        comments="voxtral_tts_p150_e2e_pcc",
    )
