# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single canonical audio tokenizer test for kernel optimization.

Runs one codes → waveform decode at HF ``params.json`` dimensions (default T=1600)
with a short warmup and Tracy ``start``/``stop`` around the timed pass. Use this
instead of the many parametrized PCC / component tests when iterating on perf.

Quick run:

    pytest models/experimental/voxtraltts/tests/test_audio_tokenizer_opt.py -sv --timeout=0

With Tracy capture:

    python -m tracy -p -r -v -o generated/profiler/voxtral_audio_tokenizer \\
      -m "pytest models/experimental/voxtraltts/tests/test_audio_tokenizer_opt.py -sv --timeout=0"

Env: ``VOXTRAL_PERF_DECODE_T`` (default 1600), ``VOXTRAL_PERF_WARMUP_T`` (default 64).
"""
from __future__ import annotations

import pytest

from models.experimental.voxtraltts.tests.audio_tokenizer_workload import (
    run_voxtral_audio_tokenizer_decode_benchmark,
)


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_voxtral_audio_tokenizer_opt(voxtral_runtime_mesh_device, reset_seeds):
    """Profile / optimize full audio tokenizer decode (codes → waveform) at T=1600."""
    result = run_voxtral_audio_tokenizer_decode_benchmark(
        voxtral_runtime_mesh_device,
        log_stages=True,
    )
    assert result.timed_frames >= 1
    assert result.wav_shape[0] == 1
