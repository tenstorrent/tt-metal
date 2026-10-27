# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Traced latency of the fp32-activation variant (AudioModel act_dtype=float32), the measured
precision lever in doc/functional_audio/README.md. Not the default path.

A separate test module, so its device (module scope) holds only this model and its traces.
"""

import pytest

import ttnn
from models.demos.blackhole.embedding_gemma2.tests.pcc.audio.test_audio_perf import PERF_CASES, time_traced
from models.demos.blackhole.embedding_gemma2.tt.audio.audio_encoder import AudioEncoder

FP32_CASES = PERF_CASES[:2]  # 1 x 10 s, 1 x 30 s


@pytest.fixture(scope="module")
def perf_encoder_fp32(eg2_device, audio_state_dict, audio_config, eg2_cache_dir, eg2_snapshot):
    from models.demos.blackhole.embedding_gemma2.tt.audio.audio_model import AudioModel

    model = AudioModel.from_state_dict(
        audio_state_dict, audio_config, eg2_device, cache_dir=eg2_cache_dir, act_dtype=ttnn.float32
    )
    enc = AudioEncoder(model, eg2_device, eg2_snapshot)
    enc.capture_all([(b, s) for b, s, _ in FP32_CASES])
    yield enc
    enc.release_traces()


@pytest.mark.parametrize("batch,bucket,cases", FP32_CASES, ids=[f"b{b}_s{s}" for b, s, _ in FP32_CASES])
def test_audio_perf_traced_fp32_act(batch, bucket, cases, perf_encoder_fp32, eg2_snapshot):
    time_traced(perf_encoder_fp32, batch, bucket, cases, eg2_snapshot, "fp32")
