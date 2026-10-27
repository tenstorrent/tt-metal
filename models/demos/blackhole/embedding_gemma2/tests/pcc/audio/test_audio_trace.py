# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Traced audio forward: replay == eager (bit-exact), trace reuse across lengths, determinism.

One trace per (batch, token bucket), all captured up front by `AudioEncoder.capture_all` (which
also prepares every conv-weight layout and compiles every shape before the first capture). Each
case replays the trace of its bucket with its own features and masks; `crop13` / `crop25` reuse the
(1, 32) trace that `short` (16 tokens) also uses, so one trace serves three logical lengths.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.embedding_gemma2.tests.common import log_result, pcc
from models.demos.blackhole.embedding_gemma2.tt.audio.audio_encoder import AudioEncoder

N_REPEAT = 5
TRACE_SHAPES = [(1, 32), (1, 64), (1, 128), (1, 256), (1, 768), (2, 256)]
TRACE_CASES = ("short", "crop13", "crop25", "s2", "tone3", "s10", "s30", "b2_s10_s2")


@pytest.fixture(scope="module")
def traced_encoder(eg2_device, tt_audio_model, eg2_snapshot):
    enc = AudioEncoder(tt_audio_model, eg2_device, eg2_snapshot)
    enc.capture_all(TRACE_SHAPES)
    yield enc
    enc.release_traces()


def _replay(enc, feats, mask):
    return [t.clone() for t in enc.encode(feats, mask, traced=True)]


@pytest.mark.parametrize("case", TRACE_CASES)
def test_trace_matches_eager(case, traced_encoder, audio_reference_for):
    feats, mask, ref = audio_reference_for(case)
    eager = traced_encoder.encode(feats, mask, traced=False)
    replay = _replay(traced_encoder, feats, mask)
    ok = True
    for b, (e, t, hf) in enumerate(zip(eager, replay, ref.soft_tokens)):
        d = (e - t).abs().max().item()
        log_result("audio_trace_vs_eager_max_abs", d, 0.0, case=case, clip=b)
        assert torch.equal(e, t), f"clip {b}: trace replay differs from eager (max {d})"
        ok &= log_result("audio_trace_soft_token_pcc", pcc(hf, t), 0.99, case=case, clip=b)
    assert ok


@pytest.mark.parametrize("case", ["s10", "b2_s10_s2"])
def test_audio_determinism(case, traced_encoder, audio_case):
    feats, mask = audio_case(case)
    base = _replay(traced_encoder, feats, mask)
    for _ in range(N_REPEAT - 1):
        for b, t in enumerate(_replay(traced_encoder, feats, mask)):
            assert torch.equal(base[b], t), f"traced replay is not deterministic (clip {b})"
    for _ in range(2):
        for b, t in enumerate(traced_encoder.encode(feats, mask, traced=False)):
            assert torch.equal(base[b], t), f"eager run differs from replay (clip {b})"
    ttnn.synchronize_device(traced_encoder.device)
    logger.info(f"audio determinism {case}: {N_REPEAT} replays + 2 eager runs bit-identical")
