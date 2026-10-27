# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Traced vision forward: replay == eager (bit-exact), trace reuse across images, determinism.

One trace per (batch, patch bucket), all captured up front by `VisionEncoder.capture_all`: the
fixed max image shape (2560-row bucket for the processor's 2520 patches) at batch 1 and 2, and the
video-frame bucket (1280 rows for 1260 patches) at batch 8. Every image case replays the same
(1, 2560) trace with its own pixels, positions, RoPE tables, mask and pooling matrix.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.embedding_gemma2.tests.common import log_result, pcc
from models.demos.blackhole.embedding_gemma2.tt.vision.vision_encoder import VisionEncoder

N_REPEAT = 5
TRACE_SHAPES = [(1, 2560), (2, 2560), (8, 1280)]
TRACE_CASES = ("landscape", "portrait", "small", "tiny", "square", "b2_square_small", "video8")


@pytest.fixture(scope="module")
def traced_encoder(eg2_device, tt_vision_model):
    enc = VisionEncoder(tt_vision_model, eg2_device)
    enc.capture_all(TRACE_SHAPES)
    yield enc
    enc.release_traces()


def _replay(enc, pv, pid):
    return [t.clone() for t in enc.encode(pv, pid, traced=True)]


@pytest.mark.parametrize("case", TRACE_CASES)
def test_trace_matches_eager(case, traced_encoder, vision_reference_for):
    pv, pid, ref = vision_reference_for(case)
    eager = traced_encoder.encode(pv, pid, traced=False)
    replay = _replay(traced_encoder, pv, pid)
    ok = True
    for b, (e, t, hf) in enumerate(zip(eager, replay, ref.soft_tokens)):
        d = (e - t).abs().max().item()
        log_result("vision_trace_vs_eager_max_abs", d, 0.0, case=case, image=b)
        assert torch.equal(e, t), f"image {b}: trace replay differs from eager (max {d})"
        ok &= log_result("vision_trace_soft_token_pcc", pcc(hf, t), 0.99, case=case, image=b)
    assert ok


@pytest.mark.parametrize("case", ["b2_square_small", "video8"])
def test_vision_determinism(case, traced_encoder, vision_case):
    pv, pid = vision_case(case)
    base = _replay(traced_encoder, pv, pid)
    for _ in range(N_REPEAT - 1):
        for b, t in enumerate(_replay(traced_encoder, pv, pid)):
            assert torch.equal(base[b], t), f"traced replay is not deterministic (image {b})"
    for _ in range(2):
        for b, t in enumerate(traced_encoder.encode(pv, pid, traced=False)):
            assert torch.equal(base[b], t), f"eager run differs from replay (image {b})"
    ttnn.synchronize_device(traced_encoder.device)
    logger.info(f"vision determinism {case}: {N_REPEAT} replays + 2 eager runs bit-identical")
