# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The host-side sampling contract of `generate()`: the seed, CFG, and termination.

Block 2's own CFG behaviour is already covered at reference level by `test_flow_pcc.py`
(`cfg_alpha_one_equals_conditional_only`, `cfg_changes_the_trajectory`). What is untested is the
PIPELINE's contract, which is what a caller actually depends on:

  - `seed=N` makes a request reproducible (that one lives in
    test_request_path_repeatability.py, where determinism belongs);
  - a DIFFERENT seed must actually change the draw, or the seed argument is decorative;
  - `seed=None` must NOT reseed, so a caller can drive the RNG themselves;
  - the [END_AUDIO] frame is excluded from the returned frames, not emitted as audio.

Run:
    pytest -svv models/experimental/voxtral_tts/tests/test_sampling.py
"""

import pytest

torch = pytest.importorskip("torch")
ttnn = pytest.importorskip("ttnn")

from models.experimental.voxtral_tts.reference.voxtral_common_ref import END_AUDIO_ID  # noqa: E402
from models.experimental.voxtral_tts.tests.reference_helpers import fixture_embeds  # noqa: E402
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import (  # noqa: E402
    CFG_ALPHA,
    TtVoxtralPipeline,
    open_device,
)

CASE = 0
N = 8  # a few frames is enough to see the draw change; this file is about plumbing, not accuracy


@pytest.fixture(scope="module")
def pipe():
    d = open_device()
    p = TtVoxtralPipeline(d)
    yield p
    p.close()
    ttnn.close_device(d)


def _gen(pipe, embeds, **kw):
    pipe.backbone.reset()
    frames, _, _ = pipe.generate(embeds, max_frames=N, verbose=False, **kw)
    return frames


@pytest.mark.slow
def test_a_different_seed_changes_the_draw(pipe):
    """Otherwise `seed` is decorative and every request returns the same audio."""
    embeds, _ = fixture_embeds(CASE, pipe.wb)
    a = _gen(pipe, embeds, seed=0)
    b = _gen(pipe, embeds, seed=12345)
    assert a.shape == b.shape, "different seeds changed the frame count, which is fine, but then "\
                               "compare lengths instead"
    assert not torch.equal(a, b), "two different seeds produced identical codes -- seed is ignored"
    print(f"\n  seed 0 vs 12345: {int((a != b).sum())} of {a.numel()} codes differ")


@pytest.mark.slow
def test_seed_none_does_not_reseed(pipe):
    """`seed=None` leaves the global RNG alone, so a caller can drive it. Two consecutive calls
    must then differ -- if they match, generate() is reseeding behind the caller's back."""
    embeds, _ = fixture_embeds(CASE, pipe.wb)
    torch.manual_seed(7)
    a = _gen(pipe, embeds, seed=None)
    b = _gen(pipe, embeds, seed=None)
    assert not torch.equal(a, b), "seed=None still produced identical draws -- generate() reseeded"


@pytest.mark.slow
def test_end_audio_is_not_returned_as_a_frame(pipe):
    """`generate()` documents "[END_AUDIO] excluded". A returned END_AUDIO would be decoded as
    audio by the codec."""
    embeds, _ = fixture_embeds(CASE, pipe.wb)
    frames = _gen(pipe, embeds, seed=0)
    assert not bool((frames[:, 0] == END_AUDIO_ID).any()), \
        "an [END_AUDIO] semantic code reached the returned frames"


@pytest.mark.slow
def test_cfg_alpha_reaches_the_model(pipe):
    """CFG has to be plumbed through the traced frame loop, not just the eager path."""
    embeds, _ = fixture_embeds(CASE, pipe.wb)
    a = _gen(pipe, embeds, seed=0, cfg_alpha=CFG_ALPHA)
    b = _gen(pipe, embeds, seed=0, cfg_alpha=1.0)  # conditional only
    assert not torch.equal(a, b), \
        f"cfg_alpha={CFG_ALPHA} and cfg_alpha=1.0 produced identical codes -- CFG is not plumbed"
    print(f"\n  cfg {CFG_ALPHA} vs 1.0: {int((a != b).sum())} of {a.numel()} codes differ")
