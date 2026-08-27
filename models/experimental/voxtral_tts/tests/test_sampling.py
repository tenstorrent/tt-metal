# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The host-side sampling contract of `generate()`: seed, CFG and termination.

Block 2's own CFG behaviour is covered at reference level in test_flow_ref.py; what is checked here
is the pipeline's contract, which is what a caller depends on.

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
    """A different seed must change the draw."""
    embeds, _ = fixture_embeds(CASE, pipe.wb)
    a = _gen(pipe, embeds, seed=0)
    b = _gen(pipe, embeds, seed=12345)
    assert a.shape == b.shape, "different seeds changed the frame count, which is fine, but then "\
                               "compare lengths instead"
    assert not torch.equal(a, b), "two different seeds produced identical codes -- seed is ignored"
    print(f"\n  seed 0 vs 12345: {int((a != b).sum())} of {a.numel()} codes differ")


@pytest.mark.slow
def test_seed_none_does_not_reseed(pipe):
    """seed=None must leave the global RNG alone, so consecutive calls differ."""
    embeds, _ = fixture_embeds(CASE, pipe.wb)
    torch.manual_seed(7)
    a = _gen(pipe, embeds, seed=None)
    b = _gen(pipe, embeds, seed=None)
    assert not torch.equal(a, b), "seed=None still produced identical draws -- generate() reseeded"


@pytest.mark.slow
def test_end_audio_is_not_returned_as_a_frame(pipe):
    """[END_AUDIO] must not be returned as a frame; the codec would decode it as audio."""
    embeds, _ = fixture_embeds(CASE, pipe.wb)
    frames = _gen(pipe, embeds, seed=0)
    assert not bool((frames[:, 0] == END_AUDIO_ID).any()), \
        "an [END_AUDIO] semantic code reached the returned frames"


@pytest.mark.slow
def test_cfg_alpha_reaches_the_model(pipe):
    """cfg_alpha must reach the traced frame loop, not only the eager path."""
    embeds, _ = fixture_embeds(CASE, pipe.wb)
    a = _gen(pipe, embeds, seed=0, cfg_alpha=CFG_ALPHA)
    b = _gen(pipe, embeds, seed=0, cfg_alpha=1.0)  # conditional only
    assert not torch.equal(a, b), \
        f"cfg_alpha={CFG_ALPHA} and cfg_alpha=1.0 produced identical codes -- CFG is not plumbed"
    print(f"\n  cfg {CFG_ALPHA} vs 1.0: {int((a != b).sum())} of {a.numel()} codes differ")
