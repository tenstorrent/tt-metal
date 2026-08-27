# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Block 1 across a SEQUENCE of requests, not one in isolation.

The codec has this class of test (`test_codec_ttnn_pcc.py`: bucketing, chunked == unchunked,
prepared-weight dedup, bias cache not growing). Blocks 1 and 2 had none -- every Block 1 test
built a fresh model, prefilled once and asserted. A server does not do that.

What can go wrong here, and why each test exists:

  - **Per-length state reused across lengths.** BUG-2 is the codec's version: prepared conv weights
    are the same SHAPE at every length and different VALUES, so cross-length reuse computes PCC
    0.19 while crashing nothing. Block 1's matmuls have no prepared weights, but its prefill pads
    to a tile multiple and its program cache is keyed per shape, so the same failure mode is
    available. Prefilling several lengths through ONE model and checking each against its own
    reference is the test.
  - **A short prompt after a long one.** The KV cache still holds the long prompt's tail. Causal
    attention should never reach it, but "should" is what a test is for.
  - **Repeated trace capture/release.** BUG-1 and BUG-5: a failed capture wedges the card, and
    allocations after a capture can be corrupted. Cycling capture/release and then checking
    prefill is still numerically right covers the quiet half of that.

Run:
    pytest -svv models/experimental/voxtral_tts/tests/test_backbone_request_path.py
"""

import pytest

torch = pytest.importorskip("torch")
ttnn = pytest.importorskip("ttnn")

pytestmark = pytest.mark.slow

from models.experimental.voxtral_tts.reference import voxtral_backbone_ref as bref  # noqa: E402
from models.experimental.voxtral_tts.reference.voxtral_common_ref import N_LAYERS  # noqa: E402
from models.experimental.voxtral_tts.tests.gates import compare_hidden  # noqa: E402
from models.experimental.voxtral_tts.tests.reference_helpers import (  # noqa: E402
    backbone_state,
    fixture_embeds,
)
from models.experimental.voxtral_tts.tt.ttnn_voxtral_gpt import TtVoxtralGPT  # noqa: E402
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import open_device  # noqa: E402

PCC_GATE = 0.999
# Deliberately unsorted and spanning a wide P range, so "worked because it was warm for this shape"
# and "worked because lengths only ever grew" both fail.
SEQUENCE = (3, 0, 2, 0)


@pytest.fixture(scope="module")
def dev():
    d = open_device()
    yield d
    ttnn.close_device(d)


@pytest.fixture(scope="module")
def w():
    return backbone_state()


@pytest.fixture(scope="module")
def gen(dev, w):
    return TtVoxtralGPT(dev, n_layers=N_LAYERS, state=w, max_seq_len=1024)


def _prefill_pcc(gen, w, ci):
    embeds, case = fixture_embeds(ci, w)
    exp = bref.reference_forward(embeds, w, n_layers=N_LAYERS)
    gen.reset()
    got = gen.prefill(embeds)
    m = compare_hidden(got[:, -1:], exp[:, -1:])
    return embeds.shape[1], case["voice"], m


def test_prefill_stays_correct_across_lengths_in_one_session(gen, w):
    """Four requests of three different lengths through ONE model, each vs its own reference."""
    worst = None
    for ci in SEQUENCE:
        P, voice, m = _prefill_pcc(gen, w, ci)
        print(f"\n  case {ci} ({voice}, P={P}): last-position PCC {m['pcc']:.6f}  "
              f"worst-sample {m['worst_pct']:.2f}%")
        assert m["pcc"] > PCC_GATE, f"case {ci} (P={P}) degraded to {m['pcc']:.6f} mid-sequence"
        worst = m["pcc"] if worst is None else min(worst, m["pcc"])
    print(f"  worst across the sequence: {worst:.6f}")


def test_short_prompt_after_a_long_one(gen, w):
    """The long prompt's KV tail must be unreachable, not merely unused-by-luck."""
    long_ci, short_ci = 2, 0          # P=312 then P=200
    P_long, _, _ = _prefill_pcc(gen, w, long_ci)
    P_short, voice, m = _prefill_pcc(gen, w, short_ci)
    assert P_short < P_long, "pick a genuinely shorter second case or this test is vacuous"
    print(f"\n  P={P_long} then P={P_short} ({voice}): PCC {m['pcc']:.6f}")
    assert m["pcc"] > PCC_GATE, (
        f"a P={P_short} prompt after a P={P_long} one scored {m['pcc']:.6f} -- the previous "
        f"request's cache tail is reachable")


def test_decode_position_accounting(gen, w):
    """`pos` after prefill+k steps must be P+k on both sides, or every later RoPE angle is wrong."""
    frames = torch.load(__import__("os").path.join(
        __import__("os").path.dirname(__import__("os").path.dirname(
            __import__("os").path.abspath(__file__))), "tests", "real_frames_fixture.pt")).long()
    embeds, _ = fixture_embeds(0, w)
    P = embeds.shape[1]
    inc = bref.IncrementalBackbone(w, n_layers=N_LAYERS)
    inc.prefill(embeds)
    gen.reset()
    gen.prefill(embeds, last_only=True)
    assert gen.pos == inc.pos == P, f"after prefill: device {gen.pos}, reference {inc.pos}, P={P}"
    k = 4
    for t in range(k):
        emb = bref.embed_frame(w, frames[t])
        inc.step(emb)
        gen.step(emb)
    assert gen.pos == inc.pos == P + k, f"after {k} steps: device {gen.pos}, reference {inc.pos}"
    print(f"\n  pos tracked exactly through prefill(P={P}) + {k} steps")
