# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Block 1 across a sequence of requests, not one in isolation.

  * several lengths through one model, unsorted, each against its own reference.
  * a short prompt after a long one, whose cache tail must stay unreachable.
  * position accounting through prefill and decode steps.

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
# Unsorted and spanning a wide P range, so neither warm-for-this-shape nor monotonically
# growing lengths can pass by accident.
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
    """Four requests of three lengths through one model, each against its own reference."""
    worst = None
    for ci in SEQUENCE:
        P, voice, m = _prefill_pcc(gen, w, ci)
        print(f"\n  case {ci} ({voice}, P={P}): last-position PCC {m['pcc']:.6f}  "
              f"worst-sample {m['worst_pct']:.2f}%")
        assert m["pcc"] > PCC_GATE, f"case {ci} (P={P}) degraded to {m['pcc']:.6f} mid-sequence"
        worst = m["pcc"] if worst is None else min(worst, m["pcc"])
    print(f"  worst across the sequence: {worst:.6f}")


def test_short_prompt_after_a_long_one(gen, w):
    """A shorter prompt after a longer one must not reach the previous request's cache tail."""
    long_ci, short_ci = 2, 0          # P=312 then P=200
    P_long, _, _ = _prefill_pcc(gen, w, long_ci)
    P_short, voice, m = _prefill_pcc(gen, w, short_ci)
    assert P_short < P_long, "pick a genuinely shorter second case or this test is vacuous"
    print(f"\n  P={P_long} then P={P_short} ({voice}): PCC {m['pcc']:.6f}")
    assert m["pcc"] > PCC_GATE, (
        f"a P={P_short} prompt after a P={P_long} one scored {m['pcc']:.6f} -- the previous "
        f"request's cache tail is reachable")


def test_decode_position_accounting(gen, w):
    """pos after prefill plus k steps must be P+k on both sides, or later RoPE angles are wrong."""
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
