# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Is a request's output independent of what ran before it in the same process?

BUG-11, and until now nothing tested it. STATUS 6.21: case 10's frame count moved 207 -> 220
reproducibly between builds and took an hour to chase, because the pipeline object and its KV
cache are reused across requests -- run the case alone and both builds give 220. `generate()`
reseeds per call, so it is not RNG.

**Determinism is not independence.** Two runs of one build match exactly, which is what made the
artifact look trustworthy. So this file tests both properties separately:

  1. same request twice           -> identical            (determinism)
  2. A alone vs B-then-reset-A    -> identical            (reset restores independence)
  3. A alone vs B-then-A          -> documents the leak   (xfail: this is BUG-11 itself)

Run:
    pytest -svv models/experimental/voxtral_tts/tests/test_request_path_repeatability.py
"""

import pytest

torch = pytest.importorskip("torch")
ttnn = pytest.importorskip("ttnn")

from models.experimental.voxtral_tts.tests.reference_helpers import (  # noqa: E402
    case_ids,
    fixture_embeds,
)
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import (  # noqa: E402
    TtVoxtralPipeline,
    open_device,
)

# BUG-11 shows up in NATURAL stop length (case 10 moved 207 -> 220), so the budget has to be big
# enough for [END_AUDIO] to decide the length. With a cap both runs stop at the cap and compare
# equal for a trivial reason -- an earlier version of this file capped at 10 and xpassed for
# exactly that reason, which is worth knowing: a repeatability test under a cap tests nothing.
MAX_FRAMES = 320
CASE_A, CASE_B = 10, 2


@pytest.fixture(scope="module")
def pipe():
    d = open_device()
    p = TtVoxtralPipeline(d)
    yield p
    p.close()
    ttnn.close_device(d)


def _run(pipe, ci):
    embeds, _ = fixture_embeds(ci, pipe.wb)
    frames, _, _ = pipe.generate(embeds, max_frames=MAX_FRAMES, seed=0, verbose=False)
    return frames


@pytest.mark.slow
def test_same_request_twice_is_identical(pipe):
    """Determinism: the cheap property, and the one that made BUG-11 look like a real divergence."""
    pipe.backbone.reset()
    a = _run(pipe, CASE_A)
    pipe.backbone.reset()
    b = _run(pipe, CASE_A)
    assert a.shape == b.shape, f"frame count moved {a.shape[0]} -> {b.shape[0]} on a repeat"
    assert torch.equal(a, b), "identical request produced different codes"
    print(f"\n  case {CASE_A} twice: {a.shape[0]} frames, codes identical")


@pytest.mark.slow
def test_reset_makes_requests_independent(pipe):
    """`reset()` must erase the previous request. This is the contract a server needs."""
    pipe.backbone.reset()
    alone = _run(pipe, CASE_A)
    pipe.backbone.reset()
    _run(pipe, CASE_B)
    pipe.backbone.reset()
    after = _run(pipe, CASE_A)
    print(f"\n  case {CASE_A} alone: {alone.shape[0]} frames | after case {CASE_B} + reset: "
          f"{after.shape[0]} frames")
    assert alone.shape == after.shape, (
        f"frame count depends on history even across reset(): {alone.shape[0]} vs {after.shape[0]}")
    assert torch.equal(alone, after), "codes depend on history even across reset()"


@pytest.mark.slow
def test_history_does_not_leak_without_reset(pipe):
    """The property a caller would ASSUME: back-to-back generate() calls are independent.

    THIS PASSES, and that is a result rather than a formality. It was written as an xfail for
    BUG-11 -- `generate()` never calls reset, so the second request prefills on top of the first
    one's KV cache -- and it xpassed: case 10 emits 184 frames alone and 184 identical frames after
    case 2 with no intervening reset. Prefill overwrites every position it then attends to, so the
    stale tail is unreachable.

    IT IS NOT PROOF THE BUG IS GONE. STATUS 6.21 observed case 10 at 207 frames inside a 15-case
    sequential run and 220 alone, so the leak it describes may need more history than two requests,
    or a specific predecessor. A 15-case sequential repeatability test is the real check and does
    not exist yet. Note also that 184 != 220: frame counts move with numerics, and 220 was an older
    build -- do not read that difference as this bug.
    """
    pipe.backbone.reset()
    alone = _run(pipe, CASE_A)
    pipe.backbone.reset()
    _run(pipe, CASE_B)
    after = _run(pipe, CASE_A)          # no reset, deliberately
    print(f"\n  case {CASE_A} alone: {alone.shape[0]} frames | after case {CASE_B}, NO reset: "
          f"{after.shape[0]} frames")
    assert torch.equal(alone, after), (
        f"history leaked: {alone.shape[0]} vs {after.shape[0]} frames without an intervening reset")


@pytest.mark.slow
def test_fifteen_case_sequential_run_does_not_change_a_length(pipe):
    """STATUS 6.21's actual experiment, which nothing reproduced until now.

    It recorded case 10 at **207** frames inside a 15-case sequential run and **220** alone, and
    concluded that an utterance's length depends on what ran before it. The two-request tests above
    do not reproduce that, so either it needed more history or it is gone. This runs all 15 cases
    in one process, in order, then re-runs the same case alone and compares.

    Every case is capped at MAX_FRAMES, so a case that would naturally run longer is compared
    cap-to-cap and proves nothing about itself -- the ones that matter are those that stop
    naturally, and the assertion is per case over exactly those.
    """
    lengths = {}
    for ci in case_ids():
        frames = _run(pipe, ci)          # no reset between cases: this is the point
        lengths[ci] = frames.shape[0]
    natural = {ci: n for ci, n in lengths.items() if n < MAX_FRAMES}
    print(f"\n  sequential run, {len(lengths)} cases: " +
          " ".join(f"{ci}:{n}" for ci, n in sorted(lengths.items())))
    print(f"  stopped naturally (comparable): {sorted(natural)}")
    assert natural, ("no case stopped before the cap, so this run cannot detect a length change -- "
                     "raise MAX_FRAMES")

    mismatched = []
    for ci in sorted(natural):
        pipe.backbone.reset()
        alone = _run(pipe, ci).shape[0]
        if alone != natural[ci]:
            mismatched.append((ci, natural[ci], alone))
        print(f"    case {ci}: in-sequence {natural[ci]} vs alone {alone}"
              f"{'   <- DIFFERS' if alone != natural[ci] else ''}")
    assert not mismatched, (
        "frame count depends on what ran before, per BUG-11: " +
        ", ".join(f"case {c} {a} in sequence vs {b} alone" for c, a, b in mismatched))
