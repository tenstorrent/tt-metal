# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Request independence: does an utterance depend on what ran before it in the same process?

The pipeline object and its KV cache are reused across requests, and `generate()` reseeds per call,
so determinism and independence are separate properties and are tested separately.

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

# Big enough for [END_AUDIO] to decide the length: under a cap both arms stop at the cap and
# compare equal for a trivial reason.
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
    """The same request twice must give the same codes."""
    pipe.backbone.reset()
    a = _run(pipe, CASE_A)
    pipe.backbone.reset()
    b = _run(pipe, CASE_A)
    assert a.shape == b.shape, f"frame count moved {a.shape[0]} -> {b.shape[0]} on a repeat"
    assert torch.equal(a, b), "identical request produced different codes"
    print(f"\n  case {CASE_A} twice: {a.shape[0]} frames, codes identical")


@pytest.mark.slow
def test_reset_makes_requests_independent(pipe):
    """reset() must erase the previous request. This is the contract a server needs."""
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
    """Back-to-back requests must be independent without an intervening reset.

    Prefill overwrites every position it then attends to, so the previous request's cache tail is
    unreachable."""
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
    """All 15 cases in order, then each naturally-terminating case alone: lengths must match.

    Cases that hit the cap are compared cap-to-cap and prove nothing, so only the rest are asserted.
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
        "frame count depends on what ran before: " +
        ", ".join(f"case {c} {a} in sequence vs {b} alone" for c, a, b in mismatched))
