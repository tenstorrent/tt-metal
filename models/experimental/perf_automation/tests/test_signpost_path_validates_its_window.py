"""The cheap path must verify as much as the expensive one before its number is believed.

Two paths size the profiling window. The ladder RUNS the model at each rung, so it observes directly
whether capping changed anything and refuses when it did not (_knob_is_inert). The signpost path
derives its number from one full-depth probe, never runs at the capped depth, and hands the number
straight through -- so the route almost every model takes is the route with no verification.

gemma-3-12b-it went down it and produced 96 for a 48-layer model whose builder ignores the cap
entirely. Both faults were invisible: nothing compared the window against the model's declared depth,
and nothing checked that the cap had any effect.

Three checks, in cost order:

  window <= declared depth        free    a window deeper than the model is a unit bug, not caution
  stack length == declared depth  free    confirms the walker tagged the DECODER stack and not the
                                          vision tower (27) or an inner sub-block list (9)
  work signal shrinks at depth    1 probe the cap actually reached the builder

None of them abort the run. A wrong window is recoverable by falling back; a dead run is not.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))

from models.experimental.perf_automation.cc_optimize import run as R  # noqa: E402

SP = "PERF_BLOCK_SIGNPOST:%d"


def _seq(n_layers, passes=1, deep_op_at=None):
    out = []
    for _ in range(passes):
        for i in range(n_layers):
            out += [SP % i, "Matmul", "LayerNorm"]
            if deep_op_at is not None and i == deep_op_at:
                out.append("RareOp")
    return out


# ---------------------------------------------------------------- assert 1: window <= declared


def test_a_window_deeper_than_the_model_is_rejected():
    """96 <= 48 is false. This is the assertion that would have caught the reported bug outright."""
    ok, why = R._validate_signpost_window(96, stack_len=96, declared=48)
    assert ok is False and "48" in why and "96" in why, why


def test_a_window_inside_the_model_passes():
    ok, _why = R._validate_signpost_window(3, stack_len=48, declared=48)
    assert ok is True


def test_a_window_exactly_at_the_declared_depth_passes():
    """The whole model is a legal window -- slow, never wrong."""
    assert R._validate_signpost_window(48, stack_len=48, declared=48)[0] is True


# ---------------------------------------------------------------- assert 2: right stack tagged


def test_the_vision_tower_being_tagged_is_caught():
    """27 tagged blocks against a declared 48: the walker found the wrong stack, and every
    op-to-layer mapping is against it."""
    ok, why = R._validate_signpost_window(3, stack_len=27, declared=48)
    assert ok is False and "27" in why, why


def test_an_inner_sub_block_list_being_tagged_is_caught():
    """The walker really did return a 9-element inner stack once; only a live run caught it."""
    ok, why = R._validate_signpost_window(2, stack_len=9, declared=48)
    assert ok is False and "9" in why, why


# ---------------------------------------------------------------- no declared depth


def test_no_declared_depth_means_no_verdict_not_a_failure():
    """A model with no config cannot be cross-checked. That is missing information, not a fault --
    the window still stands on the signposts alone."""
    assert R._validate_signpost_window(96, stack_len=96, declared=None)[0] is True
    assert R._validate_signpost_window(3, stack_len=48, declared=0)[0] is True


# ---------------------------------------------------------------- the inert knob


def test_an_identical_work_signal_means_the_cap_did_nothing():
    """gemma3: build_pipeline has no depth parameter, so capping to 2 builds the same 48 layers."""
    full = R._work_signal(_seq(48))
    capped = R._work_signal(_seq(48))
    assert R._cap_took_effect(capped, full) is False


def test_a_smaller_work_signal_means_the_cap_applied():
    full = R._work_signal(_seq(48))
    capped = R._work_signal(_seq(2))
    assert R._cap_took_effect(capped, full) is True


def test_a_missing_signal_is_not_evidence_either_way():
    """No measurement must not read as 'the cap worked'. Claiming an unverified window is the exact
    failure this check exists to stop."""
    for capped, full in ((None, R._work_signal(_seq(48))), (R._work_signal(_seq(2)), None), (None, None)):
        assert R._cap_took_effect(capped, full) is None


# ---------------------------------------------------------------- what a failure must NOT do


def test_a_failed_check_never_raises():
    """These run inside a live optimize. A wrong window is recoverable; a crash mid-run is not."""
    for args in ((96, 96, 48), (0, 0, 48), (-1, 9, 48), (3, 0, 0)):
        ok, why = R._validate_signpost_window(*args)
        assert isinstance(ok, bool) and isinstance(why, str)


def test_the_reason_is_specific_enough_to_act_on():
    """'validation failed' sends a reader nowhere. The numbers that disagreed must be in the text."""
    _ok, why = R._validate_signpost_window(96, stack_len=96, declared=48)
    assert "window" in why.lower() or "declared" in why.lower(), why
