"""Signposts are ground truth. A histogram of op counts does not get to veto them.

_signposts_agree cross-checked the signposts against _op_block_count -- "the most common repeat count
among the ops" -- and discarded them when the two differed by more than 20%. That check is backwards
twice over:

  * _op_block_count counts op EXECUTIONS, so it reports layers x passes. A perf test that prefills
    and then decodes over 48 layers reports 96. Comparing it against the number of DISTINCT blocks
    guarantees disagreement on every two-pass model, which is most of them.
  * it is a statistical guess auditing a direct measurement. The signposts are attached to the real
    stack by identity (_tag_stack sets an attribute on each block); the histogram infers structure
    from how often symbols repeat. Letting the weaker estimate overrule the stronger one inverts
    which number to believe.

So the gate is now "are there signposts at all", and _op_block_count is what runs when there are
none -- a fallback, never an auditor.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))

from models.experimental.perf_automation.cc_optimize import run as R  # noqa: E402

SP = "PERF_BLOCK_SIGNPOST:%d"


def _pass(n, extra=None):
    seq = []
    for i in range(n):
        seq += [SP % i, "Matmul", "LayerNorm"]
        for op in (extra or {}).get(i, ()):
            seq.append(op)
    return seq


# ---------------------------------------------------------------- the false rejection


def test_two_passes_no_longer_discard_the_signposts():
    """48 distinct blocks vs 96 executions was a 0.5 ratio and threw the signposts away."""
    seq = _pass(48) + _pass(48)
    assert R._signposts_usable(seq) is True
    assert R._block_start_positions(seq)[1] == "signposts"


def test_ten_passes_are_still_fine():
    """The old ratio got worse with every extra pass -- 48/480 = 0.1. Nothing about pass count says
    anything about whether the signposts are attached correctly."""
    seq = sum((_pass(48) for _ in range(10)), [])
    assert R._signposts_usable(seq) is True


def test_the_block_index_stays_inside_the_stack_however_many_passes():
    seq = _pass(48) + _pass(48) + _pass(48)
    per_stack, source = R._first_block_map(seq)
    fb = per_stack.get("stack0", per_stack)
    assert source == "signposts" and max(fb.values()) <= 47


# ---------------------------------------------------------------- the histogram is a fallback now


def test_no_signposts_still_infers_from_op_repetition():
    """Models whose blocks cannot be tagged keep the only estimate there is."""
    seq = []
    for i in range(6):
        seq += ["Anchor", "Matmul"] + (["Deep"] if i == 5 else [])
    assert R._signposts_usable(seq) is False
    starts, source = R._block_start_positions(seq)
    assert source == "inferred" and len(starts) == 6


def test_a_single_signpost_is_not_a_stack():
    """One tagged block cannot delimit anything -- there is no second boundary. Fall back."""
    seq = [SP % 0, "Matmul", "Matmul", "Matmul"]
    assert R._signposts_usable(seq) is False


def test_an_empty_or_signpost_free_sequence_is_not_usable():
    for seq in ([], None, ["Matmul", "LayerNorm"]):
        assert R._signposts_usable(seq) is False


# ---------------------------------------------------------------- what the gate must NOT do


def test_a_wildly_wrong_histogram_cannot_veto_the_signposts():
    """The point of the change. An op firing 500 times inside a 4-block stack -- a loop inside a
    block -- used to look like '500 blocks' and discredit four perfectly good signposts."""
    seq = _pass(4)
    seq += ["HotOp"] * 500
    assert R._signposts_usable(seq) is True
    assert R._block_start_positions(seq)[1] == "signposts"


def test_the_agree_ratio_env_var_no_longer_changes_anything(monkeypatch):
    """The knob tuned a check that no longer exists. Leaving it live would let an old .env silently
    reinstate the rejection."""
    seq = _pass(48) + _pass(48)
    for ratio in ("0.99", "0.0", "garbage"):
        monkeypatch.setenv("PERF_MCP_SIGNPOST_AGREE_RATIO", ratio)
        assert R._signposts_usable(seq) is True


def test_signposts_win_even_when_the_histogram_would_agree():
    """Not a regression test for the bug -- a guard that the signpost branch is taken on its own
    merits, not because the two happen to match."""
    seq = _pass(8)
    assert R._block_start_positions(seq)[1] == "signposts"
    per_stack, _ = R._first_block_map(seq)
    fb = per_stack.get("stack0", per_stack)
    assert fb["Matmul"] == 0
