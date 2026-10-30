# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The printed op sequence is cut to a budget of ops, and the structure markers survive the cut.

A head slice keeps the first ops and silently drops every marker emitted after them. Any pipeline
whose later stages run past the budget then reports full_blocks=0 -- the walk found the stacks and
the blocks announced themselves, but the sequence handed to the reader carried none of it, so the
depth knob is declared unusable and the profile runs the model whole. Stage and block names here
are this file's own; the rule is about the cut, not about any model.
"""

from __future__ import annotations

from pathlib import Path

from agent import marker_vocab as V
from cc_optimize import _op_sig_probe as P
from cc_optimize.run import _blocks_ran, _is_control, _work_signal

_PA = Path(__file__).resolve().parents[1]


def _run(ops_before: int, cap: int = 8):
    """A sequence whose block markers all fall past `cap`."""
    seq = ["ttnn.op(%d)" % i for i in range(ops_before)]
    seq += ["%sstack3:%d" % (P._SIGNPOST_PREFIX, i) for i in range(4)]
    return P._printable_seq(seq, cap=cap)


def test_markers_past_the_cut_are_kept():
    out = _run(ops_before=50)
    assert _blocks_ran(out) == 4, "every block marker reaches the reader"


def test_the_old_head_slice_loses_them():
    """What the fix is for: the slice this replaces reported a model with no blocks at all."""
    seq = ["ttnn.op(%d)" % i for i in range(50)]
    seq += ["%sstack3:%d" % (P._SIGNPOST_PREFIX, i) for i in range(4)]
    assert _blocks_ran(seq[:8]) == 0


def test_the_op_budget_is_unchanged():
    """The head is the same slice it was, so the work signal is the same number."""
    seq = ["ttnn.op(%d)" % i for i in range(50)]
    seq += ["%sstack3:%d" % (P._SIGNPOST_PREFIX, i) for i in range(4)]
    assert _work_signal(P._printable_seq(seq, cap=8)) == _work_signal(seq[:8])


def test_order_is_preserved_because_it_only_deletes():
    seq = ["a", "%sencode" % P._STAGE_PREFIX, "b", "c", "%sstack1:0" % P._SIGNPOST_PREFIX, "d"]
    out = P._printable_seq(seq, cap=3)
    assert out == [t for t in seq if t in out], "surviving tokens keep the order they ran in"
    assert out.index("%sencode" % P._STAGE_PREFIX) < out.index("%sstack1:0" % P._SIGNPOST_PREFIX)


def test_a_sequence_inside_the_budget_is_untouched():
    seq = ["ttnn.op(%d)" % i for i in range(5)]
    assert P._printable_seq(seq, cap=50) == seq


def test_every_control_token_the_readers_know_is_kept():
    """Taken from the one vocabulary, so this cannot drift from _is_control."""
    for pref in V.control_prefixes():
        tok = "%sx" % pref
        assert _is_control(tok), "the reader treats %r as structure" % pref
        assert P._printable_seq(["op", tok], cap=1) == ["op", tok]


def test_the_print_no_longer_carries_a_head_slice():
    src = (_PA / "cc_optimize" / "_op_sig_probe.py").read_text()
    assert "_SEQ[:50000]" not in src, "the head slice is gone from the print"
    assert "_printable_seq(_SEQ)" in src


def test_the_cut_keeps_what_the_readers_call_structure_and_nothing_else():
    """The filter is the readers' own predicate, so a marker invented later survives the cut too."""
    assert P._printable_seq(["op", "%sx" % V.CALLER], cap=1) == ["op", "%sx" % V.CALLER]
    assert P._printable_seq(["op", "another_op"], cap=1) == ["op"], "ops past the cut are still cut"
