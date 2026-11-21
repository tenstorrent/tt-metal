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


def test_the_work_signal_is_the_true_total_not_the_cut():
    """The head is the same slice it was, but the work signal no longer saturates at it: the repeats
    elided past the cut are counted back in. A full-depth run and a capped one used to read the same
    number once both passed the cut (GLM-5.3-Flash: 49478 vs 49912), declaring a working knob INERT."""
    seq = ["ttnn.op(%d)" % (i % 3) for i in range(50)]
    seq += ["%sstack3:%d" % (P._SIGNPOST_PREFIX, i) for i in range(4)]
    out = P._printable_seq(seq, cap=8)
    assert out[:8] == seq[:8], "the head is unchanged"
    assert _work_signal(out) == _work_signal(seq) == 50


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
    # past the cut an op's repeats are cut and counted; only its first appearance in a segment stays
    assert P._printable_seq(["op", "x", "x", "x"], cap=1) == ["op", "x", "%s2" % V.ELIDED]


def test_a_first_appearance_past_the_cut_still_sizes_the_window():
    """GLM-5.3-Flash 2026-11-09: three identical layers filled the budget, and layer 3 -- the first
    with a new kind of op -- signposted with nothing in it, so coverage sized the stack at 2."""
    from cc_optimize.run import _first_block_map

    seq = []
    for i in range(6):
        seq.append("%sstack3:%d" % (P._SIGNPOST_PREFIX, i))
        seq += ["ttnn.common"] * 10 + (["ttnn.deep_kind"] if i >= 3 else [])
        seq.append("%sstack3:%d" % (P._SIGNPOST_END_PREFIX, i))
    out = P._printable_seq(seq, cap=25)  # the cut lands inside block 2
    fb, _ = _first_block_map(out)
    assert fb["stack3"]["ttnn.deep_kind"] == 3
    assert _work_signal(out) == _work_signal(seq)
    assert len(out) < len(seq)


def test_each_segment_keeps_its_own_first_appearance():
    """Per segment, not per sequence: a block past the cut still shows every op it ran once, which is
    what the per-block readers (first appearance, lever coverage) need."""
    seq = ["h"] + ["%sstack0:%d" % (P._SIGNPOST_PREFIX, i) if j == 0 else "op" for i in range(3) for j in range(3)]
    out = P._printable_seq(seq, cap=1)
    assert out == [
        "h",
        "%sstack0:0" % P._SIGNPOST_PREFIX,
        "op",
        "%s1" % V.ELIDED,
        "%sstack0:1" % P._SIGNPOST_PREFIX,
        "op",
        "%s1" % V.ELIDED,
        "%sstack0:2" % P._SIGNPOST_PREFIX,
        "op",
        "%s1" % V.ELIDED,
    ]


def test_an_elided_count_is_structure_that_carries_work():
    tok = "%s7" % V.ELIDED
    assert _is_control(tok) and V.elided(tok) == 7
    assert V.elided("ttnn.op") == 0 and V.elided("%sjunk" % V.ELIDED) == 0
