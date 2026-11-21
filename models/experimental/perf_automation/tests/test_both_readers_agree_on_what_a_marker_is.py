# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The op sequence is read by two modules, and they must mean the same thing by "a marker".

run.py and perf_mcp.py both carry the DECOUPLED SIGNPOSTS guard: markers that all land in a clump
with no ops after them delimit nothing, because every op would attribute to block 0. The guard asks
whether any work follows the first marker -- so it can only be as correct as the reader's idea of
what work is. One module excluded all four control prefixes; the other excluded only the block
START prefix, and a block END token does not begin with it. The same sequence was therefore usable
to one reader and unusable to the other, and the lenient one decides which blocks
check_lever_coverage reports a dtype lever reached.

The clump never arrived while the printed sequence was a head slice that truncated the markers away.
It arrives now that the cut keeps them, which is what makes this worth pinning down.

Op and stage names here are this file's own. The rule is about the vocabulary, not about any model.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from agent import marker_vocab as V
from cc_optimize import _op_sig_probe as PROBE
from cc_optimize import run as R

_PA = Path(__file__).resolve().parents[1]

_SPEC = importlib.util.spec_from_file_location("perf_mcp_vocab", str(_PA / "cc_optimize" / "perf_mcp.py"))
MCP = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(MCP)


def _clumped(n_ops: int = 20, n_blocks: int = 4) -> list:
    """What a stage running past the print budget leaves behind: ops, then markers, no ops after."""
    seq = ["ttnn.op(%d)" % (i % 5) for i in range(n_ops)]
    for b in range(n_blocks):
        seq += ["%sstack2:%d" % (V.BLOCK, b), "%sstack2:%d" % (V.BLOCK_END, b)]
    return seq


def _interleaved(n_blocks: int = 4) -> list:
    seq = ["embed(1)"]
    for b in range(n_blocks):
        seq += ["%s%d" % (V.BLOCK, b), "norm(1)", "matmul(1)", "%s%d" % (V.BLOCK_END, b)]
    return seq + ["lmhead(1)"]


def test_a_block_end_does_not_begin_with_the_block_start_prefix():
    """The whole defect in one line: the two tokens share a stem, so `startswith` separates them."""
    assert not V.BLOCK_END.startswith(V.BLOCK)
    assert V.is_control(V.BLOCK_END + "0"), "an end marker is structure, never work"


def test_a_clump_of_markers_is_unusable_to_both_readers():
    seq = _clumped()
    assert R._signposts_usable(seq) is False
    assert MCP._signposts_usable(seq) is False, "an END token is not the work this guard looks for"


def test_an_interleaved_run_is_usable_to_both_readers():
    """The guard must reject the clump without rejecting the shape it exists to accept."""
    seq = _interleaved()
    assert R._signposts_usable(seq) is True
    assert MCP._signposts_usable(seq) is True
    assert MCP._block_starts(seq, 4)[1] == "signposts"


def test_the_inferred_anchor_is_never_a_control_token():
    """The fallback reads one token per block off the sequence; a marker is not a candidate.

    A stage marker that happens to occur as often as there are blocks would otherwise be picked,
    handing back stage boundaries labelled as block starts.
    """
    seq = ["%ss%d" % (V.STAGE, i) for i in range(3)]
    seq += ["ttnn.op(%d)" % (i % 7) for i in range(30)]
    seq += ["%ss%d" % (V.STAGE, i) for i in range(3, 6)]  # six stage markers, no op occurs six times
    starts, source = MCP._block_starts(seq, 6)
    assert all(not V.is_control(seq[i]) for i in starts), "every anchor position holds an op"
    assert source in ("none", "inferred")


def test_a_marker_tail_does_not_change_what_the_lever_reader_sees():
    """check_lever_coverage is the user-visible consumer: its verdict must not move."""
    from collections import Counter

    base = _interleaved()
    base = [t.replace("matmul(1)", "matmul(1,'BFLOAT8_B')") for t in base]
    tailed = base + ["%sstack9:%d" % (V.BLOCK, b) for b in range(4)]
    before = MCP.compute_lever_coverage(Counter(base), base, "matmul(", "BFLOAT16", "BFLOAT8_B")
    after = MCP.compute_lever_coverage(Counter(tailed), tailed, "matmul(", "BFLOAT16", "BFLOAT8_B")
    assert before["fully_applied"] == after["fully_applied"]
    assert before["total_instances"] == after["total_instances"]


def test_every_reader_takes_its_prefixes_from_the_one_module():
    from agent.caller_stacks import MARK

    assert (R._SIGNPOST_TOKEN, R._SIGNPOST_END_TOKEN, R._STAGE_TOKEN) == (V.BLOCK, V.BLOCK_END, V.STAGE)
    assert (PROBE._SIGNPOST_PREFIX, PROBE._SIGNPOST_END_PREFIX, PROBE._STAGE_PREFIX) == (
        V.BLOCK,
        V.BLOCK_END,
        V.STAGE,
    )
    assert MCP._SIGNPOST_PREFIX == V.BLOCK
    assert MARK == V.CALLER, "the module that emits the caller mark takes its spelling from here too"


def test_no_module_spells_a_marker_for_itself():
    """A fifth copy is how the fourth one drifted. The literals live in exactly one file."""
    owner = _PA / "agent" / "marker_vocab.py"
    for rel in (
        "cc_optimize/run.py",
        "cc_optimize/perf_mcp.py",
        "cc_optimize/_op_sig_probe.py",
        "agent/caller_stacks.py",
    ):
        src = (_PA / rel).read_text()
        code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
        for literal in ('"%s"' % V.BLOCK, '"%s"' % V.BLOCK_END, '"%s"' % V.STAGE, '"%s"' % V.CALLER):
            # a docstring may quote a token when explaining the wire format; an assignment may not
            assert ("= " + literal) not in code, "%s assigns %s instead of importing it" % (rel, literal)
    assert all('"%s"' % p in owner.read_text() for p in V.control_prefixes())


def test_the_guard_agrees_on_the_sequence_shapes_a_probe_can_produce():
    """Both readers, every shape: nothing, ops only, markers only, clumped, interleaved."""
    shapes = {
        "empty": [],
        "ops only": ["ttnn.op(1)"] * 10,
        "markers only": ["%s%d" % (V.BLOCK, b) for b in range(3)],
        "clumped": _clumped(),
        "interleaved": _interleaved(),
        "interleaved then a marker tail": _interleaved() + ["%sstack9:%d" % (V.BLOCK, b) for b in range(3)],
    }
    for name, seq in shapes.items():
        assert R._signposts_usable(seq) == MCP._signposts_usable(seq), "the readers disagree on %r" % name
