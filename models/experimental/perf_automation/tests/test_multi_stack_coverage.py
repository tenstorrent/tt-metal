# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tests for per-stack coverage computation in run.py (Task 3).

WHY THIS EXISTS
    Task 3 extends _first_block_map() to return per-stack depth maps and updates
    _coverage_layers() to return {stack_id: depth} dicts instead of a single int.
    This enables independent coverage windows for each block stack (e.g. Voxtral-Mini:
    32-layer encoder + 30-layer decoder can have different optimal depths).

    All tests use synthetic op sequences (lists of strings) — no model weights needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))

from models.experimental.perf_automation.cc_optimize.run import (  # noqa: E402
    _first_block_map,
    _parse_signpost_payload,
    _signposts_usable,
)

# Signpost format constants
SP_OLD = "PERF_BLOCK_SIGNPOST:%d"  # single-stack (old format)
SP_NEW = "PERF_BLOCK_SIGNPOST:stack%d:%d"  # multi-stack (new format)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_single_stack_seq(n_layers, ops_by_layer=None):
    """Synthetic op sequence using old single-stack signpost format."""
    seq = []
    for i in range(n_layers):
        seq.append(SP_OLD % i)
        seq.append("Matmul")
        for op in (ops_by_layer or {}).get(i, ()):
            seq.append(op)
    return seq


def _make_two_stack_seq(n0, n1, ops_by_stack_layer=None):
    """Synthetic op sequence with two stacks using new stack-prefixed format.

    n0 layers in stack0 (e.g. encoder), n1 layers in stack1 (e.g. decoder).
    ops_by_stack_layer: {(stack_idx, layer_idx): [ops]}
    """
    ops_by_stack_layer = ops_by_stack_layer or {}
    seq = []
    # Interleave both stacks: stack0 then stack1 layers
    for i in range(max(n0, n1)):
        if i < n0:
            seq.append(SP_NEW % (0, i))
            seq.append("EncoderOp")
            for op in ops_by_stack_layer.get((0, i), ()):
                seq.append(op)
        if i < n1:
            seq.append(SP_NEW % (1, i))
            seq.append("DecoderOp")
            for op in ops_by_stack_layer.get((1, i), ()):
                seq.append(op)
    return seq


# ---------------------------------------------------------------------------
# Test 1: _parse_signpost_payload handles both formats
# ---------------------------------------------------------------------------


def test_parse_old_format_returns_stack0():
    """Old format PERF_BLOCK_SIGNPOST:N maps to stack0."""
    assert _parse_signpost_payload("5") == ("stack0", 5)
    assert _parse_signpost_payload("0") == ("stack0", 0)
    assert _parse_signpost_payload("47") == ("stack0", 47)


def test_parse_new_format_stack0():
    """New format PERF_BLOCK_SIGNPOST:stack0:N maps to stack0."""
    assert _parse_signpost_payload("stack0:5") == ("stack0", 5)
    assert _parse_signpost_payload("stack0:0") == ("stack0", 0)


def test_parse_new_format_stack1():
    """New format PERF_BLOCK_SIGNPOST:stack1:N maps to stack1."""
    assert _parse_signpost_payload("stack1:12") == ("stack1", 12)
    assert _parse_signpost_payload("stack1:0") == ("stack1", 0)


def test_parse_malformed_returns_none():
    """Malformed payloads must return None, not raise."""
    assert _parse_signpost_payload("") is None
    assert _parse_signpost_payload("xyz") is None
    assert _parse_signpost_payload("stack0") is None  # no colon after stack0
    assert _parse_signpost_payload("stack0:abc") is None
    assert _parse_signpost_payload("-4") is None


# ---------------------------------------------------------------------------
# Test 2: _first_block_map single-stack (old format) → {"stack0": flat_dict}
# ---------------------------------------------------------------------------


def test_single_stack_old_format_wraps_in_stack0():
    """Old-format PERF_BLOCK_SIGNPOST:N sequence must return {"stack0": {op: block}}."""
    seq = _make_single_stack_seq(8)
    per_stack, source = _first_block_map(seq)
    assert source == "signposts"
    assert set(per_stack.keys()) == {"stack0"}
    fb = per_stack["stack0"]
    assert fb["Matmul"] == 0


def test_single_stack_first_appearance_correct():
    """Op first appearing in layer 3 must map to block 3."""
    seq = _make_single_stack_seq(10, ops_by_layer={3: ("RareOp",)})
    per_stack, _ = _first_block_map(seq)
    fb = per_stack["stack0"]
    assert fb["RareOp"] == 3
    assert fb["Matmul"] == 0


def test_single_stack_two_passes_do_not_inflate_blocks():
    """Two passes over 48 layers (gemma3 shape): max block must stay ≤ 47."""
    seq = _make_single_stack_seq(48) + _make_single_stack_seq(48)
    per_stack, source = _first_block_map(seq)
    fb = per_stack["stack0"]
    assert source == "signposts"
    assert max(fb.values()) <= 47


# ---------------------------------------------------------------------------
# Test 3: _first_block_map two-stack (new format) → independent per-stack dicts
# ---------------------------------------------------------------------------


def test_two_stack_returns_both_stack_keys():
    """Multi-stack sequence must return {"stack0": {...}, "stack1": {...}}."""
    seq = _make_two_stack_seq(4, 3)
    per_stack, source = _first_block_map(seq)
    assert source == "signposts"
    assert "stack0" in per_stack
    assert "stack1" in per_stack


def test_two_stack_ops_attributed_to_correct_stack():
    """EncoderOp must be in stack0, DecoderOp must be in stack1."""
    seq = _make_two_stack_seq(4, 3)
    per_stack, _ = _first_block_map(seq)
    assert "EncoderOp" in per_stack["stack0"]
    assert "DecoderOp" in per_stack["stack1"]
    # Cross-stack contamination: EncoderOp must not appear in stack1
    assert "EncoderOp" not in per_stack.get("stack1", {})
    assert "DecoderOp" not in per_stack.get("stack0", {})


def test_two_stack_depths_are_independent():
    """A rare op in stack0:layer5 must report depth 5 for stack0, not affect stack1."""
    seq = _make_two_stack_seq(8, 4, ops_by_stack_layer={(0, 5): ("EarlyOnlyEnc",)})
    per_stack, _ = _first_block_map(seq)
    fb0 = per_stack["stack0"]
    fb1 = per_stack["stack1"]
    assert fb0["EarlyOnlyEnc"] == 5
    assert "EarlyOnlyEnc" not in fb1


def test_two_stack_first_appearance_per_stack():
    """First appearance is recorded per-stack independently."""
    seq = _make_two_stack_seq(
        6,
        5,
        ops_by_stack_layer={
            (0, 2): ("EncSpecialOp",),
            (1, 3): ("DecSpecialOp",),
        },
    )
    per_stack, _ = _first_block_map(seq)
    assert per_stack["stack0"]["EncSpecialOp"] == 2
    assert per_stack["stack1"]["DecSpecialOp"] == 3


def test_two_stack_block_indices_from_signpost_not_ordinal():
    """Block indices must come from the signpost payload, not ordinal position."""
    # Interleave: stack0:5 then stack1:2 then stack0:2 then stack1:5
    seq = [
        SP_NEW % (0, 5),
        "OpA",
        SP_NEW % (1, 2),
        "OpB",
        SP_NEW % (0, 2),
        "OpC",
        SP_NEW % (1, 5),
        "OpD",
    ]
    per_stack, source = _first_block_map(seq)
    assert source == "signposts"
    assert per_stack["stack0"]["OpA"] == 5
    assert per_stack["stack1"]["OpB"] == 2
    assert per_stack["stack0"]["OpC"] == 2
    assert per_stack["stack1"]["OpD"] == 5


# ---------------------------------------------------------------------------
# Test 4: _signposts_usable works with both formats
# ---------------------------------------------------------------------------


def test_signposts_usable_old_format():
    """Old-format sequence with 2+ distinct signposts must be usable."""
    seq = _make_single_stack_seq(4)
    assert _signposts_usable(seq) is True


def test_signposts_usable_new_multi_stack_format():
    """New multi-stack format must also be recognized as usable."""
    seq = _make_two_stack_seq(4, 3)
    assert _signposts_usable(seq) is True


def test_signposts_usable_empty_sequence():
    """Empty sequence must not be usable."""
    assert _signposts_usable([]) is False
    assert _signposts_usable(None) is False


def test_signposts_usable_single_signpost_not_usable():
    """Only one distinct signpost cannot delimit any blocks."""
    seq = ["PERF_BLOCK_SIGNPOST:stack0:0", "OpA", "OpB", "OpA"]
    assert _signposts_usable(seq) is False


# ---------------------------------------------------------------------------
# Test 5: Empty sequence returns ({}, "none")
# ---------------------------------------------------------------------------


def test_empty_sequence_returns_empty_dict():
    """Empty and None sequences must return ({}, 'none'), not raise."""
    assert _first_block_map([]) == ({}, "none")
    assert _first_block_map(None) == ({}, "none")


# ---------------------------------------------------------------------------
# Test 6: Malformed multi-stack signposts degrade gracefully
# ---------------------------------------------------------------------------


def test_malformed_multi_stack_signpost_does_not_reset_block():
    """A malformed stack signpost must not reset the current block pointer."""
    seq = [
        SP_NEW % (0, 3),
        "OpA",
        "PERF_BLOCK_SIGNPOST:stack0:",  # malformed: no index
        "OpB",
        "PERF_BLOCK_SIGNPOST:stack0:notanint",  # malformed: non-numeric
        "OpC",
    ]
    per_stack, source = _first_block_map(seq)
    assert source == "signposts"
    fb0 = per_stack.get("stack0", {})
    # OpA was set at block 3; malformed signposts must not move it
    assert fb0.get("OpA") == 3
    # OpB and OpC stay attributed to the last valid block (3)
    assert fb0.get("OpB") == 3
    assert fb0.get("OpC") == 3


# ---------------------------------------------------------------------------
# Test 7: Single-stack callers get a single-entry dict (compat shim)
# ---------------------------------------------------------------------------


def test_single_stack_dict_has_exactly_one_entry():
    """Single-stack sequences return a dict with exactly 1 entry — callers can extract the value."""
    seq = _make_single_stack_seq(16)
    per_stack, source = _first_block_map(seq)
    assert source == "signposts"
    assert len(per_stack) == 1
    assert list(per_stack.keys()) == ["stack0"]


def test_two_stack_dict_has_exactly_two_entries():
    """Two-stack sequences return a dict with exactly 2 entries."""
    seq = _make_two_stack_seq(4, 3)
    per_stack, source = _first_block_map(seq)
    assert source == "signposts"
    assert len(per_stack) == 2
    assert set(per_stack.keys()) == {"stack0", "stack1"}
