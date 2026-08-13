"""A block index must come from the signpost's OWN index, not from counting signposts.

_install_block_signposts tags each block in the stack with its position (_tag_stack: setattr(blk,
tag, i)) and emits "PERF_BLOCK_SIGNPOST:<i>" on entry. That payload is the layer index, in the same
unit as TT_PERF_LAYERS. _first_block_map discarded it: _block_start_positions returned only WHERE the
signposts sat in the sequence, and a bisect turned that into an ordinal -- "how many block entries
precede this op".

Those two numbers agree only when the model is entered exactly once. A perf test that prefills and
then decodes enters all 48 layers twice, so the sequence holds 96 signposts and the ordinal runs
0..95: decode's layer 0 is reported as block 48. gemma-3-12b-it came out of this with a coverage
depth of 96 for a 48-layer model, which was then handed to TT_PERF_LAYERS -- a layer knob -- and
stamped into the ledger, and the report said "96 layers".

The ordinal path stays for models whose blocks raise no signposts at all (source "inferred"), where
counting is the only information available.

NOTE (Task 3): _first_block_map now returns {stack_id: {op: block}} for multi-stack support.
Single-stack sequences (old PERF_BLOCK_SIGNPOST:N format) return {"stack0": {op: block}}.
Tests below use _s0(fb) to extract the stack0 flat dict for single-stack assertions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))

from models.experimental.perf_automation.cc_optimize.run import _first_block_map  # noqa: E402

SP = "PERF_BLOCK_SIGNPOST:%d"


def _pass(n_layers, ops_by_layer=None):
    """One forward pass over n_layers, each layer emitting a Matmul plus anything extra."""
    seq = []
    for i in range(n_layers):
        seq.append(SP % i)
        seq.append("Matmul")
        for op in (ops_by_layer or {}).get(i, ()):
            seq.append(op)
    return seq


def _s0(per_stack):
    """Extract the stack0 flat map from a per-stack dict (single-stack helper)."""
    return per_stack.get("stack0", per_stack)


# ---------------------------------------------------------------- the reported bug


def test_two_passes_over_48_layers_do_not_report_96_blocks():
    """The gemma3 shape: prefill then decode. 96 signposts, 48 layers."""
    seq = _pass(48) + _pass(48)
    per_stack, source = _first_block_map(seq)
    fb = _s0(per_stack)
    assert source == "signposts"
    assert max(fb.values()) <= 47, max(fb.values())


def test_an_op_unique_to_the_second_pass_is_attributed_to_its_real_layer():
    """A decode-only op in layer 3 must read as block 3, not block 51. This is the whole failure:
    the inflated index is what set the coverage window to 96."""
    seq = _pass(48) + _pass(48, {3: ("DecodeOnlyOp",)})
    fb = _s0(_first_block_map(seq)[0])
    assert fb["DecodeOnlyOp"] == 3, fb["DecodeOnlyOp"]


def test_the_resulting_window_is_a_layer_count_not_a_signpost_count():
    """deepest + 1 is what becomes TT_PERF_LAYERS. It must never exceed the model's layer count."""
    seq = _pass(48) + _pass(48) + _pass(48)
    fb = _s0(_first_block_map(seq)[0])
    assert max(fb.values()) + 1 <= 48


# ---------------------------------------------------------------- single pass unchanged


def test_a_single_pass_is_unaffected():
    """The case that always worked: ordinal and index coincide, so the answer must not move."""
    seq = _pass(16, {9: ("RareOp",)})
    fb = _s0(_first_block_map(seq)[0])
    assert fb["RareOp"] == 9 and fb["Matmul"] == 0


def test_first_appearance_still_wins_over_later_ones():
    """fb records where an op FIRST appears; a later repeat must not overwrite it."""
    seq = _pass(8, {1: ("Shared",), 5: ("Shared",)})
    fb = _s0(_first_block_map(seq)[0])
    assert fb["Shared"] == 1


# ---------------------------------------------------------------- shapes that break naive parsing


def test_a_stack_entered_out_of_order_reads_the_index_not_the_order():
    """Nothing guarantees blocks fire in ascending order (a model may run a shared block early). The
    index is authoritative; the position in the sequence is not."""
    seq = [SP % 5, "Matmul", "OpA", SP % 2, "Matmul", "OpB", SP % 9, "Matmul", "OpC"]
    fb = _s0(_first_block_map(seq)[0])
    assert (fb["OpA"], fb["OpB"], fb["OpC"]) == (5, 2, 9), fb


def test_ops_outside_every_block_belong_to_no_block():
    """Embedding-side ops run before any block is entered, and they must not set a stack's depth.

    This used to assert they land in block 0. Same effect on sizing -- block 0 never raises max() --
    but attributing them at all was the loophole: with only a START marker, everything AFTER a
    stack's last block was attributed to that block too, and on Voxtral that meant 12573 ops (67% of
    the run, including the whole decode phase) credited to encoder block 31, which sized a 32-layer
    encoder at 32 when 1 would do. Now a block has both edges and an op outside every block is
    attributed to none.
    """
    seq = ["Embedding"] + _pass(4)
    fb = _s0(_first_block_map(seq)[0])
    assert "Embedding" not in fb, "a pre-block op is still attributed to a block"
    assert fb, "the blocks' own ops were lost too"


def test_a_malformed_signpost_does_not_crash_or_reset_the_block():
    """The probe writes these strings; a truncated or non-numeric one must degrade, not throw, and
    must not silently attribute the rest of the stack to block 0."""
    seq = [SP % 3, "Matmul", "OpA", "PERF_BLOCK_SIGNPOST:", "Matmul", "OpB", "PERF_BLOCK_SIGNPOST:xyz", "Matmul", "OpC"]
    fb = _s0(_first_block_map(seq)[0])
    assert fb["OpA"] == 3 and fb["OpB"] == 3 and fb["OpC"] == 3, fb


def test_a_negative_index_is_ignored():
    seq = [SP % 2, "Matmul", "OpA", "PERF_BLOCK_SIGNPOST:-4", "Matmul", "OpB"]
    fb = _s0(_first_block_map(seq)[0])
    assert fb["OpB"] == 2, fb


def test_non_string_tokens_are_skipped():
    fb = _s0(_first_block_map([SP % 1, "Matmul", None, 7, "OpA", SP % 2, "Matmul"])[0])
    assert fb["OpA"] == 1


def test_an_empty_sequence_is_empty_not_an_error():
    assert _first_block_map([]) == ({}, "none")
    assert _first_block_map(None) == ({}, "none")


# ---------------------------------------------------------------- the fallback path


def test_a_sequence_with_no_signposts_still_infers_blocks():
    """Models whose blocks raise nothing keep the ordinal path -- counting is all there is. Removing
    it would blind every model the signpost walker cannot tag."""
    seq = []
    for i in range(4):
        seq += ["Anchor", "Matmul"] + (["Deep"] if i == 3 else [])
    per_stack, source = _first_block_map(seq)
    fb = _s0(per_stack)
    assert source == "inferred", source
    assert fb["Deep"] == 3, fb
