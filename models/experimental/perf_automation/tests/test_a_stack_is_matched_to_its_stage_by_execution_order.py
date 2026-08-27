# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Which knob gets which number, decided by when each stack runs.

THE GAP THIS CLOSES. Coverage is sized PER STACK -- one saturates at 2, another may need 8 -- and a
model can now accept a depth per stage. Nothing connected the two: the walk labels stacks by
traversal position (stack2, stack3) and the knobs are named for stages, so nothing said stack2 IS
the encoder. The tool therefore sent max() to every stack, and on Voxtral-Mini-3B that means both
audio encoders profiled at the text decoder's depth -- several times deeper than their own op
coverage requires, on the section that dominates the op count.

EXECUTION ORDER IS THE LINK, and it needs no HF reference, no naming heuristic and no new
convention: whichever blocks run between the encode and prefill boundaries belong to encode. The
stage names come from PIPELINE_STAGES, which the contract already requires, and the boundaries are
emitted by the probe wrapping the per-stage hooks the model already exposes.

A STACK IN TWO WINDOWS IS NOT AMBIGUOUS. A text decoder runs in prefill AND decode; it is one
physical stack, so it appears under both and takes the max of their depths -- deep enough for either.
Assigning it a single stage would be the error; listing both is the fact.

CONSERVATIVE EVERYWHERE ELSE. Every unresolved case -- no stage boundaries emitted, a stack never
called, stages that interleave -- returns empty, and empty means "one uniform depth", which is
exactly the behaviour this replaces. Nothing ends up shallower than it is today, so the worst case is
the status quo and the good case is cheaper.
"""

import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))

_S = "PERF_STAGE_SIGNPOST:"
_B = "PERF_BLOCK_SIGNPOST:"


def _seq():
    """encode runs stack2; prefill and decode both run stack3 -- Voxtral's shape."""
    return [
        _S + "encode",
        _B + "stack2:0",
        _B + "stack2:1",
        _S + "prefill",
        _B + "stack3:0",
        _B + "stack3:1",
        _S + "decode",
        _B + "stack3:0",
        _B + "stack3:1",
        _B + "stack3:2",
    ]


def test_each_stage_lists_the_stacks_that_ran_in_it():
    from cc_optimize.run import stacks_by_stage

    assert stacks_by_stage(_seq()) == {
        "encode": ["stack2"],
        "prefill": ["stack3"],
        "decode": ["stack3"],
    }


def test_a_shallow_stack_is_no_longer_dragged_to_the_deepest():
    """The whole point: the encoder needs 2 and stops being profiled at the decoder's 8."""
    from cc_optimize.run import depth_per_stage

    got = depth_per_stage({"stack2": 2, "stack3": 8}, _seq())
    assert got["encode"] == 2, "the shallow stack is still taking the global max"


def test_a_shared_stack_takes_the_max_of_its_stages():
    """One physical stack cannot have two depths; the deeper requirement wins and covers both."""
    from cc_optimize.run import depth_per_stage

    got = depth_per_stage({"stack2": 2, "stack3": 8}, _seq())
    assert got["prefill"] == got["decode"] == 8


def test_missing_stage_boundaries_fall_back_rather_than_guess():
    """No signposts, an uncalled stack, interleaved stages -- all land here, and empty means the
    single uniform depth the tool used before."""
    from cc_optimize.run import depth_per_stage

    assert depth_per_stage({"stack2": 2}, [_B + "stack2:0"]) == {}
    assert depth_per_stage({}, _seq()) == {}
    assert depth_per_stage({"stack2": 2}, []) == {}


def test_the_probe_emits_the_boundaries_from_the_models_own_stages():
    """No per-model code: PIPELINE_STAGES is already contract-required, and the hooks already exist."""
    src = (_PA / "cc_optimize" / "_op_sig_probe.py").read_text()
    assert "_mark_stage_boundaries(" in src, "stage boundaries are never emitted"
    i = src.index("def _mark_stage_boundaries(")
    body = src[i : i + 3000]
    assert "PIPELINE_STAGES" in body, "the stage names are invented rather than read from the model"
    # The hook NAME now comes from the seam registry rather than being spelled here. Asserting the
    # literal "_trace_step" pinned the spelling, which is the duplication stage_seams exists to end;
    # what the test is actually about is that the per-stage hooks are what gets bound.
    assert "_seams.STEP" in body and "_seams.hook(" in body, "the per-stage hooks are not used"
