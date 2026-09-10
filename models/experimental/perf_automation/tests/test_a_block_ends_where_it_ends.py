# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Work done after a stack's last block is not that block's work.

A BLOCK HAS TWO EDGES AND ONLY ONE WAS MARKED. `PERF_BLOCK_SIGNPOST:stackN:i` says where block i
begins. Nothing said where it ended, so attribution credited every op after the final block of a
stack to that block -- and "after the final block" is the entire rest of the model.

MEASURED ON VOXTRAL-MINI-3B, 2026-08-13, from the probe's own sequence:

    encoder stack: a normal block dispatches      20 ops
                   the LAST block was credited  12573 ops   (67% of the whole run)
    op types unique to that "block": embedding, rms_norm, silu,
        scaled_dot_product_attention_decode, paged_update_cache, argmax

Those are language-model decode ops. They are not in the encoder. They ran after it.

WHAT IT COST. Coverage asks "how deep must I profile before every op type has appeared". With the LM
phase folded into encoder block 31, the answer for a 32-layer encoder was 32 -- when all 32 of its
blocks are one class emitting identical ops and 1-2 would cover it. max(2, 32, 3) then took 32 as the
window; 32 IS the encoder's full depth, so capping to it changed no work; the run concluded the depth
knob never reached the builder and profiled the entire model. That is the 18729-op, 35M-tracy-zone
path that killed every run this week.

THE RULE. An op emitted outside every block belongs to no stack -- prologue, epilogue or another
stage -- and says nothing about how deep a stack must be profiled.
"""

import importlib.util
import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))


def _run():
    spec = importlib.util.spec_from_file_location("_run_blocks", _PA / "cc_optimize" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _voxtral_shaped():
    """The measured shape: a 32-block encoder, then the LM phase, then a 3-block LM stack.

    Every encoder block emits the same two ops, so a correct coverage answer is 1.
    """
    seq = []
    for i in range(32):
        seq.append("PERF_BLOCK_SIGNPOST:stack2:%d" % i)
        seq += ["ttnn.matmul((1280,))", "ttnn.layer_norm((1280,))"]
        seq.append("PERF_BLOCK_SIGNPOST_END:stack2:%d" % i)
    # the projection + the whole decode phase -- outside every block
    seq += ["ttnn.slice((375,))", "ttnn.to_layout((375,))", "ttnn.embedding((3072,))"]
    for i in range(3):
        seq.append("PERF_BLOCK_SIGNPOST:stack3:%d" % i)
        seq += ["ttnn.rms_norm((3072,))", "ttnn.silu((3072,))"]
        seq.append("PERF_BLOCK_SIGNPOST_END:stack3:%d" % i)
    seq += ["ttnn.argmax((128256,))", "ttnn.copy((1,))"]
    return seq


def test_the_epilogue_does_not_set_the_encoders_depth():
    """THE BUG, on the shape it was measured on."""
    run = _run()
    per_stack, source = run._first_block_map(_voxtral_shaped())
    assert source == "signposts"

    enc = per_stack["stack2"]
    assert max(enc.values()) == 0, "the encoder still needs more than its first block: %s" % {
        k: v for k, v in enc.items() if v > 0
    }
    for op in ("ttnn.embedding((3072,))", "ttnn.argmax((128256,))", "ttnn.slice((375,))"):
        assert op not in enc, "%s was attributed to the encoder; it runs after it" % op

    lm = per_stack["stack3"]
    assert max(lm.values()) == 0
    assert "ttnn.argmax((128256,))" not in lm, "the trailing argmax was attributed to the LM stack"


def test_the_coverage_window_collapses_from_full_depth_to_one():
    """What the fix is FOR: the window that goes to the depth knob.

    Before, this shape sized the encoder at 32 -- its own full depth -- so capping could not reduce
    work and the run profiled everything.
    """
    run = _run()
    per_stack, _ = run._first_block_map(_voxtral_shaped())
    windows = {sid: max(m.values()) + 1 for sid, m in per_stack.items()}
    assert windows == {"stack2": 1, "stack3": 1}, windows
    assert max(windows.values()) < 32, "max() still asks for the model's full depth"


def test_an_op_outside_every_block_is_attributed_to_nothing():
    """Prologue and epilogue belong to no block.

    Two blocks, not one: a lone marker leaves the sequence without usable signposts, so attribution
    falls back to the INFERRED path, which has no end information and can only bisect on starts. That
    path cannot exclude an epilogue and is unchanged here -- it is what a model with no markers has
    always had.
    """
    run = _run()
    per_stack, source = run._first_block_map(
        [
            "ttnn.embedding((1,))",  # prologue, before any block
            "PERF_BLOCK_SIGNPOST:stack0:0",
            "ttnn.matmul((1,))",
            "PERF_BLOCK_SIGNPOST_END:stack0:0",
            "PERF_BLOCK_SIGNPOST:stack0:1",
            "ttnn.matmul((1,))",
            "PERF_BLOCK_SIGNPOST_END:stack0:1",
            "ttnn.argmax((1,))",  # epilogue
        ]
    )
    assert source == "signposts"
    assert per_stack == {"stack0": {"ttnn.matmul((1,))": 0}}


def test_a_stack_entered_twice_still_reports_layer_indices():
    """Prefill then decode enters every layer twice. The signpost carries the LAYER index, so the
    second pass must not read as blocks 32..63 -- the ordinal bug that once sized a 48-layer model
    at 96."""
    run = _run()
    seq = []
    for _pass in range(2):
        for i in range(4):
            seq += [
                "PERF_BLOCK_SIGNPOST:stack0:%d" % i,
                "ttnn.matmul((1,))",
                "PERF_BLOCK_SIGNPOST_END:stack0:%d" % i,
            ]
    per_stack, _ = run._first_block_map(seq)
    assert max(per_stack["stack0"].values()) == 0


def test_the_probe_closes_the_block_even_when_it_raises():
    """An unclosed window swallows the rest of the run exactly as the missing marker did."""
    src = (_PA / "cc_optimize" / "_op_sig_probe.py").read_text()
    i = src.index("def _wrapped(self, *a, __orig=orig, **k):")
    body = src[i : i + 420]
    assert "finally:" in body, "a block that raises never emits its end marker"
    assert "_emit(self, end=True)" in body


def test_the_end_marker_is_never_counted_as_device_work():
    """Same trap as the caller markers: a bookkeeping token counted as an op inflates the work signal
    that decides whether a cap did anything -- on a 32-layer stack that is 32 phantom ops."""
    run = _run()
    assert run._is_control("PERF_BLOCK_SIGNPOST_END:stack2:31")
    assert (
        run._work_signal(["PERF_BLOCK_SIGNPOST:stack2:0", "PERF_BLOCK_SIGNPOST_END:stack2:0", "ttnn.matmul(())"]) == 1
    )


def test_an_end_marker_is_not_mistaken_for_a_start():
    """The two tokens share a prefix up to the underscore; a naive startswith would count every end
    marker as another block start and double the reported depth."""
    run = _run()
    assert not "PERF_BLOCK_SIGNPOST_END:stack2:31".startswith(run._SIGNPOST_TOKEN)
    assert run._blocks_ran(["PERF_BLOCK_SIGNPOST:stack2:%d" % i for i in range(4)]) == 4
    assert (
        run._blocks_ran(
            ["PERF_BLOCK_SIGNPOST:stack2:%d" % i for i in range(4)]
            + ["PERF_BLOCK_SIGNPOST_END:stack2:%d" % i for i in range(4)]
        )
        == 4
    )


def test_a_stack_nested_inside_a_block_does_not_lose_the_outer_blocks_work():
    """CLOSING AN INNER BLOCK RETURNS TO THE ENCLOSING ONE, NOT TO NOTHING.

    A stack can sit inside a block -- experts within a layer, a decoder nested in a wrapper. If the
    end marker clears the current block instead of popping back, every op the OUTER block runs after
    the nested call is attributed to nothing and under-sizes it: exactly the ops that would have
    demanded a deeper window go uncounted. Flat stacks behave identically either way, which is how
    this hides.
    """
    run = _run()
    seq = [
        "PERF_BLOCK_SIGNPOST:stack0:0",
        "ttnn.pre((1,))",
        "PERF_BLOCK_SIGNPOST:stack1:0",
        "ttnn.expert((1,))",
        "PERF_BLOCK_SIGNPOST_END:stack1:0",
        "ttnn.post((1,))",  # still inside stack0 block 0
        "PERF_BLOCK_SIGNPOST_END:stack0:0",
        "ttnn.after((1,))",  # outside everything
        "PERF_BLOCK_SIGNPOST:stack0:1",
        "ttnn.pre((1,))",
        "PERF_BLOCK_SIGNPOST_END:stack0:1",
    ]
    per_stack, _ = run._first_block_map(seq)
    assert per_stack["stack0"] == {"ttnn.pre((1,))": 0, "ttnn.post((1,))": 0}, per_stack["stack0"]
    assert per_stack["stack1"] == {"ttnn.expert((1,))": 0}
    assert "ttnn.after((1,))" not in per_stack["stack0"]


def test_an_unbalanced_end_marker_does_not_unwind_the_wrong_block():
    """A block that dies before its start was recorded emits an end for a frame nobody opened. That
    must not close the enclosing block and silently drop the rest of its work."""
    run = _run()
    seq = [
        "PERF_BLOCK_SIGNPOST:stack0:0",
        "ttnn.a((1,))",
        "PERF_BLOCK_SIGNPOST_END:stack9:7",  # never opened
        "ttnn.b((1,))",
        "PERF_BLOCK_SIGNPOST_END:stack0:0",
    ]
    per_stack, _ = run._first_block_map(seq)
    assert per_stack["stack0"] == {"ttnn.a((1,))": 0, "ttnn.b((1,))": 0}, per_stack
