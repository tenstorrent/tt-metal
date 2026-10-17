# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A model with more stacks than depth knobs is repaired, not just reported.

WHAT A MISSING KNOB ACTUALLY COSTS. A model with NO depth argument is refused by the contract before
the device opens. The case here looks fine and is worse: the factory accepts `layers`, the clause
passes, and the value reaches exactly ONE stack while every other stack builds at FULL depth.

Voxtral-Mini-3B, measured 2026-08-11/12: `layers` capped the text decoder and nothing else, so a
"2-layer" profile built 2 text layers behind two 32-layer audio encoders -- 18729 dispatched ops and
35.2M tracy zones, with the baseline killed at its budget and the run continuing with no BEFORE
number. Capping every stack took the same profile to 2471 ops and the build from 30+ minutes to 7.1
seconds.

WHY REPAIR RATHER THAN REPORT. Detection alone moves the manual work earlier and nothing else -- and
for a hand-written model the run does not even stop, it warns and proceeds uniformly capped. The
optimize loop already edits model source for every lever it tries, under a PCC gate that reverts
anything breaking correctness, and the walk has just produced the stack paths, so the agent is
handed targets rather than a search.

WHY NOT AN AST REWRITE. Adding parameters is mechanical; threading them is not. On Voxtral the depth
had to reach five places (n_layers, the routed-layer loop, layer_range, both encoder truncations)
plus a shared base class so the capped encoder stayed discoverable, and the first two hand-written
attempts built cleanly and died on the first forward. A rewriter that adds parameters it cannot wire
recreates the original defect: a knob accepted and ignored.

WHAT SETTLES IT. Not the agent's claim and not the signature: the depth bridge caps and re-measures
the work signal, and reports INERT when the op count does not move.
"""

import sys
import tempfile
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))


def _model(sig: str, stages: str = '["encode", "prefill", "decode"]') -> Path:
    d = Path(tempfile.mkdtemp())
    (d / "tt").mkdir()
    (d / "tt" / "pipeline.py").write_text(
        "PIPELINE_STAGES = %s\n\n\ndef build_pipeline(%s):\n    return None\n" % (stages, sig)
    )
    return d


def test_a_model_with_no_depth_argument_is_flagged_whatever_its_shape():
    """The base knob is the one that meets the goal, and it is needed for ONE stack as much as five.
    An earlier version returned [] for a single-stack model -- the case that most needs it. Measured
    on Voxtral with no depth argument and the variable unset: n_layers=30, enc_a=32, enc_b=32."""
    from agent.stack_knob_repair import missing_knobs

    d = _model("device, model=None, **kwargs")
    assert missing_knobs(d, 1) == ["layers"]
    assert missing_knobs(d, 3) == ["layers"], "stack count alone must not invent per-stage names"


def test_a_single_stack_model_is_left_alone():
    """`layers` describes it completely; demanding overrides would be noise."""
    from agent.stack_knob_repair import missing_knobs

    d = _model("device, model=None, layers=None, **kwargs")
    assert missing_knobs(d, 1) == []


def test_a_model_already_following_the_pattern_is_left_alone():
    from agent.stack_knob_repair import missing_knobs

    d = _model("device, layers=None, encode_layers=None")
    assert missing_knobs(d, 3) == []


def test_kwargs_does_not_count_as_a_knob():
    """A filtered **kwargs dict is exactly what dropped `layers` silently on Voxtral."""
    from agent.stack_knob_repair import factory_params

    d = _model("device, model=None, **kwargs")
    assert "layers" not in factory_params(d)


def test_per_stage_names_come_from_the_mapping_not_from_position():
    """Slicing PIPELINE_STAGES by stack count asked Voxtral for `prefill_layers` when both of its
    visible stacks run in encode. Names come from which stage each stack actually ran in, and with no
    mapping the honest answer is the base knob alone -- a wrong name is worse than a missing one."""
    from agent.stack_knob_repair import missing_knobs

    d = _model("device, layers=None, **kwargs", stages='["denoise", "vae"]')
    assert missing_knobs(d, 2) == [], "per-stage names were invented without a mapping"
    assert missing_knobs(d, 2, {"denoise": ["stack0", "stack1"]}) == ["denoise_layers"]


def test_the_prompt_carries_the_targets_and_the_traps():
    """The agent is handed the stacks the walk found, and the two rules that produced real crashes:
    0 is not a sentinel, and a capped build must stay runnable."""
    from agent.stack_knob_repair import repair_prompt

    d = _model("device, layers=None, **kwargs")
    p = repair_prompt(d, [("stack2", 4, "block"), ("stack3", 4, "block")], ["encode_layers"])
    assert "stack2" in p and "stack3" in p, "the agent is not told where the stacks are"
    assert "Never treat 0" in p, "the zero-layer trap is not stated"
    assert "RUNNABLE MODEL" in p, "the empty-sub-block trap is not stated"
    assert "cap from the END" in p, "the graduated-stub placement is not stated"


def test_making_a_model_cappable_is_not_optional():
    """CAPPING IS HOW THIS TOOL PROFILES AT ALL.

    Without a depth the builder can receive, every profile builds the whole model. On Voxtral-Mini-3B
    that is 36.8M tracy zones, tracy's own 32K source-location limit exceeded ("Instrumentation
    failure: Too many source locations"), a pytest process that then never exits, and the run killed
    at its budget having measured nothing. An operator cannot opt in to the tool working.

    It is also not the same as editing a model for PERFORMANCE, which stays a decision: the depth
    argument changes no numerics at full depth -- None means every layer, exactly as before -- and
    exists so the profiler can look at a slice. Instrumentation, verified by re-measuring.
    """
    src = (_PA / "cc_optimize" / "run.py").read_text()
    assert "stack_knob_repair" in src, "the run never calls the repair"
    i = src.index("def make_model_cappable(")
    body = src[i : i + 4500]
    assert "PERF_MCP_REPAIR_MODEL" not in body, "making a model measurable is still behind a flag"
    assert "builds the whole model" in body.lower(), "the cost is not stated"
    # and it hangs off the MEASURED verdict, not off the walk
    inert = src.index("depth knob is INERT: capping to")
    assert "make_model_cappable(" in src[inert : inert + 2000], "the repair is not triggered by INERT"


def test_the_flag_is_gone_from_the_depth_path_entirely():
    """One implementation, one trigger. An earlier version repaired off the walk's result too, which
    never fired: a run that cannot cap reports INERT and leaves discovery before reaching it."""
    src = (_PA / "cc_optimize" / "run.py").read_text()
    assert "PERF_MCP_REPAIR_MODEL" not in src, "the depth repair is still gated somewhere"
    assert src.count("def make_model_cappable(") == 1, "more than one implementation"


def test_the_repair_retries_with_what_the_tool_measured():
    """ONE SHOT WAS NOT ENOUGH, and the first live attempt shows why.

    Voxtral, 2026-08-12: the agent wrote 40 correct lines -- a _cap_stack helper, both encoder towers
    trimmed, the tail kept so the graduated bodies at 28..31 survive, and it worked out unprompted
    that the two towers are the same stack and must share one depth. It put `layers` on
    VoxtralPipeline.__init__ and not on build_pipeline, which is the only entry point the harness can
    call. The tool said "added nothing" -- correct -- and then gave up, discarding an edit that was
    one signature away from working.

    A retry that says "try again" invites the same edit. The feedback is a PARSED FACT: here are the
    parameters build_pipeline now has, here is what is still missing from that list. An agent told it
    failed repeats itself; an agent told what the signature says fixes the signature.

    Reachability is all this can settle. Whether the knob CAPS is decided by re-measuring the work
    signal, which no edit can talk its way past.
    """
    import tempfile

    from agent.stack_knob_repair import _retry_feedback, _shortfall

    d = Path(tempfile.mkdtemp())
    (d / "tt").mkdir()
    (d / "tt" / "pipeline.py").write_text(
        'PIPELINE_STAGES = ["encode"]\n\n\n'
        "class P:\n    def __init__(self, device, *, layers=None):\n        self.d = layers\n\n\n"
        "def build_pipeline(device, model=None, **kwargs):\n    return None\n"
    )
    still = _shortfall(d, ["layers"])
    assert still == ["layers"], "a knob on the class is not mistaken for a knob on the factory"

    fb = _retry_feedback(d, still)
    assert "device, model" in fb, "the feedback does not state the parameters actually parsed"
    assert "Still missing" in fb, "the feedback does not name what is absent"
    assert "filtered kwargs dict drops them" in fb, "the kwargs trap is not restated"
    assert "Do not redo the work" in fb, "the agent is not told to keep the work it got right"


def test_the_retry_is_bounded():
    """Editing forever is its own failure; the caller re-measures whatever comes out."""
    src = (_PA / "agent" / "stack_knob_repair.py").read_text()
    i = src.index("def repair(model_root")
    body = src[i : i + 3500]
    assert "attempts: int = 3" in body, "the retry is unbounded"
    assert "rounds" in body, "the caller cannot see how many attempts were spent"


def test_a_knob_forwarded_through_a_kwargs_allowlist_counts():
    """**KWARGS ALONE DOES NOT COUNT; **KWARGS PLUS AN ALLOWLIST DOES.

    Voxtral's factory was build_pipeline(device, model=None, **kwargs) filtering to
    {batch_size, prefill_capacity, kv_capacity}, so a `layers` passed by the harness was dropped
    without a word -- the defect the depth clause exists to catch. The repair's natural fix is to add
    the name to that same set, after which build_pipeline(device, layers=2) survives the filter and
    reaches the pipeline.

    Reading only the signature scored that as failure TWICE on 2026-08-12: the agent had made the
    model cappable both times, the tool reported "added nothing", and the re-measure that would have
    proved it never ran. The rule is about what the factory can RECEIVE, not about how it is spelled.
    """
    import tempfile

    from agent.stack_knob_repair import factory_params, missing_knobs

    def mk(known, sig="device, model=None, **kwargs"):
        d = Path(tempfile.mkdtemp())
        (d / "tt").mkdir()
        (d / "tt" / "pipeline.py").write_text(
            'PIPELINE_STAGES = ["encode"]\n\n\ndef build_pipeline(%s):\n    known = {%s}\n    return None\n'
            % (sig, known)
        )
        return d

    forwarded = mk('"batch_size", "layers"')
    assert "layers" in factory_params(forwarded), "an allowlisted kwarg is not seen as receivable"
    assert missing_knobs(forwarded, 1) == [], "a working knob is still reported missing"

    dropped = mk('"batch_size", "kv_capacity"')
    assert "layers" not in factory_params(dropped), "a name absent from the filter must still fail"
    assert missing_knobs(dropped, 1) == ["layers"]

    no_filter = mk('"batch_size"', sig="device, model=None")
    assert missing_knobs(no_filter, 1) == ["layers"], "a factory with no route at all must fail"
