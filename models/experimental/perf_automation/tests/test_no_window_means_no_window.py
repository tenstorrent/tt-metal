"""A coverage probe that returns None is saying "no cap", not "I failed".

THE DEPTH LADDER ALREADY OWNS THIS QUESTION and answers it in order (cc_optimize/run.py):

    signposts        the k=0 probe's per-block markers -> deepest + 1, validated against the
                     model's declared depth, and REJECTED if capping proves inert
    the ladder       2 / 4 / 8 / 16, bounded by full_depth_from_config, each rung rebuilding the
                     model -- the expensive path, run only when signposts cannot answer
    the floor        _cov = 2, "unverified-floor"
    config fallback  the declared layer pattern when the k=0 probe came back empty

None is one of its ANSWERS, and the commonest reason for it is not failure. The signpost path
returns None precisely when it has PROVED the depth knob inert -- the cap left the work signal
unchanged, so the model builds every layer whatever is asked of it:

    [optimize/cc] depth knob is INERT on the signpost path: capping to 48 left the work signal
                  unchanged, so the cap never reached the builder. Profiling FULL depth.

gemma3 is exactly that case: build_pipeline(mesh_device, max_seq_len, batch_size) takes no layer
count, so TT_PERF_LAYERS has nothing to attach to.

before_loop read that None as "the probe failed" and substituted a literal 4 -- a number nothing
derived, contradicting the ladder's own floor of 2 -- then EXPORTED it as TT_PERF_LAYERS and printed

    depth-bridge WARNING: the coverage probe returned None, so the baseline is profiled at a
    SUBSTITUTED depth of 4 layers.

on a run that profiled all 48. The claim was false, the export was false, and only the bridge's
empirical check downstream ("did not reduce work; ignoring") stopped it mattering. A value that
survives solely because something later throws it away should not be produced at all.

PERF_MCP_DEPTH_DEFAULT_LAYERS is removed with it. A second fallback, with a different number, for a
question the ladder already answers is how one path says 4 while the other says 2.
"""

import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))

BL = _PA / "agent" / "before_loop.py"
RUN = _PA / "cc_optimize" / "run.py"


# ---------------------------------------------------------------- the invented default is gone


def test_the_substituted_depth_is_removed():
    assert "PERF_MCP_DEPTH_DEFAULT_LAYERS" not in BL.read_text()


def test_no_module_still_reads_it():
    """It was settable, so anything still consulting it would keep the old behaviour alive."""
    for f in _PA.rglob("*.py"):
        if "tests" in f.parts:
            continue
        assert "PERF_MCP_DEPTH_DEFAULT_LAYERS" not in f.read_text(), f


def test_nothing_is_printed_as_a_substituted_depth():
    """It described a depth the run never used. Checked on the PRINTED strings only -- the comment
    explaining the removal necessarily quotes the message it removed."""
    src = BL.read_text()
    printed = [ln for ln in src.splitlines() if '"' in ln and not ln.lstrip().startswith("#")]
    assert not [ln for ln in printed if "SUBSTITUTED depth" in ln], printed[:3]


# ---------------------------------------------------------------- None exports nothing


def _bridge_block() -> str:
    """The whole depth-bridge block, bounded by what follows it -- not a fixed character count. A
    fixed window silently truncates when the block grows, so the test starts measuring its own size
    instead of the code (the same defect test_warm_start_enforced had with an 8000-byte slice)."""
    src = BL.read_text()
    i = src.index("NO WINDOW MEANS NO WINDOW")
    return src[i : src.index("_bl_depth = _bridge_depth_env(", i)]


def test_no_cap_removes_the_env_var_rather_than_setting_one():
    """layer_depth.set_depth spells "all layers" by ABSENCE, never a sentinel -- "0" arrives as a
    truthy string and is read as "build zero layers". This must use the same spelling."""
    blk = _bridge_block()
    assert 'os.environ.pop("TT_PERF_LAYERS", None)' in blk


def test_the_export_is_conditional_on_a_real_cap():
    """Previously unconditional: `os.environ["TT_PERF_LAYERS"] = str(_bl_cov)` ran even when _bl_cov
    was the invented 4."""
    src = BL.read_text()
    # The export now writes a SCALAR: _bl_cov is a per-stack dict, and str() of it wrote
    # "{'stack3': 2, 'stack2': 2}" into the one variable a perf test parses -- which fails
    # .isdigit(), yields None, and means ALL LAYERS. The baseline was therefore measured uncapped
    # while every candidate after it ran capped. What this test pins is unchanged: the export happens
    # only when there is a real cap to export.
    i = src.index('os.environ["TT_PERF_LAYERS"] = str(_bl_scalar)')
    assert "if _bl_cov:" in src[max(0, i - 1200) : i], "the export is no longer guarded by a real cap"


def test_the_bridge_search_is_skipped_when_there_is_no_cap():
    """_bridge_depth_env hunts for an env spelling that makes a cap REACH the builder. With no cap
    there is nothing to search for, and each probe rebuilds the model."""
    src = BL.read_text()
    i = src.index("_bl_depth = _bridge_depth_env(")
    assert "if _bl_cov:" in src[max(0, i - 400) : i]


def test_the_message_states_full_depth_and_says_why():
    """The reader needs to know the run is NOT capped, and that this is a property of the model --
    not a probe that fell over."""
    blk = _bridge_block()
    assert "FULL depth" in blk
    assert "does not reach this" in blk and "builder" in blk


# ---------------------------------------------------------------- the ladder is untouched


def test_the_ladder_still_owns_the_derivation():
    """Signposts first (free -- the k=0 probe already emitted them), then the rung search (each rung
    rebuilds the model), then the floor. Nothing here replaces any of it."""
    src = RUN.read_text()
    assert "SIGNPOSTS BEFORE THE LADDER" in src
    assert "_cov_ladder" in src
    i = src.index("def _cov_ladder")
    assert "BOUNDED BY THE MODEL'S DECLARED DEPTH" in src[i : i + 300]


def test_the_ladder_floor_is_two():
    """The ONE fallback for this question. before_loop's competing 4 is what this change removes.

    Matched on the VALUE assigned beside the unverified-floor marker, not on a variable name: this
    pinned the literal `_cov = 2` and broke the day an upstream merge renamed it to `_cov_scalar`,
    which is a rename, not a behaviour change. The floor being 2 is the decision worth pinning."""
    import re

    src = RUN.read_text()
    i = src.index('blk_source = "unverified-floor"')
    window = src[max(0, i - 200) : i]
    assert re.search(r"^\s*_cov\w*\s*=\s*2\s*$", window, re.M), window


# ---------------------------------------------------------------- the reason is STATED, not inferred


def test_every_no_window_exit_states_why():
    """None is the answer for three unrelated situations. A caller handed a bare None cannot tell
    "profile everything, deliberately" from "something broke" -- and inferring it is exactly how a
    deliberate None became an invented depth of 4."""
    src = RUN.read_text()
    i = src.index("def _coverage_layers")
    body = src[i : src.index("\ndef ", i + 1)]
    # Two now, not three: the signpost path's own inert check was removed -- the bridge already
    # measures the caps it applies, and this copy probed at max(per-stack depths), which asks for
    # FULL depth on a model whose deepest stack is the model. What matters is unchanged and is
    # asserted below: EVERY None exit states its reason.
    none_exits = body.count("return None, facts")
    assert none_exits >= 2, none_exits
    assert (
        body.count('facts["no_window"]') == none_exits
    ), "a None exit that states no reason puts the caller back to guessing"


def test_the_signpost_path_does_not_re_verify_what_the_bridge_measures():
    """ONE DEPTH DECISION, MEASURED ONCE, WHERE IT IS APPLIED.

    The signpost path used to run its own inert check before returning a window, probing at
    max(per-stack depths). On a model whose deepest stack IS the model -- Voxtral: audio encoder 32
    of 32 -- that asks for FULL depth, so the work signal cannot move, and the knob was declared dead.
    Measured 2026-08-13: a correct window (stack0=2, stack2=32, stack3=3) was discarded and the run
    refused, on a model whose knobs were wired and working.

    The bridge already covers the case it was guarding: it applies the caps, measures, and on no
    reduction prints "did not reduce work ... ignoring" and enforces nothing -- which is the gemma3
    protection, done against the depths that will actually be used.
    """
    src = RUN.read_text()
    assert "_signpost_cap_is_inert" not in src, "the duplicate verification is back"
    assert "did not reduce work" in src, "the bridge's own measurement is gone too"


def test_a_genuine_failure_is_named_distinctly():
    src = RUN.read_text()
    assert 'facts["no_window"] = "probe_failed"' in src


def test_the_caller_reads_the_reason_rather_than_assuming():
    blk = _bridge_block()
    assert '(_bl_facts or {}).get("no_window")' in blk
    for reason in ("knob_inert", "probe_failed", "sizing_disabled", "no_node"):
        assert reason in blk, reason


def test_an_unreported_reason_says_so_instead_of_inventing_one():
    """The failure mode this whole change is about: an unknown must surface as unknown."""
    blk = _bridge_block()
    assert "reason not reported" in blk


def test_a_failed_probe_is_not_dressed_up_as_a_decision():
    """Both cases profile full depth, but only one is fine. The message must distinguish them."""
    blk = _bridge_block()
    i = blk.index('"probe_failed"')
    assert "NOT a decision" in blk[i : i + 300]
