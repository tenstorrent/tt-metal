"""The injector required six identifiers the generator was free not to use.

The block that emits the per-stage marks needs a PipelineStageAdapter. It used to REBUILD one from
named helpers -- _stage_inputs_from_demo, _get_pipe, prompt_ids_for_isl, get_tokenizer,
PERF_ISL_TOKENS, PERF_BATCH -- and refuse when the test did not define them. But the test is written
by an LLM from a skeleton that is explicitly advisory, so those names are its choice, not a contract.

Run 27, the first run with the model reset to pristine and the perf test therefore regenerated: the
LLM named the same things `_build_for_perf` and `_prompt_ids`, the check failed on the two private
helpers, and the marks were refused -- so the roofline shared one math-fidelity peak across every
stack for a tenth consecutive run.

Swapping in _build_for_perf would only move the hardcoded name. The test ALREADY constructs the
adapter, to hand it to measure_adapter, so the block copies that call's arguments verbatim. The one
fixed token left is PipelineStageAdapter -- the tool's own API, the same kind of anchor as
ttnn.execute_trace -- and nothing about how a generated test spells its own helpers."""
import ast

import pytest

from agent.stage_marks import _adapter_args, inject_stage_marks

_SKELETON = '''import os

PERF_BATCH = 8


def %(build)s(dev):
    return object()


def test_x_perf(device):
    %(ids)s = [1, 2, 3]
    from models.experimental.perf_automation.agent.perf_adapter import PipelineStageAdapter
    from models.experimental.perf_automation.agent.trace_replay import measure_adapter

    measure_adapter(PipelineStageAdapter(%(build)s, %(ids)s, batch=PERF_BATCH), device)
    _eager_forward()
'''


def _src(build="_build_for_perf", ids="_prompt_ids"):
    body = _SKELETON % {"build": build, "ids": ids}
    return body.replace("    _eager_forward()\n", "    _eager_forward()\n") + "\ndef _eager_forward():\n    pass\n"


def test_it_copies_whatever_the_generator_named_things():
    """The regression: the run-27 test used _build_for_perf / _prompt_ids and was refused."""
    src = _src()
    assert _adapter_args(src) == "_build_for_perf, _prompt_ids, batch=PERF_BATCH"
    out, why = inject_stage_marks(src)
    assert "injected" in why, why
    ast.parse(out)
    assert "_TtPSA(_build_for_perf, _prompt_ids, batch=PERF_BATCH)" in out


def test_a_different_naming_works_just_as_well():
    """Nothing may depend on the spelling -- that is the whole defect."""
    src = _src(build="make_pipe", ids="ids_for_this_run")
    out, why = inject_stage_marks(src)
    assert "injected" in why
    ast.parse(out)
    assert "_TtPSA(make_pipe, ids_for_this_run, batch=PERF_BATCH)" in out


def test_the_old_skeleton_names_still_work():
    """A test that DOES use the skeleton's names must not regress."""
    src = _src(build="_get_pipe", ids="_stage_inputs_from_demo")
    out, why = inject_stage_marks(src)
    assert "injected" in why
    ast.parse(out)


def test_a_multiline_call_is_copied_whole():
    """The call spans lines and its arguments contain commas and parens: a scan that stops at the
    first ')' or splits on ',' would truncate it into code that does not parse."""
    src = _src().replace(
        "PipelineStageAdapter(_build_for_perf, _prompt_ids, batch=PERF_BATCH)",
        "PipelineStageAdapter(\n            _build_for_perf,\n            _prompt_ids[0:3],\n            batch=int(PERF_BATCH),\n        )",
    )
    assert _adapter_args(src) == "_build_for_perf, _prompt_ids[0:3], batch=int(PERF_BATCH)"
    out, why = inject_stage_marks(src)
    assert "injected" in why
    ast.parse(out)


def test_a_test_with_no_adapter_call_is_refused_with_a_reason():
    src = _src().replace("PipelineStageAdapter(_build_for_perf, _prompt_ids, batch=PERF_BATCH)", "None")
    out, why = inject_stage_marks(src)
    assert out == src
    assert "no PipelineStageAdapter" in why


def test_a_file_that_will_not_parse_is_refused_rather_than_half_copied():
    """A balanced-paren text walk read PAST an unclosed call and returned
    `_build_for_perf, _prompt_ids, batch=PERF_BATCH, device` by closing on the enclosing
    measure_adapter( -- code that compiles and measures the wrong thing. Parsing refuses instead."""
    src = _src().replace("batch=PERF_BATCH)", "batch=PERF_BATCH")
    assert _adapter_args(src) == ""
    out, why = inject_stage_marks(src)
    assert out == src and "no PipelineStageAdapter" in why


def test_nothing_is_required_by_name_any_more():
    """The six-name gate is gone; only the tool's own API remains as an anchor."""
    import agent.stage_marks as sm

    assert not hasattr(sm, "_REQUIRED_NAMES"), "the name gate is back"
    assert sm._ADAPTER_CALL == "PipelineStageAdapter("
