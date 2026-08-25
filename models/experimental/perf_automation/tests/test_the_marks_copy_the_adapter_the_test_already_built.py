"""The marks run where the pipeline lives, and find it by shape.

Three versions of this injection failed, each on the same mistaken assumption -- that the block can
name what it needs:

  1. it required six identifiers (_stage_inputs_from_demo, _get_pipe, prompt_ids_for_isl,
     get_tokenizer, PERF_ISL_TOKENS, PERF_BATCH) and refused a regenerated test that had called the
     same things _build_for_perf and _prompt_ids;
  2. it copied the arguments of the test's own PipelineStageAdapter(...) call, which parsed fine and
     then raised NameError("name '_build_for_perf' is not defined") on device -- the generator had
     defined those inside a nested function, and text lifted out of one scope cannot run in another;
  3. appending the pass to the end of the function body put it under `return out`: syntactically
     valid, never executed.

What the pass actually needs is the built pipeline, and that is a LOCAL of the function the bracket
wraps. So it is injected into that function, before its return, and handed locals(). It picks the
pipeline out by the surface the adapter drives -- PIPELINE_STAGES, or the <stage>_trace_step hooks --
so nothing depends on how a generated test spells anything."""
import ast
import sys
import types

import pytest

from agent.stage_marks import find_pipeline_in_scope, inject_stage_marks

_SHAPE = '''import os

PERF_BATCH = 8


def _build_kwargs():
    return {}


def test_main_perf(device_params, device):
    def %(eager)s():
        %(pipe)s = build_pipeline(device, **_build_kwargs())
        out = %(pipe)s.run()
        return out

    def _traced_forward():
        def %(build)s(dev):
            return build_pipeline(dev, **_build_kwargs())

        %(ids)s = [1, 2, 3]
        measure_adapter(PipelineStageAdapter(%(build)s, %(ids)s, batch=PERF_BATCH), device)

    if not os.environ.get("TT_METAL_DEVICE_PROFILER") == "1":
        _traced_forward()
    else:
        %(eager)s()
'''


def _src(eager="_eager_forward", pipe="pipe", build="_build_for_perf", ids="_prompt_ids"):
    return _SHAPE % {"eager": eager, "pipe": pipe, "build": build, "ids": ids}


def _fn_of(text, line):
    """Which function encloses this 1-based line."""
    tree = ast.parse(text)
    best = None
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.lineno <= line <= (n.end_lineno or n.lineno):
            if best is None or n.lineno > best.lineno:
                best = n
    return best.name if best else None


def test_the_pass_lands_inside_the_function_that_built_the_pipeline():
    """Not in the profiling branch, where the builder is out of scope -- that raised NameError."""
    out, why = inject_stage_marks(_src())
    assert "per-stage pass in _eager_forward()" in why, why
    ast.parse(out)
    line = next(i for i, l in enumerate(out.splitlines(), 1) if "mark_stages_in_scope" in l)
    assert _fn_of(out, line) == "_eager_forward"


def test_the_pass_is_reachable_and_not_under_the_return():
    """Appending to the end of the body put it below `return out`: valid, and never executed."""
    out, _ = inject_stage_marks(_src())
    lines = out.splitlines()
    call = next(i for i, l in enumerate(lines) if "mark_stages_in_scope" in l)
    ret = next(i for i, l in enumerate(lines) if l.strip() == "return out")
    assert call < ret, "the per-stage pass sits after the return and can never run"


def test_it_does_not_care_what_anything_is_named():
    """The whole defect: every earlier version depended on the generator's spelling."""
    out, why = inject_stage_marks(_src(eager="_run_it", pipe="model", build="mk", ids="ids"))
    assert "per-stage pass in _run_it()" in why, why
    ast.parse(out)
    line = next(i for i, l in enumerate(out.splitlines(), 1) if "mark_stages_in_scope" in l)
    assert _fn_of(out, line) == "_run_it"


def test_it_is_idempotent():
    out, _ = inject_stage_marks(_src())
    again, why = inject_stage_marks(out)
    assert again == out and why == "already injected"


def test_the_last_bare_call_is_taken_as_the_profiled_branch():
    """The pre-existing rule, kept and now asserted: an earlier bare call is the trace-replay
    fallback, the last is the branch the profiler runs."""
    out, why = inject_stage_marks(_src())
    assert "per-stage pass in _eager_forward()" in why
    lines = out.splitlines()
    br = next(i for i, l in enumerate(lines) if 'signpost("start")' in l)
    assert "_eager_forward()" in lines[br + 1], "the bracket wrapped the wrong call"


def test_a_test_with_nothing_to_wrap_is_refused_with_a_reason():
    out, why = inject_stage_marks("def test_x():\n    pass\n")
    assert out == "def test_x():\n    pass\n"
    assert "no bare call to an eager pass" in why


# --- and the runtime half: finding the pipeline among locals ------------------------------------


class _WithAttr:
    PIPELINE_STAGES = ["encode", "decode"]


def _mod_backed():
    """A pipeline whose PIPELINE_STAGES lives on its MODULE, with the per-stage hooks on the object --
    the other shape perf_adapter accepts."""
    mod = types.ModuleType("fake_pipe_mod")
    mod.PIPELINE_STAGES = ["decode"]
    sys.modules["fake_pipe_mod"] = mod
    cls = type("P", (), {"decode_trace_step": lambda self: None})
    cls.__module__ = "fake_pipe_mod"
    return cls()


def test_it_finds_the_pipeline_by_its_stage_surface():
    scope = {"x": 1, "s": "str", "pipe": _WithAttr(), "device": object()}
    assert isinstance(find_pipeline_in_scope(scope), _WithAttr)


def test_it_finds_the_module_backed_shape_too():
    obj = _mod_backed()
    assert find_pipeline_in_scope({"whatever": obj}) is obj


def test_nothing_in_scope_qualifies_returns_none():
    assert find_pipeline_in_scope({"a": 1, "b": "two", "c": object()}) is None


def test_an_empty_scope_is_not_an_error():
    assert find_pipeline_in_scope({}) is None
    assert find_pipeline_in_scope(None) is None


def test_a_zero_result_says_why(capsys):
    """A silent 0 is what let nine earlier failures look identical from the outside."""
    from agent import stage_marks as sm

    assert sm.mark_stages_in_scope({"a": 1}, device=object()) == 0
    err = capsys.readouterr().err
    assert "NO per-stage boundaries" in err and "no object in scope exposes" in err
