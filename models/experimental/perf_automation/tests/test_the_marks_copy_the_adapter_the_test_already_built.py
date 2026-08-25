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
_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"


def _try_traced():
    _traced_forward()
    return True


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

    _PROFILING = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"
    if _PERF_TRACE and not _PROFILING:
        if not _try_traced():
            %(eager)s()
    else:
        %(eager)s()
        if _PERF_TRACE:
            _try_traced()
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


# --- the branch the profiler actually reaches ----------------------------------------------------


def test_it_skips_a_call_that_is_dead_under_profiling():
    """THE REGRESSION. The rule was "the last bare call is the profiled branch". In the real
    generated test the last one is `_try_traced()` inside `if _PERF_TRACE:` -- false under profiling,
    because the tracy subprocess is given TT_PERF_TRACE=0. Both the bracket and the per-stage pass
    landed there, so the capture came back with zero signposts AND zero diagnostics: silence from a
    block that never ran."""
    from agent.stage_marks import reachable_bare_calls

    got = [n for _, _, n in reachable_bare_calls(_src())]
    assert got == ["_eager_forward"], got


def test_the_marks_follow_that_decision():
    out, why = inject_stage_marks(_src())
    assert "per-stage pass in _eager_forward()" in why, why
    lines = out.splitlines()
    br = next(i for i, l in enumerate(lines) if 'signpost("start")' in l)
    assert "_eager_forward()" in lines[br + 1], "the bracket wrapped a call the profiler never makes"


def test_the_environment_comes_from_the_run_not_from_a_copy_here():
    """Two consumers had their own copies of TT_METAL_DEVICE_PROFILER=1 / TT_PERF_TRACE=0. The one
    that SETS them owns them; this reads that definition."""
    from agent.probes import PROFILING_ENV
    from agent.stage_marks import _profiling_env

    assert _profiling_env() == dict(PROFILING_ENV)
    assert PROFILING_ENV["TT_METAL_DEVICE_PROFILER"] == "1"
    assert PROFILING_ENV["TT_PERF_TRACE"] == "0"


def test_an_undecidable_condition_keeps_both_branches():
    """A call that MIGHT run beats one that certainly does not."""
    from agent.stage_marks import reachable_bare_calls

    src = _src().replace('_PERF_TRACE and not _PROFILING', 'some_unknown_flag')
    got = [n for _, _, n in reachable_bare_calls(src)]
    # Both arms of the undecidable branch are kept -- and _try_traced STILL is not, because its own
    # condition (`if _PERF_TRACE:`) remains decidable and false. Undecidable widens the search; it
    # does not switch it off.
    assert got == ["_eager_forward", "_eager_forward"], got


def test_a_module_level_flag_is_carried_into_the_function():
    """_PERF_TRACE is assigned at module scope and read inside the test function. Binding top-level
    statements into a throwaway dict left the condition undecidable and both branches reachable."""
    from agent.stage_marks import reachable_bare_calls

    assert [n for _, _, n in reachable_bare_calls(_src())] == ["_eager_forward"]


# --- the stage inputs, without asking anyone for a data file -------------------------------------


_WITH_PREPARER = '''import os

PERF_BATCH = 8
_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"


def %(prep)s(pipe):
    pipe.encode%(hook)s = lambda: 1
    pipe.decode%(hook)s = lambda: 2


def test_main_perf(device_params, device):
    def _eager_forward():
        pipe = build_pipeline(device)
        return pipe.run()

    def _traced_forward():
        p = build_pipeline(device)
        %(prep)s(p)
        measure_adapter(PipelineStageAdapter(_b, _ids, batch=PERF_BATCH), device)

    _PROFILING = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"
    if _PERF_TRACE and not _PROFILING:
        _traced_forward()
    else:
        _eager_forward()
'''


def _with_preparer(prep="_bind_stage_inputs", hook="_trace_inputs"):
    return _WITH_PREPARER % {"prep": prep, "hook": hook}


def test_the_preparer_is_found_by_what_it_does_not_by_its_name():
    """A pipeline's <stage>_trace_inputs() reads captured golden tensors the bring-up wrote. A model
    being optimised for the first time has none, and nobody should hand one over. The generated test
    already solves that for the timing path by pointing those hooks at a real batch it builds from
    the demo -- so the marks reuse it. Found by the hook suffix, which is perf_adapter's contract."""
    from agent.stage_marks import find_input_preparer

    assert find_input_preparer(_with_preparer()) == "_bind_stage_inputs"
    assert find_input_preparer(_with_preparer(prep="_wire_up")) == "_wire_up"


def test_a_test_with_no_preparer_gets_no_bind_argument():
    """Correct for a pipeline whose hooks need no preparation -- not every model captures tensors."""
    from agent.stage_marks import find_input_preparer

    src = _with_preparer().replace("_trace_inputs", "_something_else")
    assert find_input_preparer(src) == ""
    out, _ = inject_stage_marks(src)
    line = next(l for l in out.splitlines() if "mark_stages_in_scope" in l)
    assert "bind=" not in line, line


def test_the_marks_are_handed_the_preparer():
    out, why = inject_stage_marks(_with_preparer())
    assert "per-stage pass in _eager_forward()" in why, why
    ast.parse(out)
    line = next(l for l in out.splitlines() if "mark_stages_in_scope" in l)
    assert "bind=_bind_stage_inputs" in line, line


def test_the_pipeline_is_prepared_before_it_is_marked(monkeypatch):
    """The regression: the marks took the pipeline out of the EAGER scope, where the test has not
    pointed its hooks at real inputs yet -- so encode fell back to torch.load of a _captured file
    that does not exist on a pristine tree, and the whole pass died with FileNotFoundError."""
    import sys as _sys
    import types as _types

    stub = _types.ModuleType("ttnn")
    stub.synchronize_device = lambda d: None
    monkeypatch.setitem(_sys.modules, "ttnn", stub)
    from agent import stage_marks as sm

    seen = {}

    class _P:
        PIPELINE_STAGES = ["decode"]

        def decode_trace_step(self):
            return None

    pipe = _P()

    def _prep(p):
        seen["prepared"] = p is pipe

    monkeypatch.setattr(sm, "signpost", lambda n: None)
    sm.mark_stages_in_scope({"pipe": pipe}, device=object(), bind=_prep)
    assert seen.get("prepared") is True, "the marks ran against an unprepared pipeline"


def test_a_preparer_that_raises_does_not_stop_the_marks(monkeypatch, capsys):
    """Its own hooks may still work, and one stage that cannot prepare no longer costs the others."""
    import sys as _sys
    import types as _types

    stub = _types.ModuleType("ttnn")
    stub.synchronize_device = lambda d: None
    monkeypatch.setitem(_sys.modules, "ttnn", stub)
    from agent import stage_marks as sm

    class _P:
        PIPELINE_STAGES = ["decode"]

        def decode_trace_step(self):
            return None

    monkeypatch.setattr(sm, "signpost", lambda n: None)

    def _boom(_p):
        raise RuntimeError("no clips")

    sm.mark_stages_in_scope({"pipe": _P()}, device=object(), bind=_boom)
    assert "falling back to the pipeline's own hooks" in capsys.readouterr().err


def test_one_stage_that_cannot_prepare_does_not_cost_the_others():
    """perf_adapter's setup(_tin()) was unguarded: voxtral's encode torch.loads a captured tensor,
    and on a tree without one it took prefill and decode down with it -- both of which derive their
    own inputs and needed nothing from disk."""
    from pathlib import Path as _Path

    src = (_Path(__file__).resolve().parents[1] / "agent" / "perf_adapter.py").read_text()
    i = src.index("_tin = getattr(p,")
    window = src[i : i + 2600]
    assert "except Exception" in window and "continue" in window, "a failing stage is still fatal"
    assert "the others are unaffected" in window


# --- and the preparer must be REACHABLE from where the marks run ---------------------------------


_NESTED_PREPARER = '''import os

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"


def test_main_perf(device_params, device):
    def _eager_forward():
        pipe = build_pipeline(device)
        return pipe.run()

    def _traced_forward():
        def _build_for_perf(dev):
            p = build_pipeline(dev)
            p.encode_trace_inputs = lambda: 1
            return p

        measure_adapter(PipelineStageAdapter(_build_for_perf, _ids), device)

    _PROFILING = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"
    if _PERF_TRACE and not _PROFILING:
        _traced_forward()
    else:
        _eager_forward()
'''


def test_a_preparer_nested_in_another_function_is_not_used():
    """THE REGRESSION, reported by the run itself:
    STAGE_MARKS_SKIPPED=NameError("name '_build_for_perf' is not defined").

    The preparer is found by what it DOES, which is right, and the first version stopped there -- it
    matched a function that genuinely assigns the stage-input hooks but was a LOCAL of the traced
    path, invisible from the eager function where the marks run. Same scope mistake as copying the
    adapter's arguments across branches, in a new place."""
    from agent.stage_marks import find_input_preparer

    assert find_input_preparer(_NESTED_PREPARER, 0) == ""
    out, why = inject_stage_marks(_NESTED_PREPARER)
    assert "injected" in why, why
    ast.parse(out)
    line = next(l for l in out.splitlines() if "mark_stages_in_scope" in l)
    assert "bind=" not in line, "an out-of-scope preparer was referenced: %s" % line.strip()


def test_a_module_level_preparer_is_still_used():
    """Reachable from anywhere in the file, so it is exactly what the marks should use."""
    from agent.stage_marks import find_input_preparer

    src = _with_preparer(prep="_prep_anything")
    assert find_input_preparer(src, 0) == "_prep_anything"
    out, _ = inject_stage_marks(src)
    line = next(l for l in out.splitlines() if "mark_stages_in_scope" in l)
    assert "bind=_prep_anything" in line, line


def test_a_preparer_in_an_enclosing_function_is_visible():
    """Defined in the test function itself, so the pass -- which lands in a function nested inside
    it -- can see it. Rejecting this would throw away a usable preparer."""
    from agent.stage_marks import find_input_preparer, _function_body_end

    src = (
        "import os\n"
        '_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"\n'
        "\n\n"
        "def test_x_perf(device):\n"
        "    def _prep(pipe):\n"
        "        pipe.decode_trace_inputs = lambda: 1\n"
        "\n"
        "    def _eager_forward():\n"
        "        pipe = build_pipeline(device)\n"
        "        return pipe\n"
        "\n"
        '    _PROFILING = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"\n'
        "    if _PERF_TRACE and not _PROFILING:\n"
        "        pass\n"
        "    else:\n"
        "        _eager_forward()\n"
    )
    end, _ = _function_body_end(src, "_eager_forward")
    assert find_input_preparer(src, end) == "_prep"
    out, _ = inject_stage_marks(src)
    line = next(l for l in out.splitlines() if "mark_stages_in_scope" in l)
    assert "bind=_prep" in line, line


def test_a_preparer_with_extra_arguments_is_still_used():
    """run 35, exactly: a module-level preparer doing the right thing, skipped for its signature.

        def _patch_trace_inputs(pipe, batch):     <- two arguments
            ...
            pipe.encode_trace_inputs = lambda: mel

    The rule was "it takes the pipeline, and only that". One generated test wrote one parameter and
    worked; the next wrote two -- with a docstring explaining it exists precisely because the
    _captured tensors are not shipped -- and was skipped. Every stage then fell back to those missing
    files, all three were dropped, and the run marked nothing.

    The rule contained no name, so every scan for hardcoded identifiers passed it. It was an
    assumption about SHAPE, which is the thing the generator varies."""
    from agent.stage_marks import find_input_preparer

    src = (
        "import os\n"
        '_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"\n'
        "\n\n"
        "def _patch_trace_inputs(pipe, batch):\n"
        "    pipe.encode_trace_inputs = lambda: batch.mel\n"
        "    pipe.decode_trace_inputs = lambda: batch.ids\n"
        "\n\n"
        "def test_x_perf(device):\n"
        "    def _eager_forward():\n"
        "        pipe = build_pipeline(device)\n"
        "        return pipe\n"
        "\n"
        '    _PROFILING = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"\n'
        "    if _PERF_TRACE and not _PROFILING:\n"
        "        pass\n"
        "    else:\n"
        "        _eager_forward()\n"
    )
    assert find_input_preparer(src, 0) == "_patch_trace_inputs"
    out, why = inject_stage_marks(src)
    ast.parse(out)
    line = next(l for l in out.splitlines() if "mark_stages_in_scope" in l)
    assert "bind=_patch_trace_inputs" in line, line


def test_a_preparer_with_many_arguments_is_still_used():
    """Nothing about the count matters -- only that the function does the work and is reachable."""
    from agent.stage_marks import find_input_preparer

    src = (
        "import os\n"
        '_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"\n'
        "\n\n"
        "def prep(pipe, batch, head, layers=None):\n"
        "    pipe.prefill_trace_inputs = lambda: batch\n"
        "\n\n"
        "def test_x_perf(device):\n"
        "    def _eager_forward():\n"
        "        pipe = build_pipeline(device)\n"
        "        return pipe\n"
        "\n"
        '    _PROFILING = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"\n'
        "    if _PERF_TRACE and not _PROFILING:\n"
        "        pass\n"
        "    else:\n"
        "        _eager_forward()\n"
    )
    assert find_input_preparer(src, 0) == "prep"
    out, _ = inject_stage_marks(src)
    assert "bind=prep" in next(l for l in out.splitlines() if "mark_stages_in_scope" in l)
