"""Every perf test the generator has actually written, as a corpus.

WHY THIS FILE EXISTS. Each fix to the injector was verified against whichever single generated test
happened to be on disk that hour, and each one then broke on the next run -- because the generator is
an LLM that writes a structurally different file every time. Verified against a sample of one, from a
population that varies, is not verified at all.

What the three saved files actually do, which no amount of reasoning would have predicted:

    run33   _build_for_perf(pipe)          one argument, NESTED inside the traced path
    run34   no preparer function at all    the hooks are arranged some other way
    run35   _patch_trace_inputs(pipe, batch)   TWO arguments, module level

Each of those broke a rule I had written into the injector and believed was safe: "the preparer takes
exactly one argument", "the last bare call is the profiled branch", "the marks go at the end of the
body". None of them contains a model name, a stage name or a generated identifier, so every scan for
hardcoded NAMES passed them -- the assumptions were about SHAPE, and shape is what varies.

So the rule is: a change to the injector is not verified until it passes every file here, and every
new failure in the field adds its file. Drop the offending test in as <run>.py.txt and it is covered
from then on."""
import ast
from pathlib import Path

import pytest

from agent.stage_marks import inject_stage_marks, reachable_bare_calls

_CORPUS = sorted((Path(__file__).resolve().parent / "generated_perf_tests").glob("*.py.txt"))


def _any_function_assigning_stage_hooks(src: str, module_level_only: bool = False) -> str:
    """Any function that points a <stage>_trace_inputs hook somewhere, whatever its signature.

    Deliberately does not call into agent.stage_marks: a test that asks the code under test whether
    there is anything to test cannot fail when that code is wrong."""
    tree = ast.parse(src)
    for n in ast.walk(tree):
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if module_level_only and n.col_offset != 0:
            continue
        # Its OWN body, not ast.walk: descending into nested definitions made the enclosing test
        # function look like the preparer, because a helper inside it does the assigning. The
        # production code has the same rule; this reimplements it so the test cannot inherit its bug.
        stack, owns = list(n.body), False
        while stack and not owns:
            st = stack.pop()
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(st, ast.Assign) and any(
                isinstance(t, ast.Attribute) and t.attr.endswith("_trace_inputs") for t in st.targets
            ):
                owns = True
            stack.extend(c for c in ast.iter_child_nodes(st) if isinstance(c, (ast.stmt, ast.excepthandler)))
        if owns:
            return n.name
    return ""


def _fn_holding_the_pass(src: str) -> str:
    """Which function the per-stage pass ended up in, read from the file rather than the message --
    a file saved after injection has no message to read."""
    line = next(i for i, l in enumerate(src.splitlines(), 1) if "mark_stages_in_scope" in l)
    tree, best = ast.parse(src), None
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.lineno <= line <= (n.end_lineno or n.lineno):
            if best is None or n.lineno > best.lineno:
                best = n
    return best.name if best else ""


def _ids():
    return [p.stem.replace(".py", "") for p in _CORPUS]


def test_the_corpus_is_not_empty():
    """A corpus that quietly became empty would make every test below vacuously pass."""
    assert _CORPUS, "no generated perf tests saved -- the corpus cannot protect anything"
    assert len(_CORPUS) >= 3, "expected at least the three runs that each broke a different rule"


def _marked(path):
    """(source with marks, why). A file saved before injection is injected here; one saved after is
    used as it stands -- stripping an injection back out mangled two of these files and produced
    corpus failures that looked like injector bugs, so the stored bytes are trusted as they came."""
    src = path.read_text()
    if "_tt_sm" in src:
        return src, "already injected"
    return inject_stage_marks(src)


@pytest.mark.parametrize("path", _CORPUS, ids=_ids())
def test_each_real_test_parses(path):
    ast.parse(path.read_text())


@pytest.mark.parametrize("path", _CORPUS, ids=_ids())
def test_injection_succeeds_and_still_parses(path):
    """The floor: whatever the generator wrote, the marks go in and the file still runs."""
    out, why = _marked(path)
    assert "injected" in why, "%s: %s" % (path.name, why)
    ast.parse(out)


@pytest.mark.parametrize("path", _CORPUS, ids=_ids())
def test_injection_is_idempotent(path):
    once, _ = _marked(path)
    twice, why = inject_stage_marks(once)
    assert twice == once and why == "already injected", path.name


@pytest.mark.parametrize("path", _CORPUS, ids=_ids())
def test_the_marks_land_on_a_branch_the_profiler_reaches(path):
    """run 29 put both the bracket and the pass inside `if _PERF_TRACE:`, which is false under
    profiling, so the capture came back with no signposts AND no diagnostics."""
    out, _ = _marked(path)
    reached = {n for _, _, n in reachable_bare_calls(out)}
    assert reached, "%s: no bare call is reachable under the profiling environment" % path.name
    fn = _fn_holding_the_pass(out)
    assert fn in reached, "%s: pass sits in %s, which the profiler does not reach" % (path.name, fn)


@pytest.mark.parametrize("path", _CORPUS, ids=_ids())
def test_the_pass_is_reachable_within_its_function(path):
    """Appending after a `return` is valid Python that never runs."""
    out, _ = _marked(path)
    fn = _fn_holding_the_pass(out)
    tree = ast.parse(out)
    target = next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == fn)
    call = next(i for i, l in enumerate(out.splitlines(), 1) if "mark_stages_in_scope" in l and target.lineno <= i)
    for st in target.body:
        if isinstance(st, ast.Return):
            assert call < st.lineno, "%s: the pass sits after `return` and can never run" % path.name
            break


@pytest.mark.parametrize("path", _CORPUS, ids=_ids())
def test_any_preparer_it_names_is_actually_in_scope(path):
    """run 33: STAGE_MARKS_SKIPPED=NameError("name '_build_for_perf' is not defined") -- the preparer
    was real, and a local of a function the marks do not run in."""
    out, _ = _marked(path)
    line = next((l for l in out.splitlines() if "mark_stages_in_scope" in l), "")
    if "bind=" not in line:
        return  # naming none is always safe
    name = line.split("bind=")[1].split(")")[0].split(",")[0].strip()
    fn = _fn_holding_the_pass(out)
    tree = ast.parse(out)
    target = next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == fn)
    visible = {
        n.name
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and (
            n.col_offset == 0
            or (n.lineno < target.lineno and (n.end_lineno or n.lineno) >= (target.end_lineno or target.lineno))
        )
    }
    assert name in visible, "%s: bind=%s is not visible from %s()" % (path.name, name, fn)


@pytest.mark.parametrize("path", _CORPUS, ids=_ids())
def test_a_preparer_is_used_when_one_exists(path):
    """run 35: the preparer was module level and did exactly the right thing, and was skipped for
    taking two arguments instead of one. A rule about SHAPE, which is what varies."""
    src = path.read_text()
    if "_tt_sm" in src:
        pytest.skip("saved after injection: its bind reflects the code of that day, not of today")
    # DETECTED INDEPENDENTLY, not by asking the function under test. Using find_input_preparer to
    # decide whether to check find_input_preparer is circular -- and it passed run 35 for exactly that
    # reason: the finder rejected a two-argument preparer, reported none, and the test skipped itself.
    #
    # MODULE LEVEL only: a preparer nested inside another function is legitimately unusable from the
    # scope the marks run in, which is what run 33 showed. Visibility is the code's judgement to make;
    # a module-level definition is visible to everything, so there is no judgement to defer to.
    prep = _any_function_assigning_stage_hooks(src, module_level_only=True)
    if not prep:
        return  # nothing unambiguously reachable is a legitimate answer
    out, _ = _marked(path)
    line = next(l for l in out.splitlines() if "mark_stages_in_scope" in l)
    assert "bind=" in line, "%s: %s prepares the stage inputs and was not used" % (path.name, prep)
