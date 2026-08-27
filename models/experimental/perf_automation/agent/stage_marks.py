# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage boundaries for the TRACY run, which is the only run that records per-op data.

THE TOOL MEASURES TWICE, FOR TWO DIFFERENT THINGS:

    A  tracy eager profile     PROFILER=1, TT_PERF_TRACE=0, coverage depth (2 layers)
       -> per-op records: op_code, shape, FIDELITY, cores, grid, memory, bytes, device_ms
       -> the slice is sized to hold every distinct op, so op PROPERTIES are authoritative here

    B  full-depth stopwatch    profiler popped, all layers, trace+1cq
       -> TRACE_STAGE_MS / TRACE_STAGE_BYTES: times and totals. No per-op anything.

A traced replay runs as one fused program and emits NO per-op device data, which is why A is eager
and why fidelity exists ONLY in A. Five earlier attempts at per-stage fidelity were written into
trace_replay -- B's machinery -- which never produces a fidelity field at all.

WHAT A CANNOT DO BY ITSELF. Its measured call is `pipe.run_head(...)`: encode, prefill and every
decode step inside one opaque call. The ops arrive unlabelled, so the report takes the dominant
fidelity across the whole pile and applies it to all three stacks -- right only while they agree.

WHAT THIS ADDS. A SECOND, MARKED PASS after the measured one: each stage the model declares, run
once through its own `<stage>_trace_step`, bracketed by tracy signposts. tt-perf-report already
slices a capture between two signposts, so the report can then price each stack at its own peak.

WHY A SECOND PASS RATHER THAN REPLACING run_head. 94 of the 112 ops in a real capture carry no
parseable shape, so there is no way to prove by inspection that per-stage steps cover the same op
set -- and an op that only run_head reaches would vanish from the ladder's view. The measured region
keeps its exact op set; the marked pass is additive and is used for fidelity rollup only. The two
are kept apart by the conventional start/stop pair, which resolve_signposts already looks for and
refine() already slices on, so the main report sees exactly what it saw before.
"""
from __future__ import annotations

# The seam names live in ONE module -- see stage_seams. A RELATIVE import resolves under both
# names this package is imported by, so neither spelling has to be guarded.
from . import stage_seams as _seams

import ast
import sys

_UNKNOWN = object()


def signpost(name: str) -> None:
    """Emit one tracy signpost. Best-effort: a mark that cannot be written costs the split, not the run.

    tracy.signpost goes out through ttnn.tracy_message -- the same channel the op records travel --
    and process_ops_logs writes it as a row whose OP TYPE is "signpost", which is what
    tt-perf-report slices on. So a mark is a real row in the op stream, not a host annotation.
    """
    try:
        from tracy import signpost as _sp

        _sp(name)
    except Exception as exc:  # noqa: BLE001
        print(
            "  [stage-marks] could not emit signpost %r (%s: %s) -- the capture will carry no stage "
            "boundary, so every stack shares one math-fidelity peak." % (name, type(exc).__name__, str(exc)[:140]),
            file=sys.stderr,
            flush=True,
        )


def _no_marks(why: str) -> None:
    """Say why there will be no per-stage split. The absence of this line is what hid the defect.

    A zero from mark_stages reached the report as `stage_buckets {}`, which the roofline renders as
    one shared peak across every stack -- the same output an unmarked capture produces. Nothing
    distinguished "not asked", "asked and refused" and "ran and failed", so nine attempts at the
    stage axis all looked alike from the outside."""
    print(
        "  [stage-marks] NO per-stage boundaries: %s -- every stack will share one math-fidelity "
        "peak in the roofline." % why,
        file=sys.stderr,
        flush=True,
    )


def mark_stages(adapter, device) -> int:
    """Run each declared stage once, eagerly, between marks. Returns how many stages were marked.

    Eager by necessity and by policy: the profiler attributes per-op time from eager dispatch, and
    synchronising inside a trace capture is fatal ("Event Synchronization is not supported during
    trace capture"). Nothing here opens a capture.

    Zero is a real answer -- a pipeline that declares no stages, or whose steps will not run one at a
    time, simply gets no split, and every consumer keeps the whole-profile figure it already had.
    But zero must SAY WHICH, because the three reasons are not the same problem and were reported
    identically: this returned a bare 0 and printed nothing, so a caller that simply had not built
    the adapter looked exactly like a pipeline with no stages to declare.
    """
    try:
        import ttnn
    except Exception as exc:  # noqa: BLE001
        _no_marks("ttnn is not importable here (%s: %s)" % (type(exc).__name__, str(exc)[:120]))
        return 0
    # SETUP BUILDS THE STAGES. `stages` is [] until setup() runs -- __init__ only declares the
    # attribute, and setup() is what constructs the pipeline and binds one _Stage per declared stage.
    # measure_adapter calls it (trace_replay: `adapter.setup(device)`); this did not, and read the
    # empty list straight off a freshly constructed adapter. So every run returned 0 stages having
    # never asked the pipeline, emitting the start/stop pair and none of the per-stage boundaries,
    # which is exactly what the capture showed: 2 signposts where 8 were expected, and a roofline
    # that shared one math-fidelity peak across all three stacks.
    if not list(getattr(adapter, "stages", None) or []):
        _setup = getattr(adapter, "setup", None)
        if callable(_setup):
            try:
                _setup(device)
            except Exception as exc:  # noqa: BLE001
                _no_marks("adapter.setup failed (%s: %s)" % (type(exc).__name__, str(exc)[:140]))
                return 0
    stages = list(getattr(adapter, "stages", None) or [])
    if not stages:
        _no_marks("the pipeline declares no stages after setup")
        return 0
    n = 0
    for st in stages:
        name = str(getattr(st, "name", "") or "").strip()
        step = getattr(st, "step", None)
        if not name or not callable(step):
            continue
        signpost("stage:%s" % name)
        try:
            step()
            ttnn.synchronize_device(device)
            n += 1
        except Exception as exc:  # noqa: BLE001
            # One stage that will not run alone must not cost the others their boundary, nor the run.
            print(
                "  [stage-marks] stage %r could not be run on its own (%s: %s); no boundary for it"
                % (name, type(exc).__name__, str(exc)[:140]),
                file=sys.stderr,
                flush=True,
            )
        finally:
            signpost("stage:%s:end" % name)
    if not n:
        _no_marks("%d declared stage(s), none could be run one at a time" % len(stages))
    return n


def _looks_like_a_pipeline(obj) -> bool:
    """Does this object expose the stage surface the adapter drives?

    The same two things perf_adapter looks for: a PIPELINE_STAGES list, or the per-stage trace hooks
    named after its entries. Shape, not type -- the tool never imports a model's classes."""
    names = getattr(obj, "PIPELINE_STAGES", None)
    if isinstance(names, (list, tuple)) and names:
        return True
    mod = sys.modules.get(type(obj).__module__)
    names = getattr(mod, "PIPELINE_STAGES", None) if mod else None
    if not isinstance(names, (list, tuple)) or not names:
        return False
    return any(callable(getattr(obj, _seams.hook(n, _seams.STEP), None)) for n in names)


def find_pipeline_in_scope(scope: dict):
    """The pipeline object among a function's locals, or None.

    BY SHAPE, BECAUSE THE NAME IS THE GENERATOR'S TO CHOOSE. Two earlier versions of this injection
    depended on identifiers -- first six required ones, then the arguments copied out of the test's
    own adapter call -- and both broke on a regenerated test. The object itself is unambiguous: it is
    the one carrying PIPELINE_STAGES or the <stage>_trace_step hooks.

    Ordered so a caller can be told WHICH candidate was taken when more than one qualifies, and
    deterministic (insertion order of locals) so two runs of the same test pick the same object.
    """
    found = [(k, v) for k, v in (scope or {}).items() if not k.startswith("__") and _looks_like_a_pipeline(v)]
    if not found:
        return None
    if len(found) > 1:
        print(
            "  [stage-marks] %d objects in scope expose a stage surface (%s); taking %r"
            % (len(found), ", ".join(k for k, _ in found), found[0][0]),
            file=sys.stderr,
            flush=True,
        )
    return found[0][1]


# The seam perf_adapter drives a stage through. Named once here because the injector recognises the
# test's input preparer BY IT -- a function that assigns these is preparing stages, whatever it is
# called -- and nothing else in this file may name a generated identifier.
_STAGE_INPUT_HOOK = _seams.INPUTS  # named once, in stage_seams


def find_input_preparer(text: str, at_line: int = 0) -> str:
    """The name of the test's own stage-input preparer, or "".

    STRUCTURAL, like everything else here. A pipeline's <stage>_trace_inputs() reads the captured
    golden tensors the bring-up wrote; a model being optimised for the first time has none, and
    nobody should have to hand one over. The generated test already solves that for the timing path
    by pointing those hooks at a real batch it builds from the demo -- so the marks reuse it instead
    of asking for a file.

    Found by what it DOES: a function that assigns attributes ending in the hook suffix. The suffix
    is perf_adapter's contract, not the generator's spelling, so a test that calls this
    `_bind_stage_inputs`, `_wire_inputs` or anything else is recognised the same way.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return ""
    visible = _scopes_visible_at(tree, at_line)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        # NO RULE ABOUT THE SIGNATURE. This required exactly one parameter, on the reasoning that a
        # preparer "takes the pipeline and only that". One generated test wrote _bind_stage_inputs(pipe)
        # and worked; the next wrote _patch_trace_inputs(pipe, batch) -- module level, doing exactly
        # the right thing, with a docstring explaining it exists because the _captured tensors are not
        # shipped -- and was skipped for having two. Every stage then fell back to those missing files
        # and the run marked nothing. A rule about SHAPE, and shape is what the generator varies:
        # it contains no name, so every scan for hardcoded identifiers passed it.
        if id(_enclosing_scope(tree, node)) not in visible:
            continue
        if _assigns_stage_hook(node):
            return node.name
    return ""


def _assigns_stage_hook(fn) -> bool:
    """Does THIS function assign a <stage>_trace_inputs, in its own body?

    Not ast.walk: that descends into nested definitions, so the enclosing test function inherited the
    assignments of a helper defined inside it and was itself offered as the preparer -- which would
    have called the test recursively. A function is the preparer only if it does the work."""
    stack = list(fn.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue  # a nested definition is its own candidate, not this one's work
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Attribute) and t.attr.endswith(_STAGE_INPUT_HOOK) for t in node.targets
        ):
            return True
        stack.extend(
            c for c in ast.iter_child_nodes(node) if isinstance(c, ast.stmt) or isinstance(c, ast.excepthandler)
        )
    return False


def _enclosing_scope(tree, node):
    """The function that `node` is defined in, or the module."""
    best = tree
    for cand in ast.walk(tree):
        if not isinstance(cand, (ast.FunctionDef, ast.AsyncFunctionDef)) or cand is node:
            continue
        if cand.lineno < node.lineno <= (cand.end_lineno or cand.lineno):
            if best is tree or cand.lineno > best.lineno:
                best = cand
    return best


def _scopes_visible_at(tree, line: int) -> set:
    """The scopes a name at `line` can resolve in: the module and every function enclosing it.

    WHY A NAME IS NOT ENOUGH. The preparer is found by what it does, which is right, and the first
    version stopped there -- it matched a function that genuinely assigns the stage-input hooks but
    was a LOCAL of the traced path, invisible from the eager function where the marks run. That is
    the same scope mistake as copying the adapter's arguments across branches, in a new place:
    NameError("name '_build_for_perf' is not defined"), reported by the run itself this time.

    Line 0 means "module scope only", which is the safe reading for a caller that has no site yet.
    """
    out = {id(tree)}
    if not line:
        return out
    for cand in ast.walk(tree):
        if isinstance(cand, (ast.FunctionDef, ast.AsyncFunctionDef)) and cand.lineno < line <= (
            cand.end_lineno or cand.lineno
        ):
            out.add(id(cand))
    return out


def _call_preparer(bind, pipe, scope: dict):
    """Call the test's preparer, filling whatever it asks for from the live scope.

    DERIVED, NOT ASSUMED. This used to call bind(pipe) and the finder therefore required a
    single-parameter function -- a rule about shape, and shape is what the generator varies. One test
    wrote _bind_stage_inputs(pipe); the next wrote _patch_trace_inputs(pipe, batch). Removing the rule
    without changing the call only moved the failure: the preparer was found and then raised
    TypeError: missing 1 required positional argument.

    The pipeline goes first because that is the one argument the preparer must take -- it is the thing
    being prepared. Every other parameter is looked up BY NAME in the scope the marks were handed,
    which is the locals() of the function that just ran the model, so the real batch, inputs or head
    the eager pass used are exactly what is there. A parameter with a default that is not in scope is
    left to its default; one without is reported by name rather than guessed at.
    """
    import inspect

    try:
        params = list(inspect.signature(bind).parameters.values())
    except (TypeError, ValueError):
        return bind(pipe)  # unintrospectable: the single-argument form is the only thing to try
    args, missing = [], []
    for i, prm in enumerate(params):
        if prm.kind in (prm.VAR_POSITIONAL, prm.VAR_KEYWORD):
            continue
        if i == 0:
            args.append(pipe)
            continue
        if prm.name in scope:
            args.append(scope[prm.name])
        elif prm.default is not prm.empty:
            break  # the rest are optional; let the preparer use its own defaults
        else:
            missing.append(prm.name)
    if missing:
        raise TypeError(
            "%s needs %s, which the profiled scope does not contain (it has: %s)"
            % (
                getattr(bind, "__name__", "the preparer"),
                ", ".join(missing),
                ", ".join(sorted(k for k in scope if not k.startswith("__"))[:12]),
            )
        )
    return bind(*args)


def mark_stages_in_scope(scope: dict, device, bind=None) -> int:
    """Mark each stage of whatever pipeline is live in `scope`. Returns how many were marked.

    The scope is the locals() of the function that built the model, so the pipeline is already
    constructed: no builder, no second build, and nothing that has to be in scope by name.

    `bind` is the test's own preparer, when it has one. A pipeline's <stage>_trace_inputs() hooks
    read the captured golden tensors the bring-up wrote, and a tree that never ran that capture has
    none -- voxtral's encode raised FileNotFoundError on a pristine checkout. The generated test
    already solves this for the timing path by pointing those hooks at a real batch built from the
    demo, so the marks use the SAME preparation rather than requiring a data file nobody ships with a
    new model. Absent, the pipeline's own hooks are used, which is correct for a model that needs no
    preparation."""
    pipe = find_pipeline_in_scope(scope)
    if pipe is None:
        _no_marks("no object in scope exposes PIPELINE_STAGES or <stage>_trace_step hooks")
        return 0
    if callable(bind):
        try:
            _call_preparer(bind, pipe, scope)
        except Exception as exc:  # noqa: BLE001
            # Not fatal: the pipeline's own hooks may still work, and perf_adapter now skips only the
            # stages that cannot prepare themselves.
            print(
                "  [stage-marks] the test's input preparer raised (%s: %s); falling back to the "
                "pipeline's own hooks" % (type(exc).__name__, str(exc)[:120]),
                file=sys.stderr,
                flush=True,
            )
    try:
        from .perf_adapter import PipelineStageAdapter as _PSA
    except Exception as exc:  # noqa: BLE001
        _no_marks("perf_adapter is not importable here (%s)" % type(exc).__name__)
        return 0
    # THE CAPTURES THE MODEL DECLARES BUT DOES NOT SHIP. A pipeline's <stage>_trace_inputs() reads the
    # golden tensors a bring-up capture wrote; those are large and uncommitted, so on a tree that has
    # never run that capture every stage raises FileNotFoundError and the split is lost. The manifest
    # beside them IS committed and declares every shape, and a timing measurement does not read the
    # values -- so the missing file is supplied from its own description. Installed only around this
    # walk, and only for files that do not exist.
    _restore = None
    try:
        from .captured_stub import install as _install_stub

        _restore = _install_stub()
    except Exception:  # noqa: BLE001 -- a stand-in that cannot be installed must not cost the marks
        _restore = None
    try:
        return mark_stages(_PSA(lambda _d: pipe), device)
    finally:
        if _restore is not None:
            _restore()


# --- deterministic injection into the generated perf test ----------------------------------------

# The names the injected block leans on. All are in scope at the injection point -- some are test
# locals, some module-level -- and all come from the skeleton the generator works from. They are
# CHECKED before injecting rather than assumed: a test that names things differently gets no marks
# and says so, instead of shipping a NameError into the one run that measures per-op time.
# NOTHING IS REQUIRED BY NAME, AND NOTHING IS COPIED ACROSS SCOPES.
#
# This first required six identifiers the generator was free not to use, then copied the arguments of
# the test's own PipelineStageAdapter(...) call. Both failed on a freshly generated test, the second
# more subtly: the args were copied correctly and still raised
# NameError("name '_build_for_perf' is not defined"), because the generator had put the builder and
# the prompt ids inside the nested `_traced_forward`, while the marks sit in the profiling branch.
# Text lifted out of one scope cannot run in another, whatever it is called.
#
# So the pass no longer needs a builder. It runs where the model is already live -- at the end of the
# eager function the marks bracket -- and is handed that function's own locals(). It finds the
# pipeline by SHAPE: the object exposing PIPELINE_STAGES or the <stage>_trace_step hooks the adapter
# already looks for. No names, no scopes, and it verifies what it found instead of assuming.

_INJECT_TEMPLATE = """{i}# --- stage marks (injected by perf_test_gen) -------------------------------------
{i}# The measured region is bracketed by the conventional start/stop pair so the main report
{i}# slices exactly the ops run_head emitted; the pass below is additive and feeds per-stage
{i}# fidelity only. Injected rather than written by the generator: the skeleton is advisory and
{i}# a generated test simply omitted this, which is why five earlier attempts measured nothing.
{i}try:
{i}    from models.experimental.perf_automation.agent import stage_marks as _tt_sm
{i}    from models.experimental.perf_automation.agent.perf_adapter import PipelineStageAdapter as _TtPSA
{i}except Exception:  # noqa: BLE001
{i}    _tt_sm = None
{i}if _tt_sm is not None:
{i}    _tt_sm.signpost("start")
{body}{i}if _tt_sm is not None:
{i}    _tt_sm.signpost("stop")
"""

# The per-stage pass, appended INSIDE the function the bracket wraps -- the one that built the model.
# locals() is what makes it name-free: the pipeline is in there under whatever the generator called
# it, and mark_stages_in_scope picks it out by shape.
_MARK_PASS_TEMPLATE = """{i}# --- per-stage marks (injected) ---------------------------------------------------
{i}# Runs HERE, at the end of the function that built the pipeline, because that object is a LOCAL of
{i}# this scope: an earlier version copied the test's own PipelineStageAdapter(...) arguments into the
{i}# profiling branch and raised NameError, since the generator had defined them inside another
{i}# function. Handed locals() rather than a name, so nothing depends on how the test spells things.
{i}print("STAGE_MARKS_ENTER", flush=True)
{i}try:
{i}    from models.experimental.perf_automation.agent import stage_marks as _tt_sm2

{i}    print("STAGE_MARKS_RESULT=%d" % _tt_sm2.mark_stages_in_scope(locals(), device{bind}), flush=True)
{i}except Exception as _tt_e2:  # noqa: BLE001
{i}    print("STAGE_MARKS_SKIPPED=%r" % (_tt_e2,), flush=True)
"""


def _env_value(node, env: dict):
    """os.environ.get("NAME"[, default]) evaluated against `env`, or _UNKNOWN."""
    if not isinstance(node, ast.Call):
        return _UNKNOWN
    f = node.func
    if getattr(f, "attr", None) != "get":
        return _UNKNOWN
    owner = getattr(f, "value", None)
    if getattr(owner, "attr", None) != "environ" and getattr(owner, "id", None) != "environ":
        return _UNKNOWN
    if not node.args or not isinstance(node.args[0], ast.Constant):
        return _UNKNOWN
    name = node.args[0].value
    if name in env:
        return env[name]
    if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
        return node.args[1].value
    return None


def _eval(node, env: dict, names: dict):
    """The value of an expression under the profiling environment, or _UNKNOWN.

    Deliberately small: names, constants, os.environ.get, not/and/or and ==/!=. That is the whole
    vocabulary a generated test uses to decide whether it is being profiled, and anything richer is
    answered _UNKNOWN so the caller can say it could not tell instead of guessing wrong."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return names.get(node.id, _UNKNOWN)
    v = _env_value(node, env)
    if v is not _UNKNOWN:
        return v
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        inner = _eval(node.operand, env, names)
        return _UNKNOWN if inner is _UNKNOWN else (not inner)
    if isinstance(node, ast.BoolOp):
        vals = [_eval(v, env, names) for v in node.values]
        if isinstance(node.op, ast.And):
            if any(v is not _UNKNOWN and not v for v in vals):
                return False
            return _UNKNOWN if any(v is _UNKNOWN for v in vals) else True
        if any(v is not _UNKNOWN and v for v in vals):
            return True
        return _UNKNOWN if any(v is _UNKNOWN for v in vals) else False
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        left, right = _eval(node.left, env, names), _eval(node.comparators[0], env, names)
        if left is _UNKNOWN or right is _UNKNOWN:
            return _UNKNOWN
        if isinstance(node.ops[0], ast.Eq):
            return left == right
        if isinstance(node.ops[0], ast.NotEq):
            return left != right
    return _UNKNOWN


def _bind_names(body, env: dict, names: dict) -> None:
    """Record simple `x = <expr>` bindings so later conditions can be evaluated."""
    for st in body:
        if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
            val = _eval(st.value, env, names)
            if val is not _UNKNOWN:
                names[st.targets[0].id] = val


def reachable_bare_calls(text: str, env: dict | None = None) -> list:
    """[(lineno, indent, callee)] for bare `_f()` statements the PROFILING run actually reaches.

    WHY EVALUATE RATHER THAN GUESS. The rule here was "the last bare call is the profiled branch". It
    held for one generated test and not the next: that one ended with

        else:
            _eager_forward()          <- what the profiler runs
            if _PERF_TRACE:
                _try_traced()         <- last, and dead under profiling

    so the marks went into a branch that never executes and the capture came back with no signposts
    and no diagnostics -- silence from a block that never ran.

    The run's own environment settles it. probes.PROFILING_ENV is what the tracy subprocess is given,
    the test decides its branch from exactly those variables, and both facts are the tool's, not the
    generator's. Conditions that cannot be decided leave BOTH branches in, which is the safe reading:
    a call that might run is better than a call that certainly does not.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    env = dict(env or _profiling_env())
    lines = text.splitlines()
    out = []

    def walk(body, names):
        _bind_names(body, env, names)
        for st in body:
            if isinstance(st, ast.If):
                verdict = _eval(st.test, env, names)
                if verdict is _UNKNOWN:
                    walk(st.body, dict(names))
                    walk(st.orelse, dict(names))
                elif verdict:
                    walk(st.body, dict(names))
                else:
                    walk(st.orelse, dict(names))
                continue
            if isinstance(st, (ast.For, ast.While, ast.With, ast.Try)):
                for sub in ("body", "orelse", "finalbody", "handlers"):
                    inner = getattr(st, sub, None) or []
                    for h in inner:
                        if isinstance(h, ast.ExceptHandler):
                            walk(h.body, dict(names))
                        elif isinstance(h, ast.stmt):
                            walk([h], dict(names))
                continue
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue  # a definition is not a call
            if (
                isinstance(st, ast.Expr)
                and isinstance(st.value, ast.Call)
                and isinstance(st.value.func, ast.Name)
                and not st.value.args
                and not st.value.keywords
            ):
                line = lines[st.lineno - 1]
                out.append((st.lineno, line[: len(line) - len(line.lstrip())], st.value.func.id))

    # MODULE SCOPE FIRST, and carried in. `_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"`
    # sits at module level while the branch that reads it is inside the test function -- binding each
    # top-level statement into a throwaway dict left the function unable to decide its own condition,
    # so both branches came back reachable and the marks went to the wrong one anyway.
    module_names: dict = {}
    _bind_names(tree.body, env, module_names)
    # ONLY THE TEST FUNCTION. pytest enters through it, so that is where "the profiled path" starts;
    # walking every module-level def also collected the bare calls inside helper functions, which are
    # reached only if something calls them. `test_` is pytest's own contract -- the same one the node
    # ids in the manifest are built from -- not a guess about this generator.
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            walk(node.body, dict(module_names))
    return out


def _profiling_env() -> dict:
    """The env a tracy run executes under -- from probes, which is what sets it."""
    try:
        from .probes import PROFILING_ENV

        return dict(PROFILING_ENV)
    except Exception:  # noqa: BLE001
        return {}


def inject_stage_marks(text: str) -> tuple:
    """Wrap the profiled eager measurement in marks and append the per-stage pass. (text, why).

    Deterministic, because the skeleton is not. _SKELETON_REF is "structural reference handed to the
    LLM", so anything added there is a suggestion: the generated test for voxtral came back with zero
    references to it, and five attempts' worth of downstream machinery sat starved behind that.

    Idempotent, and refuses rather than guesses: no bare `_eager_forward()` statement, or a test that
    does not define the helpers the block needs, means no injection and a stated reason.
    """
    if "_tt_sm" in text:
        return text, "already injected"
    lines = text.splitlines(keepends=True)
    # THE CALL THE PROFILER ACTUALLY REACHES, decided by evaluating the test's own branches under the
    # environment the tracy subprocess is given. Position is not a signal: the previous rule took the
    # LAST bare call and put the marks inside `if _PERF_TRACE:`, which is false under profiling.
    reach = reachable_bare_calls(text)
    if not reach:
        return text, "no bare call to an eager pass on the profiled path"
    lineno, ind, fname = reach[-1]
    k = lineno - 1
    # WHERE THE PIPELINE LIVES. The per-stage pass needs the built model, and that is a local of the
    # function being called here -- not of this scope. Copying the test's adapter arguments into this
    # branch is what raised NameError on a regenerated test. So the bracket stays here and the pass is
    # appended to the END of that function's body, where its locals() still hold the pipeline.
    end, find = _function_body_end(text, fname)
    if end is None:
        return text, "cannot find the body of %s() to append the per-stage pass to" % fname
    lines[k] = _INJECT_TEMPLATE.format(i=ind, body=lines[k])
    out = "".join(lines)
    # Re-locate after the bracket edit shifted the lines below it.
    end2, find2 = _function_body_end(out, fname)
    if end2 is None:
        return text, "lost the body of %s() after bracketing" % fname
    o = out.splitlines(keepends=True)
    _prep = find_input_preparer(out, end2)
    o.insert(end2, _MARK_PASS_TEMPLATE.format(i=find2, bind=(", bind=%s" % _prep) if _prep else ""))
    return "".join(o), "injected at line %d, per-stage pass in %s()" % (k + 1, fname)


def _function_body_end(text: str, name: str):
    """(index of the line AFTER the last statement of `name`, that body's indent), or (None, "").

    Parsed, so a comment or a nested def cannot be mistaken for the end of the body."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None, ""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name and node.body:
            indent = " " * (node.body[0].col_offset)
            # BEFORE THE RETURN, not after the last statement. Appending to the end of the body put
            # the pass underneath `return out` -- syntactically fine, never executed, and it would
            # have reported "no marks" from a block that ran zero times. The pipeline is already
            # bound by then, so immediately before the return is both reachable and late enough.
            for st in node.body:
                if isinstance(st, ast.Return):
                    return st.lineno - 1, indent
            last = node.body[-1]
            return (last.end_lineno or last.lineno), indent
    return None, ""
