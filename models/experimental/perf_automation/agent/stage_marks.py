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

import ast
import sys


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


# --- deterministic injection into the generated perf test ----------------------------------------

# The names the injected block leans on. All are in scope at the injection point -- some are test
# locals, some module-level -- and all come from the skeleton the generator works from. They are
# CHECKED before injecting rather than assumed: a test that names things differently gets no marks
# and says so, instead of shipping a NameError into the one run that measures per-op time.
# NOTHING IS REQUIRED BY NAME ANY MORE.
#
# This listed six identifiers the block used to rebuild an adapter -- _stage_inputs_from_demo,
# _get_pipe, prompt_ids_for_isl, get_tokenizer, PERF_ISL_TOKENS, PERF_BATCH -- and refused when the
# test did not define them. But the test is written by an LLM from an advisory skeleton, so those are
# ITS naming choices, not a contract. Run 27, with the model reset to pristine and the test therefore
# regenerated: it named the same things _build_for_perf and _prompt_ids, the check failed on the two
# private helpers, and the marks were refused -- so the roofline shared one peak across every stack
# for a tenth run, because the injector was coupled to how the generator spells things.
#
# Swapping in _build_for_perf would only move the hardcoded name. The test already CONSTRUCTS the
# adapter the marks need -- measure_adapter(PipelineStageAdapter(...), device) -- so the block copies
# that call's arguments verbatim and reuses them, whatever they are called. The single fixed token is
# PipelineStageAdapter, which is the tool's own API -- an anchor of the same kind as any other ttnn
# entry point the harness names, and nothing about how a generated test spells its own helpers.
_ADAPTER_CALL = "PipelineStageAdapter("

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
{i}    try:
{i}        _tt_sm.mark_stages(_TtPSA({adapter_args}), device)
{i}    except Exception as _tt_e:  # noqa: BLE001
{i}        print("STAGE_MARKS_SKIPPED=%r" % (_tt_e,), flush=True)
"""



def _adapter_args(text: str) -> str:
    """The argument text of the test's own PipelineStageAdapter(...) call, or "".

    The test already builds the adapter the marks need, for measure_adapter. Copying its arguments
    means the block works whatever the generator named the builder and the prompt ids -- which is the
    whole point: those names are the LLM's choice, and requiring particular ones is what refused the
    injection on a freshly generated test.

    PARSED, NOT SCANNED. A balanced-paren walk over the text reads past a call whose parens do not
    close and swallows whatever follows -- on the real file it returned `..., batch=PERF_BATCH, device`
    by closing on the enclosing measure_adapter(. The file is Python that has to run, so ast can find
    the call properly; a file that will not parse is refused, which is the honest answer.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return ""
    name = _ADAPTER_CALL.rstrip("(")
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        ident = getattr(f, "id", None) or getattr(f, "attr", None)
        if ident != name:
            continue
        parts = []
        for a in node.args:
            seg = ast.get_source_segment(text, a)
            if not seg:
                return ""
            parts.append(" ".join(seg.split()))
        for kw in node.keywords:
            seg = ast.get_source_segment(text, kw.value)
            if not seg:
                return ""
            star = "**" if kw.arg is None else "%s=" % kw.arg
            parts.append("%s%s" % (star, " ".join(seg.split())))
        return ", ".join(parts)
    return ""


def inject_stage_marks(text: str) -> tuple:
    """Wrap the profiled eager measurement in marks and append the per-stage pass. (text, why).

    Deterministic, because the skeleton is not. _SKELETON_REF is "structural reference handed to the
    LLM", so anything added there is a suggestion: the generated test for voxtral came back with zero
    references to it, and five attempts' worth of downstream machinery sat starved behind that.

    Idempotent, and refuses rather than guesses: no bare `_eager_forward()` statement, or a test that
    does not define the helpers the block needs, means no injection and a stated reason.
    """
    import re

    if "_tt_sm" in text:
        return text, "already injected"
    args = _adapter_args(text)
    if not args:
        return text, "no PipelineStageAdapter(...) call to copy the adapter from"
    lines = text.splitlines(keepends=True)
    at = [k for k, l in enumerate(lines) if re.match(r"^(\s*)_eager_forward\(\)\s*$", l)]
    if not at:
        return text, "no bare _eager_forward() statement"
    # The LAST one is the profiled branch; an earlier one is the trace-replay fallback.
    k = at[-1]
    ind = re.match(r"^(\s*)", lines[k]).group(1)
    lines[k] = _INJECT_TEMPLATE.format(i=ind, body=lines[k], adapter_args=args)
    return "".join(lines), "injected at line %d" % (k + 1)
