# SPDX-License-Identifier: Apache-2.0
"""Model-agnostic trace-replay latency measurement for optimize/perf.

`measure_adapter(adapter, device)` drives a PipelineStageAdapter (see perf_adapter.py): for EACH stage
emit-e2e emitted (adapter.stages, from the pipeline's PIPELINE_STAGES) it captures one steady-state,
host-op-free step as a device trace and replays it under trace on a single command queue. It prints,
per stage:

    TRACE_STAGE_MS[<stage>]=<float> path=trace+1cq

plus the headline clean, GPU-comparable wall the harness parses:

    TRACE_PER_TOKEN_MS=<float>     (the AR/decode stage if present, else the whole-pipeline sum)

which the harness (agent/tracy_tool.py + cc_optimize/perf_mcp.py) reads as the `trace` metric source
(vs the `eager` fallback FORWARD_WALL_MS). This is the companion of perf_adapter.py: the adapter is
the shell (setup + per-stage step), this module is the engine (warmup -> capture -> timed replay ->
emit the numbers). A legacy single-step adapter (PipelineDecodeAdapter, no .stages) is wrapped as one
"decode" stage, so the old path is unchanged.

Measurement is trace+1cq end to end. Caller contract (the generated perf test): call inside the
`if _PERF_TRACE:` block, on the SAME device the test opened WITH `trace_region_size`. Any failure here
(notably a repeat-prefill pipeline with no `decode_step`, which raises in `adapter.setup`) propagates
out; the perf test's guard catches it, prints `TRACE_REPLAY_SKIPPED=...`, and falls back to
FORWARD_WALL_MS.
"""

from __future__ import annotations

import inspect
import os
import sys
import time

import ttnn

# Warmup compiles kernels + populates RoPE/mask/KV caches (trace capture can neither compile nor
# upload from host); replay iters are averaged for a stable per-token number. Both env-tunable.
_WARMUP_ITERS = max(1, int(os.environ.get("TT_TRACE_WARMUP_ITERS", "3")))
# How far into a container the read-set walk looks for tensors. Op arguments nest a level or two
# (a list of tensors, a dict of them); anything deeper is structure, not a read set.
_NOTE_MAX_DEPTH = 3
# The replay count an operator may ask for (emit-e2e / optimize --trace-replays set it); this is its
# one spelling, which the CLI and the gate server forward by name.
REPLAY_ITERS_ENV = "TT_TRACE_REPLAY_ITERS"
_REPLAY_ITERS = max(1, int(os.environ.get(REPLAY_ITERS_ENV, "16")))


# HOW MANY REPLAYS TO AVERAGE IS A CONSEQUENCE OF WHAT ONE COSTS, NOT A NUMBER PER MODEL.
#
# 16 is a noise-reduction choice, and a good one for a step costing milliseconds: the timer's own
# jitter is a large fraction of one iteration, so averaging many is how the figure becomes stable.
# For a stage costing 68 SECONDS an iteration it buys nothing measurable -- the jitter is already
# negligible against the signal -- and costs 18 minutes of device time per stage. That is what put a
# Qwen-Image-Edit capture past the budget it was given, where an absolute ceiling (4 x a typed 900 s)
# killed it mid-progress, twice, 3611 s apart, with one stage traced and the next mid-replay.
#
# The caller already states the constraint: `run_fresh_trace_capture` puts its budget in the
# environment and the capture child inherits it. So ONE replay is timed first, and only as many more
# as that budget affords are added -- at least two, so there is always an average, and never more
# than was asked for, so a cheap stage is measured exactly as before. The count is the means; the
# budget is the constraint, and it is the caller's to state rather than this file's to guess.
# BUDGETS ONLY. This listed PERF_MCP_MEASURE_STALL_SEC, which is a STALL WINDOW -- run.py passes it
# as `stall_s=`, the no-progress detector's window, not an allowance for the work. Sizing a
# measurement from it would have cut sample counts in the optimize domain, where it is routinely set,
# for a reason that has nothing to do with how long the measurement may take.
_REPLAY_BUDGET_ENVS = ("TT_PERF_REPLAY_BUDGET_S", "PERF_MCP_VALIDATE_TIMEOUT", "PERF_MCP_MEASURE_BACKSTOP")
_MIN_REPLAY_ITERS = 2  # an average needs two; dimensionless, so it assumes nothing about the model


def _measurement_budget_s(stages=1) -> float:
    """This stage's share of the budget the caller gave the capture, or 0 when nothing was stated."""
    for key in _REPLAY_BUDGET_ENVS:
        try:
            v = float(os.environ.get(key) or 0)
        except ValueError:
            v = 0.0
        if v > 0:
            return v / max(1, int(stages or 1))
    return 0.0


# WHAT A STAGE SPENDS BEFORE ITS FIRST REPLAY, AND CANNOT NOT SPEND.
#
# Every stage runs its step in full five times before a single replay is timed: once for
# _count_op_dispatches ("Run `fn` once"), _WARMUP_ITERS times inside _capture_step_trace's _warm, and
# once more inside the capture itself. _affordable_iters only ever sizes the REPLAY count, so those
# five are not negotiable -- and a budget handed out as if they were free is overspent before the
# sizing is consulted. On Qwen-Image-Edit's denoise (119.9s a step) they are 599s of a stage's share.
_FIXED_STEP_CALLS = 1 + _WARMUP_ITERS + 1


def _replay_share(budget_s, per_iter_s) -> float:
    """What is left of a stage's budget once its unavoidable pre-replay executions are paid for.

    0 budget means "nothing stated" to _affordable_iters, which then honours the request in full, so
    an unstated budget must stay exactly 0 here rather than becoming a tiny positive number."""
    try:
        budget_s = float(budget_s or 0.0)
        per_iter_s = float(per_iter_s or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if budget_s <= 0 or per_iter_s <= 0:
        return budget_s if budget_s > 0 else 0.0
    return max(0.0, budget_s - _FIXED_STEP_CALLS * per_iter_s)


def _affordable_iters(requested, per_iter_s, budget_s) -> int:
    """As many replays as the budget affords: >= _MIN_REPLAY_ITERS, <= what was asked for."""
    try:
        requested = max(1, int(requested))
    except (TypeError, ValueError):
        requested = 1
    if not budget_s or not per_iter_s or per_iter_s <= 0:
        return requested  # nothing stated, or no cost observed -> behave exactly as before
    affordable = int(budget_s // per_iter_s)
    return max(min(requested, _MIN_REPLAY_ITERS), min(requested, affordable))


# A LOOP THAT PRINTS NOTHING READS AS A HANG, AND IT IS NOT ONE.
#
# Warmup, capture and replay ran their iterations in silence, and the replay batch is enqueued
# non-blocking and then waited on in ONE `synchronize_device` -- so a stage whose step costs tens of
# seconds produces no log line, no syscall and no stack movement for the whole of 3 + 16 iterations.
# Every supervising watchdog in this tree judges liveness on exactly those signals
# (probes.progress_signature), so it reports "no forward progress" and kills work that is perfectly
# healthy. On 2026-09-29 that killed one Qwen-Image-Edit stage three times over, each kill reported as
# a device wedge, and the retry loop re-ran the same silence.
#
# THE CURE FOR SILENCE IS TO SPEAK, NOT TO PICK A LARGER TIMEOUT. Whatever number anyone chooses is a
# guess about a stage nobody has measured yet; a line per iteration is a fact. Growing the log is
# already one of the four progress signals, so no watchdog, window or caller needs to change.
#
# The cadence is the one thing that must not be typed: it has to stay INSIDE whatever window is
# supervising us, and only that supervisor knows it. It tells us in the environment, so the beat is
# read from there and divided; a dimensionless divisor carries no assumption about the model, the
# stage or the box. With nothing said, fall back to the tightest cadence any caller supervises at.
_MARK_PREFIX = "TRACE_"  # every line this module prints for a reader to parse starts with it
_HEARTBEAT_MARK = _MARK_PREFIX + "STAGE_WAITING"  # a sign of life while a stage runs, not a result
_HEARTBEAT_DIVISOR = 4  # beats per supervising window, so a missed beat is never the first evidence
_STALL_WINDOW_ENVS = ("PERF_MCP_VALIDATE_STALL_SEC", "PERF_MCP_MEASURE_STALL_SEC", "PERF_MCP_ROUND_STALL_SEC")


def _heartbeat_s() -> float:
    """Seconds between liveness lines: a fraction of the tightest window supervising this process."""
    windows = []
    for key in _STALL_WINDOW_ENVS:
        try:
            v = float(os.environ.get(key) or 0)
        except ValueError:
            v = 0.0
        if v > 0:
            windows.append(v)
    if not windows:
        try:  # what probes._execute supervises at when no caller narrowed it
            from .probes import _execute

            import inspect

            windows.append(float(inspect.signature(_execute).parameters["stall_timeout_s"].default))
        except Exception:  # noqa: BLE001
            return 0.0  # nothing to stay inside of -> the caller is unsupervised, stay quiet
    return max(1.0, min(windows) / _HEARTBEAT_DIVISOR)


class _Alive:
    """Say so, on a beat, while a blocking device call holds this process silent.

    The same shape as the thermal wait's heartbeat (cc_optimize/run's _COOL_HEARTBEAT_S): a child
    that is busy ON PURPOSE re-asserts it, rather than every supervisor being taught to expect this
    particular silence."""

    def __init__(self, what: str):
        self._what = what
        self._stop = None
        self._t = None

    def __enter__(self):
        beat = _heartbeat_s()
        if beat <= 0:
            return self
        import threading

        self._stop = threading.Event()
        t0 = time.monotonic()

        def _beat():
            while not self._stop.wait(beat):
                print(
                    "%s[%s] %ds on device" % (_HEARTBEAT_MARK, self._what, int(time.monotonic() - t0)),
                    flush=True,
                )

        self._t = threading.Thread(target=_beat, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        if self._stop is not None:
            self._stop.set()
            self._t.join(timeout=1)
        return False


def _iterate(step, iters, what, keep=False):
    """Run `iters` steps, reporting on the heartbeat cadence, keeping the results only if asked.

    THROTTLED, NOT PER-ITERATION, and it does not retain what it does not need: this runs inside a
    timed region, a fast stage does thousands of iterations, and the replay loop deliberately drops
    each result so the device buffers it holds are freed. Beating at the same cadence as the waits
    costs nothing measurable and breaks the silence just as well."""
    out = []
    n = max(0, iters)
    beat = _heartbeat_s()
    last = time.monotonic()
    # THE LOOP CAN ONLY SPEAK BETWEEN ITERATIONS, AND ONE ITERATION CAN OUTLAST THE WINDOW. A stage
    # whose single step costs more than the supervising window is silent for the whole of it, so
    # reporting per iteration is not enough on its own -- the beat has to come from somewhere that is
    # not waiting on the step. Hence the heartbeat across the whole loop: the ITER lines mark
    # progress, the heartbeat proves liveness, and no step duration can outrun it.
    with _Alive(what):
        for i in range(n):
            r = step()
            if keep:
                out.append(r)
            now = time.monotonic()
            if i + 1 == n or (beat > 0 and now - last >= beat):
                last = now
                print("TRACE_STAGE_ITER[%s]=%d/%d" % (what, i + 1, n), flush=True)
    return out


def _warm(step, iters):
    """Run the warmup steps, keeping what each one handed back.

    THE ADVANCE CHECK RIDES HERE. This loop already happens, it sits outside the timed region, and it
    is the one thing the capture path and the self-traced path have in common -- putting the check in
    either alone would have missed every pipeline that takes the other, and gemma3 takes the other.
    """
    return _iterate(step, iters, "warmup", keep=True)  # _check_advance reads the samples


def _check_advance(samples):
    """Refuse to time a decode that is not decoding.

    The e2e PCC gate drives the model's forward while advancing the position ITSELF and teacher-forcing
    reference tokens, so the loop measured here -- its token pick, its position advance, its feeding of
    its own output back in -- is never on the correctness path. This is the only place that sees it,
    and it sees it on exactly the code that gets timed.
    """
    try:
        from .perf_adapter import DecodeNotAdvancing, advance_verdict
    except Exception:  # noqa: BLE001 -- instrumentation must never be what breaks a measurement
        return
    verdict = advance_verdict(samples)
    print("TRACE_DECODE_ADVANCE=%s" % verdict["status"], flush=True)
    if verdict["status"] == "stuck":
        raise DecodeNotAdvancing(verdict["reason"])


def _capture_step_trace(device, step):
    """Warm up, then capture exactly one host-op-free, fixed-shape step as a trace on cq0."""
    _check_advance(_warm(step, _WARMUP_ITERS))
    with _Alive("capture"):
        ttnn.synchronize_device(device)
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        step()
        ttnn.end_trace_capture(device, tid, cq_id=0)
        ttnn.synchronize_device(device)
    return tid


def _replay_1cq(device, tid, iters, budget_s=0.0):
    # The replays are enqueued non-blocking ON PURPOSE -- that is what makes this a steady-state
    # throughput measurement -- so the wait is one silent block that no per-iteration print can
    # break. Hence the heartbeat: it grows the log without touching the timed region.
    #
    # ONE FIRST, TO LEARN WHAT IT COSTS. Its own time is a valid steady-state sample (the trace is
    # already captured and warmed), so nothing is spent to find out; it just decides how many more
    # are worth adding. See _affordable_iters.
    t1 = time.perf_counter()
    ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    with _Alive("replay 1"):
        ttnn.synchronize_device(device)
    one_s = time.perf_counter() - t1
    n = _affordable_iters(iters, one_s, _replay_share(budget_s, one_s))
    print("TRACE_STAGE_REPLAYS=%d of %d requested (%.1fs each)" % (n, iters, one_s), flush=True)
    if n <= 1:
        return one_s
    t0 = time.perf_counter()
    for _ in range(n):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    with _Alive("replay x%d" % n):
        ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) / n


def _replay(device, tid, iters, budget_s=0.0):
    """Run whatever `_replay_1cq` is in force, handing it the budget only if it accepts one.

    A perf test may install its own replay over _replay_1cq, and one written before the budget
    existed takes three arguments. Qwen-Image-Edit 2026-10-01: its per-iteration progress replay was
    called with four, every stage raised TypeError, and the run timed nothing for hours. The budget
    is additive, so such a replay is called exactly as it was before it."""
    fn = _replay_1cq
    try:
        inspect.signature(fn).bind(device, tid, iters, budget_s)
    except TypeError:
        return fn(device, tid, iters)
    except ValueError:  # no signature to read -- call it the current way
        pass
    return fn(device, tid, iters, budget_s)


# A REPLAYED TRACE DISPATCHES ONE OP. An eager pass dispatches one per ttnn call in the model --
# 3,564 of them for a 48-layer gemma-3 prefill. Nothing else about the two paths differs by three
# orders of magnitude, so the dispatch count is the signal, and anything above this is eager.
_TRACED_OP_DISPATCH_MAX = int(os.environ.get("TT_TRACE_DISPATCH_MAX", "32"))


def _count_op_dispatches(fn, detail=None, tables=None):
    """Run `fn` once; return (dispatches, streamed_bytes) or (None, 0) if the hook did not install.

    WHAT A STAGE ACTUALLY READS, MEASURED. The roofline needs the bytes one unit of work streams, and
    until now nothing measured it. Every route was a guess about the checkpoint: the tower name list
    (`audio_tower|vision_tower|...`), or the stage_roots section map -- which was tried and dropped
    lm_head on every untied model. _stage_roofs already said why the profile could not settle it:
    "_top_ops keys on (op_code, shape, memory) and records nothing about which phase an op ran in",
    and the per-stage byte reader that once stood in summary consequently returned 0 on every real
    profile -- buckets carry no `bytes` key and their regime tag is "na" for all of them -- so the
    "observed-first" line in _bytes_for always took its fallback branch. That reader is now deleted.

    But the hook that counts dispatches sees EVERY op the stage runs, with its arguments -- and
    trace_replay already runs each stage in isolation. So the read set is right here: the distinct
    DEVICE tensors this stage's ops streamed. That is an observation of this stage on this build, not
    an inference from names, and it needs no new pass -- it rides the warmup call that was happening
    anyway.

    DISTINCT, BY IDENTITY. A weight used by two ops is one resident tensor, and the quantity this
    feeds -- params x width -- counts each weight once, so the working set is the comparable figure.
    Counting every use instead would be a different (larger) number and would not be comparable to
    the ceiling it is meant to replace.

    Host tensors are excluded by asking the tensor, via the census's own _on_device: voxtral keeps an
    fp32 host copy of its weights alive while the device holds bf16, and counting those reported
    29.96 GB for an 11.3 GB device footprint.

    Best-effort throughout. A tensor that raises when inspected is skipped rather than ending the
    step, because this is instrumentation riding a measurement: it may return less than the truth,
    never an exception.

    STREAMED, NOT TOUCHED. The first version summed every distinct device tensor an op was HANDED,
    and handed is not read. A decode step is handed its whole KV cache allocation and reads the rows
    filled so far; it is handed a replicated embedding table and reads one row per user; it is handed
    the buffers it is about to overwrite. Devstral 2026-10-15: 9.03 GB per chip reported for a step
    that streams about 4.6 GB of weights, so the memory floor came out at 17.6 ms and a 17.98 ms
    decode read as 98% of peak bandwidth -- inside the band, and the stop gate cleared it with half
    its ceiling still above it. Three rules, none of which names a model or a tensor:

      * A buffer an op PRODUCED OR UPDATED during the step -- returned as a new tensor, or returned
        as one of its own inputs when it had others to write from (update_cache, generic_op's
        preallocated output, output_tensor=) -- is state the step writes: an activation, an
        accumulator, a cache. It is not part of the weight stream. What one unit of work reads of
        such state (the KV history, the activations it carries) is priced from the model's geometry
        by perf_target.active_bytes, and summary adds that beside this figure -- the same split the
        census draws between weights and scratch, for the same reason: KV is linear in the token
        count and the weights are flat in it.
      * A buffer returned by an op that was handed ONLY that buffer is a view (reshape, a no-op
        relayout): nothing was written, and a weight seen through one stays streamed.
      * A table the PIPELINE DECLARED it reads by index (`tables`, from
        stage_seams.GATHERED_TABLES_ATTR), handed to an op beside an integer tensor -- the index it
        is selected from by -- is charged what came out of that op, not what it holds. Handed
        without an index, or to an op that returned nothing, it is streamed: nothing was selected
        from it (a tied embedding table multiplied by the vocabulary matmul is the real cost). A
        table nobody declared is streamed in full whatever the op is called. The first version
        keyed this rule on the op's ttnn API name (ttnn.embedding, ttnn.gather) and went blind the
        moment a model wrote its own gather as a generic_op program -- Devstral run 4 (2026-10-18)
        did, in its first round, and the 1.34 GB table was back in decode's floor.

    `tables` is the flat list stage_marks.pipeline_gathered_tables returns; a nested one is walked
    the same way every op argument is. `detail`, when a dict is passed, receives the breakdown beside
    the figure: in_step_bytes (what the step wrote, so did not stream), gathered_saved_bytes (table
    bytes no op read) and buffers.
    """
    # WHICHEVER CLASS THE BUILD REGISTERED. This patched `Operation` because that is the class
    # decorators.py defines first and it has the __call__ a dispatch hook wants. But ttnn chooses at
    # REGISTRATION time -- `operation_class = FastOperation if ttnn.CONFIG.enable_fast_runtime_mode
    # else Operation` (decorators.py) -- and that flag defaults to true in config.hpp. Measured on
    # this build: 735 ops are FastOperation and 0 are Operation, so the hook sat on a class nothing
    # instantiates, intercepted nothing, and reported 0 bytes for every stage on every run.
    #
    # Patching BOTH is the point. Swapping one name for the other would just move the blind spot to
    # any build running with fast runtime off, or with logging or comparison mode on, where
    # FastOperation delegates to Operation instead.
    try:
        from ttnn import decorators as _dec
    except Exception:  # noqa: BLE001
        fn()
        return None, 0
    targets = []
    for _cn in ("FastOperation", "Operation"):
        _c = getattr(_dec, _cn, None)
        if _c is not None and callable(getattr(_c, "__call__", None)):
            targets.append((_c, _c.__call__))
    if not targets:
        fn()
        return None, 0
    # ONE DEFINITION OF "HOW BIG IS THIS DEVICE TENSOR". The first version of this re-implemented
    # weight_census._tensor_entry inline -- the residency check, the shape fallback, the numel loop,
    # the dtype resolution -- and got it subtly wrong: it read `.dtype` only, missing the
    # `get_dtype()` fallback the census has, so a tensor exposing the latter counted as zero bytes.
    # Two definitions of one quantity is how the ceiling's divisor came to disagree with itself
    # before; this calls the census's.
    try:
        from .weight_census import _tensor_entry as _entry, bytes_per_elem as _bpe, tensor_key as _tkey
    except Exception:  # noqa: BLE001
        _entry = None
    n = [0]
    # Keyed by the buffer identity weight_census.tensor_key gives, over the whole step:
    inputs: dict = {}  # every device tensor some op was handed -> its bytes
    written: set = set()  # buffers some op produced, or updated in place
    streamed: set = set()  # buffers read by an op that reads what it is handed
    gathered: dict = {}  # table -> bytes the gather ops that read it produced from it

    def _note(x, depth=0):
        """Every device tensor in `x` -- bare, or inside a list, tuple or dict -- as a list of
        (key, bytes, dtype name), to a bounded depth.

        A tensor is not always an argument: it is very often INSIDE one. The first hook looked at
        bare positional args and kwargs values only, so every tensor passed in a list, tuple or
        dict -- ttnn.concat takes a list, generic_op its whole io list -- contributed nothing at
        all. Bounded depth, because this rides a measurement: op arguments nest a level or two, and
        a walk that chases arbitrary structure is how the census once wandered into a 131k-token
        vocabulary."""
        found: list = []
        if _entry is None or depth > _NOTE_MAX_DEPTH:
            return found
        try:
            ent = _entry(x)  # (numel, dtype_name) for a DEVICE tensor, None otherwise
            if ent:
                found.append((_tkey(x, ent), ent[0] * _bpe(ent[1]), str(ent[1])))
                return found
            if isinstance(x, (list, tuple, set, frozenset)):
                for _i in x:
                    found.extend(_note(_i, depth + 1))
            elif isinstance(x, dict):
                for _i in x.values():
                    found.extend(_note(_i, depth + 1))
        except Exception:  # noqa: BLE001
            return found
        return found

    def _is_index(dtype_name) -> bool:
        """An integer tensor beside a declared table is the index it is read by."""
        return str(dtype_name or "").startswith(("uint", "int"))

    # THE DECLARED TABLES, by the same identity the walk gives every buffer an op is handed, so a
    # table reached under another name still matches. Empty when the pipeline declared none.
    table_keys = {_k for _k, _b, _dt in _note(tables)} if tables is not None else set()

    def _make_counting(orig):
        def counting(self, *a, **kw):
            n[0] += 1
            ins: list = []
            for _x in a:
                ins.extend(_note(_x))
            for _x in kw.values():
                ins.extend(_note(_x))
            for _k, _b, _dt in ins:
                inputs.setdefault(_k, _b)
            out = orig(self, *a, **kw)
            # WHAT THE OP WROTE, read off what it returned. Instrumentation only: a return value
            # that cannot be walked leaves the inputs counted as read, never the call failed.
            outs = _note(out)
            in_keys = {_k for _k, _b, _dt in ins}
            out_keys = {_k for _k, _b, _dt in outs}
            for _k in out_keys:
                if _k in in_keys and len(in_keys) == 1:
                    continue  # one tensor in, the same buffer out: a view, nothing written
                written.add(_k)
            # WHAT THE OP READ OF WHAT IT WAS HANDED. A declared table beside an index -- an integer tensor
            # the op READ, not one it wrote into -- with something out of the op to price it by, is read at
            # what came out; everything else is read whole, the index itself included.
            _rows = sum(_b for _k, _b, _dt in outs)
            _indexed = bool(outs) and any(_is_index(_dt) for _k, _b, _dt in ins if _k not in out_keys)
            for _k, _b, _dt in ins:
                if _k in out_keys:
                    continue
                if _indexed and _k in table_keys:
                    gathered[_k] = gathered.get(_k, 0) + _rows
                else:
                    streamed.add(_k)
            return out

        return counting

    for _cls, _orig in targets:
        _cls.__call__ = _make_counting(_orig)
    try:
        fn()
    finally:
        for _cls, _orig in targets:
            _cls.__call__ = _orig
    # THE STREAMED SET: handed, never written, and of a table only what came out of it.
    total, in_step, saved = 0.0, 0.0, 0.0
    for _k, _b in inputs.items():
        if _k in written:
            in_step += _b
            continue
        if _k in streamed or _k not in gathered:
            total += _b
            continue
        _g = min(_b, gathered[_k])
        total += _g
        saved += _b - _g
    if isinstance(detail, dict):
        detail.update({"in_step_bytes": int(in_step), "gathered_saved_bytes": int(saved), "buffers": len(inputs)})
    return n[0], int(total)


def _report_read_set(stage_name, dispatches, ws_bytes, detail=None):
    """Say which of the three things happened. A read set of 0 and a hook that never ran were the
    SAME observable: both returned (0 bytes), both failed `if bytes > 0`, both printed nothing, and
    both arrived as `bytes: {}`. That silence is why a hook pointed at the wrong class survived every
    run and every test -- there was no output to disagree with.

    The two facts needed to tell them apart are already in hand: `dispatches` is None when the hook
    could not install and 0 when it installed and intercepted nothing. Print all three cases, and
    carry the op count beside the bytes so the magnitude can be sanity-checked at a glance."""
    if ws_bytes > 0:
        # THE BREAKDOWN RIDES BEHIND THE FIGURE. perf_mcp reads the integer up to the first space, so
        # trailing fields cost the parser nothing and tell a reader what the step wrote instead of
        # streamed, and how much table it was handed but did not read.
        _more = ""
        if isinstance(detail, dict) and detail:
            _more = " in_step=%d gathered_saved=%d" % (
                int(detail.get("in_step_bytes") or 0),
                int(detail.get("gathered_saved_bytes") or 0),
            )
        print("TRACE_STAGE_BYTES[%s]=%d ops=%s%s" % (stage_name, ws_bytes, dispatches, _more), flush=True)
        return
    if dispatches is None:
        _why = "hook did not install (ttnn.decorators unavailable or no op class)"
    elif not dispatches:
        _why = "hook installed but intercepted 0 op dispatches"
    else:
        _why = "%d op dispatches carried no device tensors" % dispatches
    print("TRACE_STAGE_BYTES_NONE[%s] reason=%s" % (stage_name, _why), flush=True)


def _measure_native(device, stage, budget_s=0.0, tables=None):
    """Time a SELF-TRACED stage: the pipeline owns its trace capture (persistent-buffer / vLLM-style
    decode, e.g. GLM's decode(enable_trace=True)), so we must NOT begin_trace_capture around it --
    doing so raises "Writes/Reads are not supported during trace capture" because the step does
    host<->device I/O + execute_trace internally. Instead warm it (the pipeline lazily captures its own
    trace on the first call) and time steady-state replays.

    THE PATH IS OBSERVED, NOT ASSUMED. This returned "trace+1cq" for every self-traced stage, on the
    strength of the stage having DECLARED itself self-traced -- but declaring the capability is not
    the same as exercising it. gemma-3's prefill declares it and then falls back to eager, because
    `can_enable_trace` consults an allow-list that the model config sets EMPTY for every device; the
    harness recorded 91.33 ms as `path=trace+1cq` while the model logged `trace: False` on every call.
    Half that number is Python dispatch, and the roofline then graded it against a band that assumes
    a traced measurement.

    So the last warmup is counted rather than merely run. `stage.trace_path()` still wins when a
    pipeline reports its own path; the count is what decides when nothing does."""
    _check_advance(_warm(stage.step, _WARMUP_ITERS - 1))
    _det: dict = {}
    dispatched, _ws_bytes = _count_op_dispatches(stage.step, _det, tables)
    _report_read_set(stage.name, dispatched, _ws_bytes, _det)
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    # Same rule as _replay_1cq: one step is a valid sample and tells us what the rest would cost.
    _t1 = time.perf_counter()
    stage.step()
    _one = time.perf_counter() - _t1
    _n = _affordable_iters(_REPLAY_ITERS, _one, _replay_share(budget_s, _one))
    print("TRACE_STAGE_REPLAYS=%d of %d requested (%.1fs each)" % (_n, _REPLAY_ITERS, _one), flush=True)
    _iterate(stage.step, max(0, _n - 1), stage.name)
    with _Alive(stage.name):
        ttnn.synchronize_device(device)
    # Divided by what actually RAN, not by what was requested: _n may be fewer, and dividing by the
    # request would report a per-iteration time smaller than any iteration took.
    per_s = (time.perf_counter() - t0) / max(1, _n)
    tp = getattr(stage, "trace_path", None)
    if callable(tp):
        try:
            path = str(tp())
        except Exception:  # noqa: BLE001
            path = "unknown"
    elif dispatched is None:
        # The counter could not be installed, so nothing was observed. "unknown" is the honest label:
        # it withholds the verdict downstream instead of asserting a path on no evidence.
        path = "unknown"
    else:
        path = "trace+1cq" if dispatched <= _TRACED_OP_DISPATCH_MAX else "eager"
    if dispatched is not None:
        print("TRACE_STAGE_OPS[%s]=%d path=%s" % (stage.name, dispatched, path), flush=True)
    # THE BYTES THIS STAGE READ, beside the time that read them. The one quantity the roofline could
    # never measure -- every other route infers it from the checkpoint and a naming convention.
    # Silent when zero: nothing observed is not "this stage reads nothing", and a stage with no
    # reported bytes keeps the existing estimate rather than being given a ceiling of infinity.
    return per_s * 1000.0, path


def _measure_stage(device, stage, budget_s=0.0, tables=None):
    """Capture stage.step as a trace, replay it on a single command queue, return (ms, path).

    THE READ SET IS OBSERVED HERE, for every stage. It was measured inside _measure_native, which
    only a SELF-TRACED pipeline reaches -- voxtral declares no self_traced, so all three of its
    stages took the capture branch below and the measurement never ran once. The stage doc recorded
    `bytes: {}` on every run while the roofline reported per-stage figures that were, in fact, the
    checkpoint estimate.

    One warmup call is spent on it, and it is the call _capture_step_trace was going to make anyway
    to reach steady state: the hook counts the distinct DEVICE tensors this stage's ops stream --
    handed and never written, a gathered table at what came out of it -- which is the quantity
    params x width is trying to approximate.
    """
    if getattr(stage, "self_traced", False):
        return _measure_native(device, stage, budget_s, tables=tables)
    _ws_bytes, _n, _det = 0, None, {}
    try:
        _n, _ws_bytes = _count_op_dispatches(stage.step, _det, tables)
    except Exception as _he:  # noqa: BLE001 -- instrumentation must never cost the measurement
        print("TRACE_STAGE_BYTES_NONE[%s] reason=hook raised %s" % (stage.name, type(_he).__name__), flush=True)
        _ws_bytes, _n = 0, None
    else:
        _report_read_set(stage.name, _n, _ws_bytes, _det)
    tid = _capture_step_trace(device, stage.step)
    try:
        per_s = _replay(device, tid, _REPLAY_ITERS, budget_s)
        path = "trace+1cq"
    finally:
        try:
            ttnn.release_trace(device, tid)
        except Exception:
            pass
    return per_s * 1000.0, path


class _LegacyStage:
    """Wrap a legacy single-step adapter (PipelineDecodeAdapter: .step()) as a stage."""

    def __init__(self, adapter):
        # The legacy adapter IS the decode contract -- PipelineDecodeAdapter raises NotTraceCapable
        # without decode_step(state) -- so one token per call is a fact here, not a guess.
        self.name = "decode"
        self.items = 1
        self.recurring = True
        self.step = adapter.step
        self.self_traced = bool(getattr(adapter, "self_traced", False))
        self.trace_path = getattr(adapter, "trace_path", None)


def _checkpoint_for_census(pipeline=None):
    """Where this model's weights are, for the census's checkpoint-name vocabulary. None if unknown.

    FOUND FROM THE PIPELINE ITSELF. The obvious source is an env var naming the model root, and the
    harness does not export one -- checked against a live run's whole process tree: no
    PERF_MCP_MODEL_ROOT anywhere in it. A fix resting on that would have been inert and looked
    installed.

    The object being measured knows where it lives: its class's module file sits inside the model's
    own directory, and walking up from there reaches the root whose source names the hub repo. The
    walk stops at a repository boundary so this can never wander up and scan a whole monorepo.
    """
    import os as _os
    import sys as _sys
    from pathlib import Path as _Path

    cands = []
    root = (_os.environ.get("PERF_MCP_MODEL_ROOT") or _os.environ.get("TT_PERF_MODEL_ROOT") or "").strip()
    if root:
        cands.append(_Path(root))
    _mod = _sys.modules.get(type(pipeline).__module__) if pipeline is not None else None
    _f = getattr(_mod, "__file__", None) if _mod else None
    if _f:
        for _par in list(_Path(_f).resolve().parents)[:4]:
            if (_par / ".git").exists():
                break  # the repository root: everything above the model, and far too big to scan
            cands.append(_par)
    for _c in cands:
        try:
            from .stack_survey import model_id_from_source

            _mid = str(model_id_from_source(_c) or "").strip()
            if _mid:
                return _mid
        except Exception:  # noqa: BLE001
            continue
    return None


class _StdoutLines:
    """While active, every complete line printed to sys.stdout is also kept in `.lines`; output is
    unchanged. Python-level prints only -- which is all this module's rows are."""

    def __enter__(self):
        self.lines, self._part, self._out = [], "", sys.stdout
        sys.stdout = self
        return self

    def write(self, text):
        self._out.write(text)
        self._part += text
        while "\n" in self._part:
            line, self._part = self._part.split("\n", 1)
            self.lines.append(line)
        return len(text)

    def flush(self):
        self._out.flush()

    def __getattr__(self, name):
        return getattr(self._out, name)

    def __exit__(self, *exc):
        sys.stdout = self._out
        if self._part:
            self.lines.append(self._part)
        return False


def measure_adapter(adapter, device) -> float:
    """Trace-replay per-stage latency for WHATEVER the pipeline emitted. Traces every stage in
    adapter.stages; prints TRACE_STAGE_MS[<stage>] per stage and TRACE_PER_TOKEN_MS (the AR/decode
    stage if present, else the whole-pipeline sum). Returns that headline ms.

    Raises (propagating to the perf test's guard) if the pipeline has no traceable step at all."""
    # setup() builds the pipeline + binds stages. A pipeline that GENUINELY cannot trace (repeat-prefill /
    # no decode_step) raises NotTraceCapable — the ONE legitimate eager terminal. Emit a STABLE marker so
    # the generation-time validator can tell this apart from an incidental setup bug (which it must keep
    # correcting) and accept the eager path, then re-raise so the perf test guard falls to FORWARD_WALL_MS.
    try:
        from .perf_adapter import NotTraceCapable
    except Exception:  # pragma: no cover - perf_adapter always importable alongside this module
        NotTraceCapable = ()
    # THE PIPELINE'S LOOKUP TABLES (stage_marks.pipeline_gathered_tables; stage_seams.GATHERED_TABLES_ATTR),
    # read ONCE after setup() -- before it the tensors do not exist -- and handed to both readers: the
    # census, which credits their bytes to the subtree they sit in for the estimate's marker, and the
    # per-stage hook, which charges an op reading one beside an integer index for the rows that came
    # out. One declaration, because nothing else tells a table from a weight of the same shape.
    # Unstated, every table is charged in full.
    _tables: list = []
    try:
        adapter.setup(device)
        try:
            from .stage_marks import pipeline_gathered_tables as _pipeline_gathered_tables

            _tables = _pipeline_gathered_tables(getattr(adapter, "_pipe", None) or adapter)
        except Exception:  # noqa: BLE001 -- an unstated declaration charges every table in full
            _tables = []
        # THE ONLY MOMENT THE SERVED WIDTH EXISTS. setup() has built the model and the loader has
        # decided each tensor's dtype; the checkpoint records what was on disk, and every byte rule
        # downstream is a PREDICTION of what happened here. Walk it once and state it. Best-effort:
        # a census that cannot run leaves the ceiling on its existing fallbacks rather than failing
        # a measurement run.
        try:
            from .weight_census import (
                census as _census,
                gathered_marker as _gmarker,
                marker as _cmarker,
                sections_marker as _smarker,
            )

            # WITH THE CHECKPOINT, so the split is recorded in the vocabulary stage_roots speaks.
            #
            # census() records a subtree's bytes under TWO names: the attribute it was reached
            # through, and the checkpoint section it came from. Only the second can be looked up by
            # a stage_roots entry -- and the checkpoint argument was never passed, so only the first
            # existed. Run 10, 2026-08-19, published a complete device_section_bytes keyed
            # enc_a / enc_b / lm_layers / lm_head / embed / kv / mlp / attn, with no `audio_tower`
            # and no `language_model` in it. The measured split was present, correct, and unusable:
            # every stage fell back to apportioning by the checkpoint's disk ratio, which is the
            # single input the census exists to replace.
            _c = _census(
                getattr(adapter, "_pipe", None) or adapter,
                scope="pipeline",
                checkpoint=_checkpoint_for_census(getattr(adapter, "_pipe", None)),
                gathered=_tables,
            )
            if _c.get("weight_bytes"):
                print(_cmarker(_c), flush=True)
                # The split, measured in the same walk. Printed beside the total rather than derived
                # later from the checkpoint, which is the only other place a split was available and
                # states disk precision, not served precision.
                # WHICH RESIDENT WEIGHTS ARE TABLES, from the one declaration the census was handed. The
                # census sums every weight and the ceiling assumes each is streamed once per unit --
                # true of a matmul weight, false of a lookup table, and nothing in the checkpoint
                # tells them apart (voxtral's embed_tokens and lm_head are both [131072, 3072]).
                # Silent when the pipeline declared none: no number changes.
                _gm = _gmarker(_c)
                if _gm:
                    print(_gm, flush=True)
                _sm = _smarker(_c)
                if _sm:
                    print(_sm, flush=True)
        except Exception:  # noqa: BLE001
            pass
    except NotTraceCapable as exc:
        print("TRACE_NOT_TRACE_CAPABLE=1", flush=True)
        print("TRACE_REPLAY_SKIPPED=%r" % (exc,), flush=True)
        raise

    # Report the topology the trace ACTUALLY ran on (from the mesh device the perf test opened), so the
    # full-pipeline scorecard reflects the REAL mesh (DP x TP) instead of falling back to its default.
    # A >1-chip mesh means the pipeline ran tensor-parallel (sharded). Emitted here, in the tool, so it
    # is correct for EVERY generated perf test without relying on the test to print it.
    def _mesh_dims(dev):
        s = getattr(dev, "shape", None)
        if s is not None:
            try:
                d = [int(x) for x in tuple(s)]
                if len(d) >= 2:
                    return d[0], d[1]
                if len(d) == 1:
                    return 1, d[0]
            except Exception:  # noqa: BLE001
                pass
            if hasattr(s, "num_rows") and hasattr(s, "num_cols"):
                return int(s.num_rows), int(s.num_cols)
        return None

    _dims = _mesh_dims(device)
    if _dims is None:
        try:
            from .perf_adapter import resolve_mesh_shape

            _dims = tuple(int(x) for x in resolve_mesh_shape(1, 1))
        except Exception:  # noqa: BLE001
            _dims = (1, 1)
    _dp, _tp = int(_dims[0]), int(_dims[1])
    # THE PIPELINE'S OWN SPLIT, when it states one, over the mesh's axis order. rows x cols says how
    # many chips, not which axis is tensor-parallel: Qwen-Image-Edit opens 8x4 and runs TP=8 over axis
    # 0, which this printed as TP=4 (stage_marks.pipeline_tp; stage_seams.TP_ATTR).
    try:
        from .stage_marks import pipeline_tp as _pipeline_tp

        _own_tp = _pipeline_tp(getattr(adapter, "_pipe", None) or adapter)
        if _own_tp and (_dp * _tp) % _own_tp == 0:
            _dp, _tp = (_dp * _tp) // _own_tp, _own_tp
    except Exception:  # noqa: BLE001 -- an unstated split keeps the mesh's
        pass
    # THE PIPELINE'S OWN SEQUENCE SPLIT, when it states one (stage_marks.pipeline_sp; stage_seams.SP_ATTR):
    # those rows are groups cutting one request's tokens, not replicas, so the replica count is what is
    # left of the rows once they are taken out. SP=1 when unstated, which is every pipeline until now.
    _seqp = 1
    try:
        from .stage_marks import pipeline_sp as _pipeline_sp

        _own_sp = _pipeline_sp(getattr(adapter, "_pipe", None) or adapter)
        if _own_sp > 1 and _dp % _own_sp == 0:
            _dp, _seqp = _dp // _own_sp, _own_sp
    except Exception:  # noqa: BLE001 -- an unstated split keeps every row a replica
        pass
    print("DP=%d TP=%d SP=%d shard_active=%s" % (_dp, _tp, _seqp, bool(_dp * _tp * _seqp > 1)), flush=True)

    stages = list(getattr(adapter, "stages", None) or [])
    if not stages:
        # Legacy PipelineDecodeAdapter (exposes .step(), no .stages): one decode stage.
        if not callable(getattr(adapter, "step", None)):
            raise AttributeError("adapter exposes neither .stages nor a callable .step()")
        stages = [_LegacyStage(adapter)]

    results = []
    # {stage: why} for every declared stage without a row: the adapter's, then the ones below.
    _unmeasured = dict(getattr(adapter, "unmeasured", None) or {})
    # EACH STAGE GETS ITS SHARE, not the whole budget: they run in sequence, so a stage that spends
    # everything leaves the ones after it nothing and the capture dies with rows missing.
    _stage_budget = _measurement_budget_s(len(stages))
    # A RETRY OF THIS STEP KEEPS WHAT AN EARLIER ATTEMPT MEASURED (probes.stages_kept_across_retries).
    # A board too hot to go on ends this process at a stage boundary (probes.thermal_yield); the step is
    # re-run from a cool board, and a stage already measured is replayed from its own lines rather than
    # measured again -- otherwise every attempt re-heats the board on the same stages and is stopped at
    # the same place.
    from .stage_seams import unmeasured_line as _unmeasured_line, unmeasured_why as _unmeasured_why

    _node = os.environ.get("PYTEST_CURRENT_TEST", "").rsplit(" (", 1)[0]
    try:
        from .probes import keep_stage, kept_stages

        _kept = kept_stages(_node)
    except Exception:  # noqa: BLE001 -- nothing kept is the old behaviour
        keep_stage, _kept = None, {}
    for st in stages:
        _k = _kept.get(st.name)
        if _k is not None:
            print(
                "  [trace-replay] stage %r: measured by an earlier attempt of this step (%.4f ms); "
                "kept, not measured again" % (st.name, float(_k["ms"])),
                file=sys.stderr,
                flush=True,
            )
            for _ln in _k["lines"]:
                print(_ln, flush=True)
            results.append((st.name, float(_k["ms"]), _k["path"]))
            continue
        try:
            from .probes import thermal_yield

            thermal_yield(str(getattr(st, "name", "") or ""))
        except Exception:  # noqa: BLE001 -- cooling is a courtesy, never a precondition
            pass
        with _StdoutLines() as _said:
            # ONE STAGE THAT CANNOT BE MEASURED MUST NOT COST THE OTHERS THEIR ROW. The adapter already
            # holds this line for the stage that cannot PREPARE its inputs; the loop that MEASURES them
            # had no such guard, so a raise here lost every stage rather than one -- the whole replay,
            # and with it every roofline row, for a fault in a single tower. Same rule, one step later.
            try:
                ms, path = _measure_stage(device, st, _stage_budget, tables=_tables)
            except Exception as exc:  # noqa: BLE001
                # SHORTENED THE ONE WAY THAT KEEPS THE DEVICE'S OWN SIZING (stage_seams.unmeasured_why):
                # a region too small reports the bytes it wanted well past a fixed head clip, and the
                # grow loops size themselves off this line.
                _why = _unmeasured_why(exc)
                print(
                    "  [trace-replay] stage %r could not be measured (%s: %s); it gets no row, the "
                    "others are unaffected" % (st.name, type(exc).__name__, _why),
                    file=sys.stderr,
                    flush=True,
                )
                _unmeasured[st.name] = "%s: %s" % (type(exc).__name__, _why)
                continue
            results.append((st.name, ms, path))
            print("TRACE_STAGE_MS[%s]=%.4f path=%s" % (st.name, ms, path), flush=True)
            # BESIDE THE TIME THAT MEASURED IT. The compute ceiling is 2 x params x items-in-the-unit,
            # and the run recorded that number for exactly one stage -- the parser wrote it to the key
            # "prefill", by hand -- so every other stage was priced at one item however much work it did.
            # Printed per stage, from what the pipeline declares, so a third tower can have a real
            # arithmetic ceiling instead of a placeholder. Silent when unstated: 1 stays the fallback.
            _n = int(getattr(st, "items", 0) or 0)
            if _n > 0:
                print("TRACE_STAGE_ITEMS[%s]=%d" % (st.name, _n), flush=True)
            # AND HOW MANY DATA-PARALLEL GROUPS SHARE THEM (stage_seams.SPLIT), so the ceiling prices what
            # one chip does. Printed only for a stage that is split: 1 is the reader's fallback.
            _sp = int(getattr(st, "split", 0) or 0)
            if _sp > 1:
                print("TRACE_STAGE_SPLIT[%s]=%d" % (st.name, _sp), flush=True)
            # AND HOW MANY GROUPS CUT ONE REQUEST'S TOKENS in it (stage_seams.SEQ_SPLIT), for the same reason.
            _sq = int(getattr(st, "seq_split", 0) or 0)
            if _sq > 1:
                print("TRACE_STAGE_SEQ_SPLIT[%s]=%d" % (st.name, _sq), flush=True)
            # AND HOW MANY TIMES ONE REQUEST RUNS IT (stage_seams.REPEATS): the pass below times it once.
            _rp = int(getattr(st, "repeats", 0) or 0)
            if _rp > 1:
                print("TRACE_STAGE_REPEATS[%s]=%d" % (st.name, _rp), flush=True)
        if keep_stage is not None:
            keep_stage(
                _node,
                st.name,
                ms,
                path,
                [ln for ln in _said.lines if ln.startswith(_MARK_PREFIX) and not ln.startswith(_HEARTBEAT_MARK)],
            )

    # A STAGE WITHOUT A ROW IS STATED, so the whole-pipeline sum below is never read as the whole
    # pipeline when it is not (stage_seams.UNMEASURED_MARK).
    for _un, _uw in _unmeasured.items():
        print(_unmeasured_line(_un, _uw), flush=True)

    # WHICH MODULES EACH STAGE RUNS, read from the pipeline's own code (stage_marks.stage_module_paths)
    # so perf_mcp can price each stage's compute from the weights it actually multiplies instead of
    # charging every stage the whole model. Silent when the stages cannot be matched.
    try:
        from .stage_marks import stage_module_paths as _stage_module_paths

        for _sn, _sp in _stage_module_paths(getattr(adapter, "_pipe", None) or adapter).items():
            print("TRACE_STAGE_MODULES[%s]=%s" % (_sn, ",".join(_sp)), flush=True)
    except Exception:  # noqa: BLE001
        pass

    pipeline_ms = sum(ms for _, ms, _ in results)
    # THE UNIT IS A STRUCTURAL FACT, so read the STRUCTURE, not a name. This matched
    # `"decode" in stage_name` -- a substring test on free text emit-e2e wrote -- which is a guess
    # wearing an observation's clothes: a pipeline whose recurring stage is called `generate` reads
    # as one-pass, and one that names any stage `decode` reads as autoregressive whether it loops or
    # not. The decode CONTRACT is the real signal: a pipeline exposing decode_step(state) retires one
    # token per call by definition, which is what PipelineDecodeAdapter already requires and raises
    # NotTraceCapable without. Below it a stage-adapter pipeline, which exposes per-stage hooks rather
    # than the single decode contract, is asked what one call retires. No name is read on any path.
    from models.experimental.perf_automation.agent.perf_adapter import headline_unit as _hu

    _unit = _hu([r[0] for r in results], getattr(adapter, "_pipe", None))
    # THE RECURRING STAGE, AS THE STAGE ITSELF REPORTED IT. `recurring` is set from the item count
    # the pipeline declares (one item per call is what recurring means) and unconditionally for the
    # legacy decode contract. This used to match a name first, and that was a guess in both
    # directions: a loop called `generate` read as one-pass, and any stage called `decode` read as
    # the recurring one whether it looped or not.
    _rec = {st.name for st in stages if getattr(st, "recurring", False)}
    # WHICH UNIT OF WORK THE HEADLINE MEASURES. This selection already knew -- a decode stage means
    # the number is per TOKEN, no decode stage means it is one whole pipeline pass -- but it never
    # said so, and the roofline band on the other end assumed "token" unconditionally. A step-unit
    # (diffusion) or inference-unit (classifier) model therefore had its per-unit ceiling scored
    # against whatever this printed, silently: the same per-token-vs-per-profile unit mix that once
    # made every module read ABOVE_BAND, on a different axis. Name it here, at the only place that
    # can know, so the band can refuse a mismatch instead of guessing.
    # THE HEADLINE STAGE FOLLOWS THE UNIT, which headline_unit derived without reading a name. A
    # per-item unit means one stage recurs and the headline is that stage; an inference unit means
    # nothing recurs and the headline is the whole pass. Both the stage and its unit therefore come
    # from what the pipeline stated -- `recurring` from the item count it declares, the unit from
    # PIPELINE_UNIT or the decode contract -- and neither is inferred from the word a stage is called.
    _headline = next((r for r in results if r[0] in _rec), None) if _unit != "inference" else None
    # A pipeline that stated a per-item unit but exposed no per-stage count still has a recurring
    # stage: it is the last one to run, the loop every earlier stage feeds.
    if _headline is None and _unit != "inference" and results:
        _headline = results[-1]
    if _headline is not None:
        headline_ms, headline_path, headline_unit = _headline[1], _headline[2], _unit
    else:
        headline_ms, headline_path, headline_unit = pipeline_ms, "trace+pipeline", "inference"
    batch = int(getattr(adapter, "batch", 1) or 1)
    per_s = headline_ms / 1000.0
    tokens_per_sec = (batch / per_s) if per_s > 0 else 0.0
    # THE line the harness parses (tracy_tool.py:_PER_TOKEN_RE / perf_mcp.py). Keep the name verbatim.
    print("TRACE_PER_TOKEN_MS=%.4f" % headline_ms, flush=True)
    print("TRACE_HEADLINE_UNIT=%s" % headline_unit, flush=True)
    print(
        "TRACE_PIPELINE_MS=%.4f TRACE_STAGES=%d%s"
        % (pipeline_ms, len(results), "" if _headline is not None else " (no recurring stage: headline=pipeline sum)"),
        flush=True,
    )
    print(
        "TRACE_REPLAY_PATH=%s TRACE_TOKENS_PER_SEC=%.2f batch=%d warmup=%d iters=%d"
        % (headline_path, tokens_per_sec, batch, _WARMUP_ITERS, _REPLAY_ITERS),
        flush=True,
    )
    return headline_ms
