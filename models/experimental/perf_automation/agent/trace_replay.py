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

import os
import time

import ttnn

# Warmup compiles kernels + populates RoPE/mask/KV caches (trace capture can neither compile nor
# upload from host); replay iters are averaged for a stable per-token number. Both env-tunable.
_WARMUP_ITERS = max(1, int(os.environ.get("TT_TRACE_WARMUP_ITERS", "3")))
_REPLAY_ITERS = max(1, int(os.environ.get("TT_TRACE_REPLAY_ITERS", "16")))


def _capture_step_trace(device, step):
    """Warm up, then capture exactly one host-op-free, fixed-shape step as a trace on cq0."""
    for _ in range(_WARMUP_ITERS):
        step()
    ttnn.synchronize_device(device)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    step()
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    return tid


def _replay_1cq(device, tid, iters):
    t0 = time.perf_counter()
    for _ in range(iters):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) / iters


# A REPLAYED TRACE DISPATCHES ONE OP. An eager pass dispatches one per ttnn call in the model --
# 3,564 of them for a 48-layer gemma-3 prefill. Nothing else about the two paths differs by three
# orders of magnitude, so the dispatch count is the signal, and anything above this is eager.
_TRACED_OP_DISPATCH_MAX = int(os.environ.get("TT_TRACE_DISPATCH_MAX", "32"))


def _count_op_dispatches(fn):
    """Run `fn` once and return how many ttnn ops it dispatched from PYTHON, or None if the counter
    could not be installed.

    Counting is wrapped around a call the caller was going to make anyway (the last warmup), so this
    adds no execution: no second pass, no eager run beside the traced one. The point is only to learn
    which path the step took."""
    try:
        from ttnn import decorators as _dec

        target = _dec.Operation
        orig = target.__call__
    except Exception:  # noqa: BLE001
        fn()
        return None
    n = [0]

    def counting(self, *a, **kw):
        n[0] += 1
        return orig(self, *a, **kw)

    target.__call__ = counting
    try:
        fn()
    finally:
        target.__call__ = orig
    return n[0]


def _measure_native(device, stage):
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
    for _ in range(max(0, _WARMUP_ITERS - 1)):
        stage.step()
    dispatched = _count_op_dispatches(stage.step)
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(_REPLAY_ITERS):
        stage.step()
    ttnn.synchronize_device(device)
    per_s = (time.perf_counter() - t0) / _REPLAY_ITERS
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
    return per_s * 1000.0, path


def _measure_stage(device, stage):
    """Capture stage.step as a trace, replay it on a single command queue, return (ms, path)."""
    if getattr(stage, "self_traced", False):
        return _measure_native(device, stage)
    tid = _capture_step_trace(device, stage.step)
    try:
        per_s = _replay_1cq(device, tid, _REPLAY_ITERS)
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
        self.name = "decode"
        self.step = adapter.step
        self.self_traced = bool(getattr(adapter, "self_traced", False))
        self.trace_path = getattr(adapter, "trace_path", None)


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
    try:
        adapter.setup(device)
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
    print("DP=%d TP=%d shard_active=%s" % (_dp, _tp, bool(_dp * _tp > 1)), flush=True)

    stages = list(getattr(adapter, "stages", None) or [])
    if not stages:
        # Legacy PipelineDecodeAdapter (exposes .step(), no .stages): one decode stage.
        if not callable(getattr(adapter, "step", None)):
            raise AttributeError("adapter exposes neither .stages nor a callable .step()")
        stages = [_LegacyStage(adapter)]

    results = []
    for st in stages:
        ms, path = _measure_stage(device, st)
        results.append((st.name, ms, path))
        print("TRACE_STAGE_MS[%s]=%.4f path=%s" % (st.name, ms, path), flush=True)

    pipeline_ms = sum(ms for _, ms, _ in results)
    # THE UNIT IS A STRUCTURAL FACT, so read the STRUCTURE, not a name. This matched
    # `"decode" in stage_name` -- a substring test on free text emit-e2e wrote -- which is a guess
    # wearing an observation's clothes: a pipeline whose recurring stage is called `generate` reads
    # as one-pass, and one that names any stage `decode` reads as autoregressive whether it loops or
    # not. The decode CONTRACT is the real signal: a pipeline exposing decode_step(state) retires one
    # token per call by definition, which is what PipelineDecodeAdapter already requires and raises
    # NotTraceCapable without. The name match stays BELOW it, for stage-adapter pipelines that expose
    # per-stage hooks rather than the single decode contract.
    from models.experimental.perf_automation.agent.perf_adapter import headline_unit as _hu

    _unit = _hu([r[0] for r in results], getattr(adapter, "_pipe", None))
    decode = next((r for r in results if "decode" in r[0].lower()), None)
    if _unit == "token" and decode is None and results:
        decode = results[-1]
    # WHICH UNIT OF WORK THE HEADLINE MEASURES. This selection already knew -- a decode stage means
    # the number is per TOKEN, no decode stage means it is one whole pipeline pass -- but it never
    # said so, and the roofline band on the other end assumed "token" unconditionally. A step-unit
    # (diffusion) or inference-unit (classifier) model therefore had its per-unit ceiling scored
    # against whatever this printed, silently: the same per-token-vs-per-profile unit mix that once
    # made every module read ABOVE_BAND, on a different axis. Name it here, at the only place that
    # can know, so the band can refuse a mismatch instead of guessing.
    step = next((r for r in results if any(k in r[0].lower() for k in ("step", "denoise", "diffus"))), None)
    if decode:
        headline_ms, headline_path, headline_unit = decode[1], decode[2], "token"
    elif step:
        headline_ms, headline_path, headline_unit = step[1], step[2], "step"
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
        % (pipeline_ms, len(results), "" if decode else " (no decode stage: per-token=pipeline sum)"),
        flush=True,
    )
    print(
        "TRACE_REPLAY_PATH=%s TRACE_TOKENS_PER_SEC=%.2f batch=%d warmup=%d iters=%d"
        % (headline_path, tokens_per_sec, batch, _WARMUP_ITERS, _REPLAY_ITERS),
        flush=True,
    )
    return headline_ms
