# SPDX-License-Identifier: Apache-2.0
"""Model-agnostic trace-replay latency measurement for optimize/perf.

`measure_adapter(adapter, device, mode="auto")` drives a PipelineStageAdapter (see perf_adapter.py):
for EACH stage emit-e2e emitted (adapter.stages, from the pipeline's PIPELINE_STAGES) it captures one
steady-state, host-op-free step as a device trace and replays it under trace — optionally with a
second command queue overlapping host<->device I/O with compute for stages that stage their inputs
(`<stage>_write_inputs`). It prints, per stage:

    TRACE_STAGE_MS[<stage>]=<float> path=trace+2cq|trace+1cq

plus the headline clean, GPU-comparable wall the harness parses:

    TRACE_PER_TOKEN_MS=<float>     (the AR/decode stage if present, else the whole-pipeline sum)

which the harness (agent/tracy_tool.py + cc_optimize/perf_mcp.py) reads as the `trace` metric source
(vs the `eager` fallback FORWARD_WALL_MS). This is the companion of perf_adapter.py: the adapter is
the shell (setup + per-stage step/write), this module is the engine (warmup -> capture -> timed
replay -> emit the numbers). A legacy single-step adapter (PipelineDecodeAdapter, no .stages) is
wrapped as one "decode" stage, so the old path is unchanged.

Modes:
    "auto"  (default) : 2-CQ path IFF the pipeline exposes decode_write_inputs (adapter.write_inputs is
                        set in setup) AND the device was opened with >=2 command queues; else single-CQ.
    "trace" / "1cq"   : force single-CQ trace replay.
    "2cq"             : force the 2-CQ path (I/O on cq1 overlapping compute on cq0); auto-degrades to
                        single-CQ if the device has <2 command queues or the 2-CQ path errors.

Caller contract (the generated perf test): call inside the `if _PERF_TRACE:` block, on the SAME
device the test opened WITH `trace_region_size` (and `num_command_queues=2` for the 2-CQ path). Any
failure here (notably a repeat-prefill pipeline with no `decode_step`, which raises in
`adapter.setup`) propagates out; the perf test's guard catches it, prints `TRACE_REPLAY_SKIPPED=...`,
and falls back to FORWARD_WALL_MS.
"""
from __future__ import annotations

import os
import time

import ttnn

# Warmup compiles kernels + populates RoPE/mask/KV caches (trace capture can neither compile nor
# upload from host); replay iters are averaged for a stable per-token number. Both env-tunable.
_WARMUP_ITERS = max(1, int(os.environ.get("TT_TRACE_WARMUP_ITERS", "3")))
_REPLAY_ITERS = max(1, int(os.environ.get("TT_TRACE_REPLAY_ITERS", "16")))


def _num_command_queues(device) -> int | None:
    """Best-effort read of how many CQs the device was opened with. None = unknown."""
    for attr in ("num_command_queues", "num_hw_cqs"):
        fn = getattr(device, attr, None)
        try:
            if callable(fn):
                return int(fn())
            if isinstance(fn, int):
                return int(fn)
        except Exception:
            pass
    return None


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


def _replay_2cq(device, tid, write, iters):
    """Overlap the next step's input upload (staged on cq1 by <stage>_write_inputs) with the traced
    compute on cq0, synchronized with events — the canonical trace + 2-CQ loop."""
    t0 = time.perf_counter()
    for _ in range(iters):
        if callable(write):
            write()  # host->device staged on cq1 (pipeline's <stage>_write_inputs)
            ev = ttnn.record_event(device, 1)  # signal once the cq1 write is enqueued
            ttnn.wait_for_event(0, ev)  # cq0 waits for inputs before running the traced step
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) / iters


def _want_2cq(mode, has_write, ncq):
    if mode in ("trace", "1cq"):
        return False
    want = True if mode == "2cq" else bool(has_write)  # auto: 2CQ iff the stage stages inputs
    if want and ncq is not None and ncq < 2:
        return False  # device opened single-CQ -> can't overlap
    return want


def _measure_native(device, stage, ncq):
    """Time a SELF-TRACED stage: the pipeline owns its trace capture + CQ1 input-staging (persistent-
    buffer / vLLM-style decode, e.g. GLM's decode(enable_trace=True)), so we must NOT begin_trace_capture
    around it -- doing so raises "Writes/Reads are not supported during trace capture" because the step
    does host<->device I/O + execute_trace internally. Instead warm it (the pipeline lazily captures its
    own trace on the first call) and time steady-state replays. The pipeline reports its real path via an
    optional trace_path(); default by CQ count."""
    for _ in range(_WARMUP_ITERS):
        stage.step()
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
        except Exception:
            path = "trace+2cq" if (ncq is None or ncq >= 2) else "trace+1cq"
    else:
        path = "trace+2cq" if (ncq is None or ncq >= 2) else "trace+1cq"
    return per_s * 1000.0, path


def _measure_stage(device, stage, mode, ncq):
    """Capture stage.step as a trace, replay it (2CQ if it stages inputs), return (ms, path)."""
    if getattr(stage, "self_traced", False):
        return _measure_native(device, stage, ncq)
    tid = _capture_step_trace(device, stage.step)
    try:
        if _want_2cq(mode, stage.write is not None, ncq):
            try:
                per_s = _replay_2cq(device, tid, stage.write, _REPLAY_ITERS)
                path = "trace+2cq"
            except Exception as exc:  # any 2-CQ / event issue -> degrade, never fail the measurement
                print("TRACE_2CQ_FALLBACK[%s]=%r" % (stage.name, exc), flush=True)
                per_s = _replay_1cq(device, tid, _REPLAY_ITERS)
                path = "trace+1cq"
        else:
            per_s = _replay_1cq(device, tid, _REPLAY_ITERS)
            path = "trace+1cq"
    finally:
        try:
            ttnn.release_trace(device, tid)
        except Exception:
            pass
    return per_s * 1000.0, path


class _LegacyStage:
    """Wrap a legacy single-step adapter (PipelineDecodeAdapter: .step()/.write_inputs) as a stage."""

    def __init__(self, adapter):
        self.name = "decode"
        self.step = adapter.step
        w = getattr(adapter, "write_inputs", None)
        self.write = w if callable(w) else None
        self.self_traced = bool(getattr(adapter, "self_traced", False))
        self.trace_path = getattr(adapter, "trace_path", None)


def measure_adapter(adapter, device, mode: str = "auto") -> float:
    """Trace-replay per-stage latency for WHATEVER the pipeline emitted. Traces (+2CQ) every stage in
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

    stages = list(getattr(adapter, "stages", None) or [])
    if not stages:
        # Legacy PipelineDecodeAdapter (exposes .step()/.write_inputs, no .stages): one decode stage.
        if not callable(getattr(adapter, "step", None)):
            raise AttributeError("adapter exposes neither .stages nor a callable .step()")
        stages = [_LegacyStage(adapter)]

    ncq = _num_command_queues(device)
    results = []
    for st in stages:
        ms, path = _measure_stage(device, st, mode, ncq)
        results.append((st.name, ms, path))
        print("TRACE_STAGE_MS[%s]=%.4f path=%s" % (st.name, ms, path), flush=True)

    pipeline_ms = sum(ms for _, ms, _ in results)
    decode = next((r for r in results if "decode" in r[0].lower()), None)
    headline_ms = decode[1] if decode else pipeline_ms
    headline_path = decode[2] if decode else "trace+pipeline"
    batch = int(getattr(adapter, "batch", 1) or 1)
    per_s = headline_ms / 1000.0
    tokens_per_sec = (batch / per_s) if per_s > 0 else 0.0
    # THE line the harness parses (tracy_tool.py:_PER_TOKEN_RE / perf_mcp.py). Keep the name verbatim.
    print("TRACE_PER_TOKEN_MS=%.4f" % headline_ms, flush=True)
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
