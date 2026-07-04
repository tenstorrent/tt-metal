# SPDX-License-Identifier: Apache-2.0
"""Model-agnostic trace-replay per-token latency measurement for optimize/perf.

`measure_adapter(adapter, device, mode="auto")` drives a PerfAdapter (see perf_adapter.py) through
the standard decode contract, captures ONE steady-state decode step as a device trace, replays it
under trace (optionally with a second command queue overlapping host<->device I/O with compute),
and prints the clean, GPU-comparable per-token wall as:

    TRACE_PER_TOKEN_MS=<float>

which the harness (agent/tracy_tool.py + cc_optimize/perf_mcp.py) reads as the `trace` metric source
(vs the `eager` fallback FORWARD_WALL_MS). This is the missing companion of perf_adapter.py: the
adapter is the shell (setup/step/write_inputs), this module is the engine (warmup -> capture ->
timed replay -> emit the number).

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


def _capture_decode_trace(device, adapter):
    """Warm up, then capture exactly one host-op-free, fixed-shape decode step as a trace on cq0."""
    for _ in range(_WARMUP_ITERS):
        adapter.step()
    ttnn.synchronize_device(device)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    adapter.step()
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    return tid


def _replay_1cq(device, tid, iters):
    t0 = time.perf_counter()
    for _ in range(iters):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) / iters


def _replay_2cq(device, tid, adapter, iters):
    """Overlap the next step's input upload (staged on cq1 by decode_write_inputs) with the traced
    compute on cq0, synchronized with events — the canonical trace + 2-CQ decode loop."""
    write = getattr(adapter, "write_inputs", None)
    t0 = time.perf_counter()
    for _ in range(iters):
        if callable(write):
            write()  # host->device staged on cq1 (pipeline's decode_write_inputs)
            ev = ttnn.record_event(device, 1)  # signal once the cq1 write is enqueued
            ttnn.wait_for_event(0, ev)  # cq0 waits for inputs before running the traced step
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) / iters


def measure_adapter(adapter, device, mode: str = "auto") -> float:
    """Trace-replay per-token decode latency. Prints TRACE_PER_TOKEN_MS=<ms>; returns ms/token.

    Raises (propagating to the perf test's guard) if the pipeline has no cached decode_step."""
    # setup() builds the pipeline + prefills; raises AttributeError for repeat-prefill pipelines.
    adapter.setup(device)

    has_2cq_hook = hasattr(adapter, "write_inputs")
    if mode in ("trace", "1cq"):
        want_2cq = False
    elif mode == "2cq":
        want_2cq = True
    else:  # auto
        want_2cq = has_2cq_hook
    ncq = _num_command_queues(device)
    if want_2cq and ncq is not None and ncq < 2:
        want_2cq = False  # device opened single-CQ -> can't overlap; use single-CQ trace

    tid = _capture_decode_trace(device, adapter)
    try:
        if want_2cq:
            try:
                per_token_s = _replay_2cq(device, tid, adapter, _REPLAY_ITERS)
                path = "trace+2cq"
            except Exception as exc:  # any 2-CQ / event issue -> degrade, never fail the measurement
                print("TRACE_2CQ_FALLBACK=%r" % (exc,), flush=True)
                per_token_s = _replay_1cq(device, tid, _REPLAY_ITERS)
                path = "trace+1cq"
        else:
            per_token_s = _replay_1cq(device, tid, _REPLAY_ITERS)
            path = "trace+1cq"
    finally:
        try:
            ttnn.release_trace(device, tid)
        except Exception:
            pass

    per_token_ms = per_token_s * 1000.0
    batch = int(getattr(adapter, "batch", 1) or 1)
    tokens_per_sec = (batch / per_token_s) if per_token_s > 0 else 0.0
    # THE line the harness parses (tracy_tool.py:_PER_TOKEN_RE / perf_mcp.py). Keep the name verbatim.
    print("TRACE_PER_TOKEN_MS=%.4f" % per_token_ms, flush=True)
    print(
        "TRACE_REPLAY_PATH=%s TRACE_TOKENS_PER_SEC=%.2f batch=%d warmup=%d iters=%d"
        % (path, tokens_per_sec, batch, _WARMUP_ITERS, _REPLAY_ITERS),
        flush=True,
    )
    return per_token_ms
