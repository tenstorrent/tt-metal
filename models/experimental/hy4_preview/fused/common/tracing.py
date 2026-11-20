# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Traced-replay timing shared by the fused pieces: capture a function as a trace, replay it back to back with one
synchronize at the end (the measurement tests/test_perf_decode.py uses)."""

import time

import ttnn


def sync(mesh_device) -> float:
    ttnn.synchronize_device(mesh_device)
    return time.perf_counter()


def idle_sync_ms(mesh_device, repeats: int = 20) -> float:
    start = sync(mesh_device)
    for _ in range(repeats):
        ttnn.synchronize_device(mesh_device)
    return 1e3 * (time.perf_counter() - start) / repeats


def capture(mesh_device, fn):
    """Capture ``fn()`` as a trace; returns (trace id, what fn returned). ``fn`` must already have run once (its
    programs compiled) and must not read or write from the host."""
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    try:
        result = fn()
    finally:
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    return trace_id, result


def replay_ms(mesh_device, trace_id, repeats: int = 20) -> float:
    """ms per replay: ``repeats`` replays enqueued back to back, one synchronize at the end."""
    start = sync(mesh_device)
    for _ in range(repeats):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    return 1e3 * (sync(mesh_device) - start) / repeats


def traced_ms(mesh_device, fn, repeats: int = 20, rounds: int = 3, calls: int = 1) -> dict:
    """ms per call of ``fn`` in a trace of ``calls`` calls: run it once untraced, capture, replay once blocking,
    then time ``rounds`` rounds of ``repeats`` replays. Returns {"replay_ms": the best round, "rounds_ms": all of
    them}, both per call.

    One call per trace is what a stage costs as a trace of its own: every replay sends the program's kernel
    binaries again. Many calls per trace is what it costs inside a long trace, where they are sent once
    (tests/test_perf_decode.py measures its per-op floors that way, 32 calls per trace)."""

    def body():
        for _ in range(calls):
            fn()

    fn()
    ttnn.synchronize_device(mesh_device)
    trace_id, _ = capture(mesh_device, body)
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        times = [replay_ms(mesh_device, trace_id, repeats) / calls for _ in range(rounds)]
    finally:
        ttnn.release_trace(mesh_device, trace_id)
    return {"replay_ms": min(times), "rounds_ms": times}
