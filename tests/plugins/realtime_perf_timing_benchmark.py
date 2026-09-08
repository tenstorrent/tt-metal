# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Measurement-only pytest plugin for session-wide real-time profiler collection."""

from __future__ import annotations

import json
import os
import statistics
import threading
import time

import pytest


_MODE = os.environ.get("RT_PERF_TIMING_MODE", "stream")
_OUTPUT = os.environ.get("RT_PERF_TIMING_OUTPUT", "/tmp/rt_perf_timing_results.jsonl")
_CALL_NS: list[int] = []
_WAIT_NS: list[int] = []
_CALLBACK_NS: list[int] = []
_BATCH_SIZES: list[int] = []
_RECORDS: list[dict] = []
_DROPPED = 0
_HANDLE = None
_CONDITION = threading.Condition()
_ACTIVE = False
_TIMED_OUT = 0
_COLLECT = None
_REGISTER_NS = 0


def _ns() -> int:
    return time.perf_counter_ns()


def _summary(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "sum_ms": 0.0, "mean_us": 0.0, "median_us": 0.0, "p95_us": 0.0}
    ordered = sorted(values)
    return {
        "count": len(values),
        "sum_ms": sum(values) / 1e6,
        "mean_us": statistics.fmean(values) / 1e3,
        "median_us": statistics.median(values) / 1e3,
        "p95_us": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))] / 1e3,
    }


def pytest_sessionstart(session):
    if os.environ.get("UP_FRONT_COLLECT") == "1":
        return

    def collect(batch):
        global _DROPPED
        callback_start = _ns()
        copied = []
        for record in batch.records:
            start = int(record.start_timestamp)
            end = int(record.end_timestamp)
            frequency = float(record.frequency)
            copied.append(
                {
                    "runtime_id": int(record.runtime_id),
                    "chip_id": int(record.chip_id),
                    "duration_ns": (end - start) / frequency if frequency > 0 and end >= start else None,
                    "kernel_sources": tuple(str(source) for source in record.kernel_sources),
                }
            )
        with _CONDITION:
            _DROPPED += int(batch.dropped)
            _RECORDS.extend(copied)
            _BATCH_SIZES.append(len(copied))
            _CALLBACK_NS.append(_ns() - callback_start)
            _CONDITION.notify_all()

    global _COLLECT
    _COLLECT = collect


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_call(item):
    if os.environ.get("UP_FRONT_COLLECT") == "1":
        return (yield)

    if _MODE == "off":
        start = _ns()
        try:
            return (yield)
        finally:
            _CALL_NS.append(_ns() - start)

    import ttnn

    global _ACTIVE, _HANDLE, _REGISTER_NS, _TIMED_OUT
    if _HANDLE is None:
        register_start = _ns()
        _ACTIVE = bool(ttnn.device.IsProgramRealtimeProfilerActive())
        _HANDLE = ttnn.device.RegisterProgramRealtimeProfilerCallback(_COLLECT)
        _REGISTER_NS = _ns() - register_start

    start = _ns()
    try:
        return (yield)
    finally:
        if _MODE == "wait_per_case":
            target = len(_CALL_NS) + 1
            wait_start = _ns()
            deadline = time.monotonic() + 1.0
            with _CONDITION:
                while len(_RECORDS) < target:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        _TIMED_OUT += 1
                        break
                    _CONDITION.wait(remaining)
            _WAIT_NS.append(_ns() - wait_start)
        _CALL_NS.append(_ns() - start)


def pytest_sessionfinish(session, exitstatus):
    if os.environ.get("UP_FRONT_COLLECT") == "1":
        return

    import ttnn

    unregister_start = _ns()
    if _HANDLE is not None:
        ttnn.device.UnregisterProgramRealtimeProfilerCallback(_HANDLE)
    unregister_ns = _ns() - unregister_start

    valid = [record for record in _RECORDS if record["runtime_id"] and record["duration_ns"] is not None]
    source_sets = sorted(
        {tuple(sorted(source.rsplit("/", 1)[-1] for source in record["kernel_sources"])) for record in valid}
    )
    result = {
        "mode": _MODE,
        "exitstatus": exitstatus,
        "cases": len(_CALL_NS),
        "active": _ACTIVE,
        "records": len(_RECORDS),
        "valid_records": len(valid),
        "dropped": _DROPPED,
        "timed_out": _TIMED_OUT,
        "call": _summary(_CALL_NS),
        "wait": _summary(_WAIT_NS),
        "callback": _summary(_CALLBACK_NS),
        "callback_batches": len(_BATCH_SIZES),
        "mean_records_per_batch": statistics.fmean(_BATCH_SIZES) if _BATCH_SIZES else 0.0,
        "register_ms": _REGISTER_NS / 1e6,
        "unregister_ms": unregister_ns / 1e6,
        "duration_ns": _summary([int(record["duration_ns"]) for record in valid]),
        "source_sets": source_sets,
    }
    with open(_OUTPUT, "a", encoding="utf-8") as output:
        output.write(json.dumps(result, sort_keys=True) + "\n")
    print("RT_PERF_TIMING_RESULT " + json.dumps(result, sort_keys=True), flush=True)
