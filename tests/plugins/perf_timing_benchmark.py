# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Measurement-only timers for the eval per-test device performance profiler."""

from __future__ import annotations

import json
import os
import statistics
import time

import pytest


_MODE = os.environ.get("PERF_TIMING_MODE", "off")
_OUTPUT = os.environ.get("PERF_TIMING_OUTPUT", "/tmp/perf_timing_results.jsonl")
_CALL_NS: list[int] = []
_FLUSH_NS: list[int] = []
_CAPTURE_NS: list[int] = []
_READ_DEVICE_FLUSH_NS: list[int] = []
_READ_DEVICE_CAPTURE_NS: list[int] = []
_GET_LATEST_NS: list[int] = []
_PHASE = ""


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

    import ttnn
    from eval import profiling

    original_read_device = ttnn.ReadDeviceProfiler
    original_get_latest = ttnn.get_latest_programs_perf_data
    original_flush = profiling.flush_device_profiler
    original_capture = profiling.read_device_perf

    def timed_read_device(device):
        start = _ns()
        try:
            return original_read_device(device)
        finally:
            elapsed = _ns() - start
            if _PHASE == "flush":
                _READ_DEVICE_FLUSH_NS.append(elapsed)
            elif _PHASE == "capture":
                _READ_DEVICE_CAPTURE_NS.append(elapsed)

    def timed_get_latest():
        start = _ns()
        try:
            return original_get_latest()
        finally:
            _GET_LATEST_NS.append(_ns() - start)

    def timed_flush(device):
        global _PHASE
        start = _ns()
        _PHASE = "flush"
        try:
            return original_flush(device)
        finally:
            _PHASE = ""
            _FLUSH_NS.append(_ns() - start)

    def timed_capture(device):
        global _PHASE
        start = _ns()
        _PHASE = "capture"
        try:
            return original_capture(device)
        finally:
            _PHASE = ""
            _CAPTURE_NS.append(_ns() - start)

    ttnn.ReadDeviceProfiler = timed_read_device
    ttnn.get_latest_programs_perf_data = timed_get_latest
    profiling.flush_device_profiler = timed_flush
    profiling.read_device_perf = timed_capture


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_call(item):
    if os.environ.get("UP_FRONT_COLLECT") == "1":
        return (yield)
    start = _ns()
    try:
        return (yield)
    finally:
        _CALL_NS.append(_ns() - start)


def pytest_sessionfinish(session, exitstatus):
    if os.environ.get("UP_FRONT_COLLECT") == "1":
        return
    result = {
        "mode": _MODE,
        "exitstatus": exitstatus,
        "cases": len(_CALL_NS),
        "call": _summary(_CALL_NS),
        "flush": _summary(_FLUSH_NS),
        "capture": _summary(_CAPTURE_NS),
        "read_device_flush": _summary(_READ_DEVICE_FLUSH_NS),
        "read_device_capture": _summary(_READ_DEVICE_CAPTURE_NS),
        "get_latest": _summary(_GET_LATEST_NS),
    }
    with open(_OUTPUT, "a", encoding="utf-8") as output:
        output.write(json.dumps(result, sort_keys=True) + "\n")
    print("PERF_TIMING_RESULT " + json.dumps(result, sort_keys=True), flush=True)
