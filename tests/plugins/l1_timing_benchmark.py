# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Measurement-only pytest plugin for graded-run L1 profiling overhead.

Select a mode with ``L1_TIMING_MODE``:

* baseline: test call only, program cache unchanged
* pc_off: disable/clear the program cache around every test call
* capture: pc_off plus a C++ NORMAL graph capture; do not reduce the trace
* full: capture plus extract_resource_usage_per_core
* full_python: capture plus an equivalent direct Python reduction (avoids a Python -> JSON -> C++ round trip)

Append one aggregate JSON object per process to ``L1_TIMING_OUTPUT``.
"""

from __future__ import annotations

import json
import os
import statistics
import time

import pytest


_MODE = os.environ.get("L1_TIMING_MODE", "baseline")
_OUTPUT = os.environ.get("L1_TIMING_OUTPUT", "/tmp/l1_timing_results.jsonl")
_VALID_MODES = {"baseline", "pc_off", "capture", "full", "full_python"}
_ROWS: list[dict[str, int | str]] = []


def _device(item):
    args = getattr(item, "funcargs", None) or {}
    return args.get("device") or args.get("mesh_device")


def _ns() -> int:
    return time.perf_counter_ns()


def _extract_peak_total_python(trace) -> int:
    """The production reducer's accounting, kept minimal for bridge-cost measurement."""
    cb = dataflow = scratchpad = total = peak = 0
    for node in trace:
        kind = node["node_type"]
        params = node.get("params", {})
        delta = 0
        if kind == "circular_buffer_allocate" and int(params["globally_allocated"]) != 1:
            delta = int(params["size"])
            cb += delta
        elif kind == "dataflow_buffer_allocate" and int(params["borrows_memory"]) != 1:
            delta = int(params["size"])
            dataflow += delta
        elif kind == "scratchpad_allocate":
            delta = int(params["size"])
            scratchpad += delta
        elif kind == "circular_buffer_deallocate_all":
            delta = -(cb + dataflow + scratchpad)
            cb = dataflow = scratchpad = 0
        elif kind in {"buffer_allocate", "buffer_deallocate"} and params["type"] != "DRAM":
            delta = int(params["max_size_per_bank"])
            if kind == "buffer_deallocate":
                delta = -delta
        total += delta
        peak = max(peak, total)
    return peak


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    if os.environ.get("UP_FRONT_COLLECT") == "1" or _MODE not in _VALID_MODES:
        return (yield)

    device = _device(item)
    if device is None:
        return (yield)

    import ttnn

    row: dict[str, int | str] = {
        "nodeid": item.nodeid,
        "disable_ns": 0,
        "begin_ns": 0,
        "body_ns": 0,
        "end_ns": 0,
        "reduce_ns": 0,
        "enable_ns": 0,
        "nodes": 0,
    }
    trace = None
    total_start = _ns()

    if _MODE != "baseline":
        start = _ns()
        device.disable_and_clear_program_cache()
        row["disable_ns"] = _ns() - start

    if _MODE in {"capture", "full", "full_python"}:
        # Call the C++ binding directly. L1 accounting does not need Python argument/I/O
        # recording, which the public report-oriented wrapper also enables.
        start = _ns()
        ttnn.graph._cpp_begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        row["begin_ns"] = _ns() - start

    start = _ns()
    try:
        result = yield
    finally:
        row["body_ns"] = _ns() - start

        if _MODE in {"capture", "full", "full_python"}:
            start = _ns()
            trace = ttnn.graph._cpp_end_graph_capture()
            row["end_ns"] = _ns() - start
            row["nodes"] = len(trace)

        if _MODE == "full":
            start = _ns()
            usage = ttnn.graph.extract_resource_usage_per_core(trace)
            row["reduce_ns"] = _ns() - start
            row["peak_total"] = int(usage.peak_total)
        elif _MODE == "full_python":
            start = _ns()
            row["peak_total"] = _extract_peak_total_python(trace)
            row["reduce_ns"] = _ns() - start

        if _MODE != "baseline":
            start = _ns()
            device.enable_program_cache()
            row["enable_ns"] = _ns() - start

        row["wrapper_ns"] = _ns() - total_start
        row["program_cache_entries"] = int(device.num_program_cache_entries())
        _ROWS.append(row)

    return result


def _summary(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "sum_ms": sum(values) / 1e6,
        "mean_us": statistics.fmean(values) / 1e3,
        "median_us": statistics.median(values) / 1e3,
        "p95_us": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))] / 1e3,
    }


def pytest_sessionfinish(session, exitstatus):
    if not _ROWS:
        return

    fields = ("disable_ns", "begin_ns", "body_ns", "end_ns", "reduce_ns", "enable_ns", "wrapper_ns")
    result = {
        "mode": _MODE,
        "exitstatus": exitstatus,
        "cases": len(_ROWS),
        "nodes_per_case": {
            "min": min(int(row["nodes"]) for row in _ROWS),
            "median": statistics.median(int(row["nodes"]) for row in _ROWS),
            "max": max(int(row["nodes"]) for row in _ROWS),
        },
        "max_program_cache_entries": max(int(row["program_cache_entries"]) for row in _ROWS),
        **{field.removesuffix("_ns"): _summary([int(row[field]) for row in _ROWS]) for field in fields},
    }
    if _MODE in {"full", "full_python"}:
        result["nonzero_l1_cases"] = sum(int(row.get("peak_total", 0)) > 0 for row in _ROWS)

    with open(_OUTPUT, "a", encoding="utf-8") as output:
        output.write(json.dumps(result, sort_keys=True) + "\n")
    print("L1_TIMING_RESULT " + json.dumps(result, sort_keys=True), flush=True)
