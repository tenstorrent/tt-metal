#!/usr/bin/env python3
"""Collect one signpost-bounded replay for reconcile.py; not a timing harness."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import ttnn
from tracy import signpost


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("phi_advisor_harness", HERE / "harness.py")
harness = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(harness)

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
try:
    state = harness.build(mesh, {})
    harness.decode(state)
    ttnn.synchronize_device(mesh)
    trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
    harness.decode(state)
    ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
    ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
    ttnn.synchronize_device(mesh)
    signpost(header="PERF_DECODE")
    ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
    ttnn.synchronize_device(mesh)
    signpost(header="PERF_DECODE_END")
finally:
    ttnn.close_mesh_device(mesh)
