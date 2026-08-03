"""Collect exactly one bounded replay for reconcile.py; this does not time candidates."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import ttnn
from tracy import signpost

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("phi35_challenger_harness", HERE / "harness.py")
harness = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(harness)

device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
try:
    state = harness.build(device, {"policy_name": "final"})
    harness.decode(state)
    ttnn.synchronize_device(device)
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    harness.decode(state)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    ttnn.synchronize_device(device)
    signpost(header="PERF_DECODE")
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    ttnn.synchronize_device(device)
    signpost(header="PERF_DECODE_END")
    ttnn.release_trace(device, trace_id)
finally:
    ttnn.close_mesh_device(device)
