"""One-replay profiler companion to the fixed challenger harness."""
from __future__ import annotations

import json
import os

from tracy import signpost

import ttnn
from models.autoports.microsoft_phi_3_5_mini_instruct.doc.advisor_challenger.harness import build, decode


def main():
    with open(os.environ["CHALLENGER_POLICY_JSON"]) as handle:
        policy = json.load(handle)["shipped_policy"]
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        state = build(mesh, policy)
        decode(state)
        ttnn.synchronize_device(mesh)
        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        decode(state)
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        for _ in range(10):
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
        ttnn.synchronize_device(mesh)
        ttnn.ReadDeviceProfiler(mesh)
        signpost(header="PERF_DECODE")
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
        ttnn.synchronize_device(mesh)
        signpost(header="PERF_DECODE_END")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
