"""Emit one bounded batch-32 replay for op-level profiling (not a timing harness)."""
import json

import ttnn
from tracy import signpost

from models.autoports.meta_llama_llama_3_2_1b_instruct.doc.advisor_challenger.harness import build, decode

policy = json.load(open("models/autoports/meta_llama_llama_3_2_1b_instruct/doc/advisor_challenger/policy.json"))["shipped_policy"]
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
    signpost(header="PERF_DECODE")
    ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
    ttnn.synchronize_device(mesh)
    signpost(header="PERF_DECODE_END")
finally:
    ttnn.close_mesh_device(mesh)
