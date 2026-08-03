"""Profile exactly one replay using the same model hooks as harness.py; no timing result is produced."""

from tracy import signpost

import ttnn
from models.autoports.qwen_qwen3_6_27b.doc.advisor_challenger import harness


def main():
    frozen = harness.template.json.load(open(harness.template.os.environ["CHALLENGER_POLICY_JSON"]))
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        state = harness.build(mesh, frozen["shipped_policy"])
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


if __name__ == "__main__":
    main()
