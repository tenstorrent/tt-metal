"""Emit one bounded eager decode replay for op-level reconciliation only.

Latency decisions are made exclusively by harness.py's imported fixed protocol.
"""

from tracy import signpost

import ttnn
from models.autoports.qwen_qwen3_6_27b.doc.advisor_challenger.harness import build, decode


device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
try:
    state = build(device, {"candidate": "default"})
    decode(state)
    ttnn.synchronize_device(device)
    signpost(header="PERF_DECODE")
    decode(state)
    ttnn.synchronize_device(device)
    signpost(header="PERF_DECODE_END")
finally:
    ttnn.close_mesh_device(device)
