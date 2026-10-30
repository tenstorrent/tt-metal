"""A body exception must restore dispatch before the next trace or device close."""

import torch

import ttnn
from models.autoports.google_gemma_4_31b_it.tests.optimized_trace import capture

m = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
try:
    x = ttnn.from_torch(torch.ones(1, 1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=m)
    y = ttnn.add(x, 1)
    del y

    def fail():
        y = ttnn.add(x, 1)
        raise ValueError("controlled capture-body failure")

    try:
        capture(m, fail)
    except ValueError as e:
        assert str(e) == "controlled capture-body failure"
    else:
        raise AssertionError("capture did not rethrow")
    tid, y = capture(m, lambda: ttnn.add(x, 1))
    ttnn.execute_trace(m, tid, cq_id=0, blocking=True)
    assert torch.equal(ttnn.to_torch(y), torch.full((1, 1, 32, 32), 2.0))
    ttnn.release_trace(m, tid)
    print("CAPTURE_CLEANUP_PASS")
finally:
    ttnn.close_mesh_device(m)
