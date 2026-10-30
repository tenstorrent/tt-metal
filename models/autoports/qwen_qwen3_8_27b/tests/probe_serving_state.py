# SPDX-License-Identifier: Apache-2.0
"""Small exact-state probe; no model weights or serving performance claim."""
import json
from pathlib import Path
import torch
import ttnn

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
results = []
try:
    for shape, dtype, layout in [((4, 3, 2560), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
                                 ((4, 12, 128, 128), ttnn.float32, ttnn.TILE_LAYOUT)]:
        x = torch.randn(shape).to(torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32)
        mapping = torch.tensor([3, 1, 0, 2])
        idx = mapping.reshape(4, *([1] * (len(shape)-1))).expand(shape).contiguous().int()
        def put(t, dt):
            return ttnn.from_torch(t, dtype=dt, layout=layout, device=mesh,
                                  mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        device_x = put(x, dtype)
        device_idx = put(idx, ttnn.uint32)
        output = ttnn.gather(device_x, 0, device_idx)
        actual = ttnn.to_torch(ttnn.get_device_tensors(output)[0])
        expected = x[mapping]
        row = dict(shape=shape, dtype=str(dtype), exact=torch.equal(actual, expected),
                   max_error=float((actual-expected).abs().max()))
        results.append(row)
        assert row['exact'], row
        del output, device_idx, device_x
finally:
    ttnn.close_mesh_device(mesh)
    path = Path('models/autoports/qwen_qwen3_8_27b/readiness_vllm/state_gather_probe.json')
    path.write_text(json.dumps(results, indent=2)+'\n')
