"""Device-only exact integer positions and selected attention-row extraction."""

import torch

import ttnn

from .run_functional import to_device

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
try:
    for value in [0, 31, 32, 777, 524287]:
        pos = to_device(torch.tensor([value], dtype=torch.int32), mesh, True)
        tiled = ttnn.to_layout(ttnn.reshape(pos, (1, 1, 1, 1)), ttnn.TILE_LAYOUT)
        offset = ttnn.to_layout(ttnn.bitwise_and(tiled, -32), ttnn.ROW_MAJOR_LAYOUT)
        offset = ttnn.reshape(offset, (1,))
        index = ttnn.typecast(ttnn.bitwise_and(tiled, 31), ttnn.uint32)
        index = ttnn.repeat(index, (1, 32, 1, 128))
        host = torch.randn(1, 32, 32, 128).bfloat16()
        attention = to_device(host, mesh)
        selected = ttnn.gather(attention, 2, index)
        assert ttnn.to_torch(offset).item() == value // 32 * 32
        assert torch.equal(ttnn.to_torch(selected), host[:, :, value % 32 : value % 32 + 1, :])
        print("POSITION_MAPPING_PASS", value, flush=True)
finally:
    ttnn.close_mesh_device(mesh)
