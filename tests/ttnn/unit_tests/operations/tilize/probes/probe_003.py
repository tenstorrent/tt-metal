import torch, ttnn
from ttnn.operations.tilize.tilize import create_program_descriptor

dev = ttnn.open_device(device_id=0)
D = ttnn.DRAM_MEMORY_CONFIG
for th in (16, 8, 4, 2, 1):
    for shape in [(1, 1, 32, 64), (1, 1, 64, 128), (1, 1, 4096, 64)]:
        x = torch.randn(shape).bfloat16()
        t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=D)
        spec = ttnn.TensorSpec(
            list(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.DRAM, tile=ttnn.Tile([th, 32])
        )
        try:
            out = ttnn.allocate_tensor_on_device(spec, dev)
            desc = create_program_descriptor(t, out)
            y = ttnn.generic_op([t, out], desc)
            ref = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, tile=ttnn.Tile([th, 32]))
            print("RES", th, shape, torch.equal(ttnn.to_torch(y), x), y.tile.tile_shape)
        except Exception as e:
            print("RES", th, shape, "EXC", str(e)[:200])
ttnn.close_device(dev)
