import torch, ttnn
from ttnn.operations.tilize import tilize

device = ttnn.open_device(device_id=0)
try:
    for shape, ih, oh in [
        ([1, 1, 32, 64], 32, 16),
        ([1, 1, 32, 64], 16, 32),
        ([1, 1, 64, 128], 8, 4),
        ([1, 1, 32, 64], 4, 2),
        ([1, 1, 32, 64], 2, 1),
        ([1, 1, 32, 64], 1, 32),
        ([1, 1, 32, 64], 32, 32),
    ]:
        x = torch.randn(shape).to(torch.bfloat16)
        t = ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            tile=ttnn.Tile([ih, 32]),
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        o = tilize(t, tile=ttnn.Tile([oh, 32]))
        y = ttnn.to_torch(o)
        ok = torch.equal(y, x)
        print("R", shape, ih, oh, list(o.tile.tile_shape), ok, (y.float() - x.float()).abs().max().item())
finally:
    ttnn.close_device(device)
