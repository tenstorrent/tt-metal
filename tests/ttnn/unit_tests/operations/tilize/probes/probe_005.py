import torch, ttnn
from ttnn.operations.tilize import tilize

device = ttnn.open_device(device_id=0)
try:
    for shape, th in [
        ([1, 1, 32, 64], 16),
        ([1, 1, 32, 64], 8),
        ([1, 1, 64, 128], 1),
        ([1, 1, 128, 256], 4),
        ([1, 1, 128, 256], 2),
        ([2, 3, 64, 96], 16),
        ([1, 1, 16384, 64], 8),
    ]:
        x = torch.randn(shape).to(torch.bfloat16)
        t = ttnn.from_torch(
            x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        o = tilize(t, tile=ttnn.Tile([th, 32]))
        y = ttnn.to_torch(o)
        print(shape, th, list(o.tile.tile_shape), torch.equal(y, x), (y.float() - x.float()).abs().max().item())
finally:
    ttnn.close_device(device)
