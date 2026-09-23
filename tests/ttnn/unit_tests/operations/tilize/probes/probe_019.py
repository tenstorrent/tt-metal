import torch, ttnn
from ttnn.operations.tilize import tilize

dev = ttnn.open_device(device_id=0)
for th in [32, 16, 8]:
    t = ttnn.Tile([th, 32])
    print(
        "RES tile",
        th,
        "bfp8",
        t.get_tile_size(ttnn.bfloat8_b),
        "bfp4",
        t.get_tile_size(ttnn.bfloat4_b),
        "bf16",
        t.get_tile_size(ttnn.bfloat16),
    )
# Pattern: row r has value (r+1) at all cols, 32x32 image, output tile 16x32
x = torch.zeros(1, 1, 32, 32)
for r in range(32):
    x[0, 0, r, :] = r + 1
x = x.bfloat16()
tin = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
y = tilize(tin, dtype=ttnn.bfloat8_b, tile=ttnn.Tile([16, 32]))
o = ttnn.to_torch(y)
print("RES bfp8 t16 col0", o[0, 0, :, 0].tolist())
print("RES bfp8 t16 row0", o[0, 0, 0, :].tolist())
print("RES bfp8 t16 row16", o[0, 0, 16, :].tolist())
print("RES buffer page", y.buffer_page_size() if hasattr(y, "buffer_page_size") else None)
ttnn.close_device(dev)
