import torch, ttnn
from ttnn.operations.g6_untilize import g6_untilize

device = ttnn.open_device(device_id=0)
try:
    x = torch.randint(0, 2**31, (1, 1, 32, 128), dtype=torch.int64).to(torch.int32).view(torch.uint32)
    try:
        t = ttnn.from_torch(
            x,
            dtype=ttnn.uint32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            tile=ttnn.Tile((16, 32)),
        )
        print("PROBE tile_shape", t.tile.tile_shape)
        try:
            g6_untilize(t)
            print("PROBE RESULT: no error (BAD)")
        except ValueError as e:
            print("PROBE RESULT: ValueError:", e)
    except Exception as e:
        print("PROBE RESULT: cannot build 16x32 uint32 tile tensor:", type(e).__name__, str(e)[:200])
finally:
    ttnn.close_device(device)
