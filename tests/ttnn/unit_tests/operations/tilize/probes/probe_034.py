import torch, ttnn
from ttnn.operations.tilize import tilize

device = ttnn.open_device(device_id=0)
try:
    shape = [1, 1, 128, 256]
    x = torch.arange(128 * 256).reshape(shape).float() % 1000
    for a, b in [(32, 8), (1, 32), (32, 16), (16, 32), (8, 4), (32, 32), (2, 16)]:
        try:
            tt = ttnn.from_torch(
                x.bfloat16(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                tile=ttnn.Tile([a, 32]),
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            out = tilize(
                tt,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                use_multicore=True,
                tile=ttnn.Tile([b, 32]),
            )
            got = ttnn.to_torch(out).float()
            ok = torch.equal(got, x.bfloat16().float())
            print(
                f"retile {a:2d}->{b:2d}: tile={list(out.tile.tile_shape)} equal={ok} maxdiff={(got-x.bfloat16().float()).abs().max().item()}"
            )
        except Exception as e:
            print(f"retile {a:2d}->{b:2d}: EXC {type(e).__name__}: {str(e)[:200]}")
finally:
    ttnn.close_device(device)
