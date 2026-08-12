import torch, ttnn
from ttnn.operations.tilize.tilize import _dispatch

device = ttnn.open_device(device_id=0)


def check(name, shape, in_tile, out_tile, dtype, levers):
    torch.manual_seed(0)
    if dtype == ttnn.uint8:
        src = torch.randint(0, 256, shape, dtype=torch.uint8)
    else:
        src = torch.randn(shape).bfloat16()
    if in_tile is None:
        tt = ttnn.from_torch(
            src, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
    else:
        tt = ttnn.from_torch(
            src,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            tile=ttnn.Tile([in_tile, 32]),
        )
    kw = dict(tile=ttnn.Tile([out_tile, 32]))
    ref = ttnn.to_torch(_dispatch(tt, ttnn.DRAM_MEMORY_CONFIG, levers=dict(), **kw))
    got = ttnn.to_torch(_dispatch(tt, ttnn.DRAM_MEMORY_CONFIG, levers=levers, **kw))
    ok_ref = torch.equal(ref.to(src.dtype), src)
    ok = torch.equal(got, ref)
    print(f"{name}: shipped==torch {ok_ref} | lever{levers} bit-identical-to-shipped {ok}")


check("retile_shrink 32->8, read128", (1, 1, 1024, 1024), 32, 8, ttnn.bfloat16, dict(target_read_bytes=128))
check("retile_grow 8->32, read128", (1, 1, 1024, 1024), 8, 32, ttnn.bfloat16, dict(target_read_bytes=128))
check("uint8 narrow Wt=1, noc_split=0", (1, 1, 8192, 32), None, 32, ttnn.uint8, dict(noc_split=0))
check("tile_1 th=1, read128", (1, 1, 512, 2048), None, 1, ttnn.bfloat16, dict(target_read_bytes=128))

ttnn.close_device(device)
