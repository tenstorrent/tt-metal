import torch, ttnn
from ttnn.operations.tilize import tilize

device = ttnn.open_device(device_id=0)


def run(tag, shape, th, dtype, out_dtype, in_mc, out_mc):
    try:
        if dtype in (ttnn.uint32, ttnn.uint16, ttnn.int32, ttnn.uint8):
            t = torch.randint(0, 100, shape, dtype=torch.int32)
        else:
            t = torch.randn(shape)
        tt_in = ttnn.from_torch(t, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_mc)
        out = tilize(tt_in, memory_config=out_mc, dtype=out_dtype, tile=ttnn.Tile([th, 32]))
        got = ttnn.to_torch(out).float()
        ref = ttnn.to_torch(tt_in).float()
        print(f"case {tag}: OK maxdiff={(got - ref).abs().max().item()}")
    except Exception as e:
        print(f"case {tag}: FAIL {type(e).__name__}: {str(e)[:250]}")


grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))])
ws = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    ttnn.BufferType.L1,
    ttnn.ShardSpec(grid, [32, 32], ttnn.ShardOrientation.ROW_MAJOR),
)
DR = ttnn.DRAM_MEMORY_CONFIG
try:
    for th in [8, 16, 1]:
        run(f"wshard th={th}", (1, 1, 32, 1024), th, ttnn.bfloat16, ttnn.bfloat16, ws, ws)
    for dt in [ttnn.float32, ttnn.uint32, ttnn.uint8, ttnn.uint16]:
        for th in [16, 2, 1]:
            run(f"dtype={dt} th={th}", (1, 1, 128, 256), th, dt, dt, DR, DR)
    run("cast bf16->fp32 th=8", (1, 1, 128, 256), 8, ttnn.bfloat16, ttnn.float32, DR, DR)
    run("cast bf16->bfp8 th=8", (1, 1, 128, 256), 8, ttnn.bfloat16, ttnn.bfloat8_b, DR, DR)
    run("cast bf16->bfp8 th=1", (1, 1, 128, 256), 1, ttnn.bfloat16, ttnn.bfloat8_b, DR, DR)
finally:
    ttnn.close_device(device)
