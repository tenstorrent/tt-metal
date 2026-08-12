import torch, ttnn
from ttnn.operations.tilize import tilize

device = ttnn.open_device(device_id=0)
DR = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG


def run(tag, shape, in_th, out_th, dtype, in_mc, out_mc):
    try:
        n = 1
        for d in shape:
            n *= d
        t = torch.arange(n, dtype=torch.float32).reshape(shape) % 4096
        if dtype == ttnn.bfloat16:
            t = t.to(torch.bfloat16).float()
        tt_in = ttnn.from_torch(
            t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in_mc, tile=ttnn.Tile([in_th, 32])
        )
        ref = ttnn.to_torch(tt_in).float()
        out = tilize(tt_in, memory_config=out_mc, tile=ttnn.Tile([out_th, 32]))
        got = ttnn.to_torch(out).float()
        print(
            f"case {tag}: tile={out.tile.tile_shape} equal={torch.equal(got, ref)} maxdiff={(got - ref).abs().max().item()}"
        )
    except Exception as e:
        print(f"case {tag}: FAIL {type(e).__name__}: {str(e)[:250]}")


grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))])
bs = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ttnn.BufferType.L1,
    ttnn.ShardSpec(grid, [32, 32], ttnn.ShardOrientation.ROW_MAJOR),
)
try:
    for a, b in [(32, 8), (1, 32), (32, 16), (16, 32), (8, 32), (2, 4), (4, 2), (32, 1), (1, 16)]:
        run(f"il {a}->{b}", (1, 1, 128, 256), a, b, ttnn.bfloat16, DR, DR)
    run("block-shard 32->16", (1, 1, 256, 256), 32, 16, ttnn.bfloat16, bs, bs)
    run("block-shard 32->8", (1, 1, 256, 256), 32, 8, ttnn.bfloat16, bs, bs)
    run("rank3 32->8", (3, 128, 256), 32, 8, ttnn.bfloat16, DR, DR)
    run("uint8 32->8", (1, 1, 128, 256), 32, 8, ttnn.uint8, DR, DR)
    run("fp32 32->8", (1, 1, 128, 256), 32, 8, ttnn.float32, DR, DR)
    run("il 32->8 l1", (1, 1, 128, 256), 32, 8, ttnn.bfloat16, L1, L1)
finally:
    ttnn.close_device(device)
