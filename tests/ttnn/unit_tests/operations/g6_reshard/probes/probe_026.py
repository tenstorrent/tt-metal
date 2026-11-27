# Non-production shapes: per-image tile padding (d2 % 32 != 0 with H % 32 == 0) and tails, exact compare.
import torch, ttnn
from ttnn.operations.g6_reshard import g6_reshard


def cfg(grid, shard, orient):
    (x0, y0), (x1, y1) = grid
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))})
    o = ttnn.ShardOrientation.ROW_MAJOR if orient == "r" else ttnn.ShardOrientation.COL_MAJOR
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, ttnn.ShardSpec(crs, list(shard), o)
    )


device = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(1)
    cases = [
        # shape, in (grid, shard, orient), out (grid, shard, orient)
        ((2, 1, 48, 96), (((0, 0), (2, 1)), (64, 32), "r"), (((3, 2), (4, 3)), (64, 64), "c")),
        ((1, 2, 16, 128), (((0, 0), (3, 1)), (32, 32), "r"), (((1, 1), (1, 1)), (64, 128), "r")),
        ((1, 1, 320, 224), (((0, 0), (2, 3)), (96, 96), "r"), (((5, 0), (7, 1)), (160, 96), "c")),
    ]
    for shape, (ig, ish, io), (og, osh, oo) in cases:
        t = torch.randn(shape).to(torch.bfloat16)
        x = ttnn.from_torch(
            t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=cfg(ig, ish, io)
        )
        xh = ttnn.to_torch(x)
        out = g6_reshard(x, cfg(og, osh, oo))
        oh = ttnn.to_torch(out)
        same = bool((oh.view(torch.int16) == xh.view(torch.int16)).all())
        mc = out.memory_config().shard_spec
        print(
            f"CASE {shape} in={ig},{ish},{io} out={og},{osh},{oo}: exact={same} out_grid={mc.grid} shard={mc.shape} orient={mc.orientation}"
        )
finally:
    ttnn.close_device(device)
