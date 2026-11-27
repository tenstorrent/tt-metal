# Tail blocks on both sides with a COL_MAJOR output off the origin, exact compare.
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
    for shape, (ig, ish, io), (og, osh, oo) in [
        ((1, 1, 320, 224), (((0, 0), (2, 3)), (96, 96), "r"), (((5, 0), (6, 2)), (160, 96), "c")),
        ((1, 1, 320, 224), (((1, 1), (3, 4)), (96, 96), "c"), (((0, 0), (6, 9)), (32, 32), "r")),
    ]:
        t = torch.randn(shape).to(torch.bfloat16)
        x = ttnn.from_torch(
            t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=cfg(ig, ish, io)
        )
        xh = ttnn.to_torch(x)
        out = g6_reshard(x, cfg(og, osh, oo))
        same = bool((ttnn.to_torch(out).view(torch.int16) == xh.view(torch.int16)).all())
        print(f"CASE {shape} in={ig},{ish},{io} out={og},{osh},{oo}: exact={same}")
finally:
    ttnn.close_device(device)
