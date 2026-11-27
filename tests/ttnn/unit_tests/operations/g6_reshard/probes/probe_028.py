# COL_MAJOR input off the origin into 1-tile output shards (70 out cores), exact compare.
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
    shape = (1, 1, 320, 224)
    t = torch.randn(shape).to(torch.bfloat16)
    x = ttnn.from_torch(
        t,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=cfg(((1, 1), (4, 3)), (96, 96), "c"),
    )
    xh = ttnn.to_torch(x)
    out = g6_reshard(x, cfg(((0, 0), (6, 9)), (32, 32), "r"))
    same = bool((ttnn.to_torch(out).view(torch.int16) == xh.view(torch.int16)).all())
    print(f"CASE {shape} in=(1,1)-(4,3) 96x96 col_major -> out (0,0)-(6,9) 32x32 row_major: exact={same}")
finally:
    ttnn.close_device(device)
