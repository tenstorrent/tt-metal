import torch, ttnn
from ttnn.operations.tilize import tilize


def crs(*ranges):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(*s), ttnn.CoreCoord(*e)) for (s, e) in ranges})


_ROW = ttnn.ShardOrientation.ROW_MAJOR
_COL = ttnn.ShardOrientation.COL_MAJOR
_L1 = ttnn.BufferType.L1
H = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
W = ttnn.TensorMemoryLayout.WIDTH_SHARDED
B = ttnn.TensorMemoryLayout.BLOCK_SHARDED


def legacy(grid, sshape, orient, scheme):
    return ttnn.MemoryConfig(scheme, _L1, ttnn.ShardSpec(grid, sshape, orient))


device = ttnn.open_device(device_id=0)
try:
    cases = [
        ("HEIGHT same-spec", [1, 1, 512, 64], legacy(crs(((0, 0), (3, 0))), (128, 64), _ROW, H), "same"),
        ("WIDTH same-spec", [1, 1, 64, 512], legacy(crs(((0, 0), (3, 0))), (64, 128), _ROW, W), "same"),
        ("BLOCK COL same", [1, 1, 128, 128], legacy(crs(((0, 0), (1, 1))), (64, 64), _COL, B), "same"),
        ("HEIGHT COL same", [1, 1, 256, 64], legacy(crs(((0, 0), (0, 3))), (64, 64), _COL, H), "same"),
        ("WIDTH COL same", [1, 1, 32, 256], legacy(crs(((0, 0), (0, 3))), (32, 64), _COL, W), "same"),
    ]
    for name, shape, mc, kind in cases:
        x = torch.arange(torch.Size(shape).numel()).reshape(shape).float().bfloat16()
        t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc)
        out = tilize(t, memory_config=mc, dtype=ttnn.bfloat16)
        got = ttnn.to_torch(out)
        ok = torch.equal(got.float(), x.float())
        print(f"{name}: match={ok}", "" if ok else f"maxdiff={(got.float()-x.float()).abs().max()}")
        ttnn.deallocate(t)
        ttnn.deallocate(out)
finally:
    ttnn.close_device(device)
