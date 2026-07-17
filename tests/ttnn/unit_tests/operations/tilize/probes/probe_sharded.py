import torch, ttnn


def run(name, shape, scheme, grid, shard_shape, orient, dtype=ttnn.bfloat16):
    dev = device
    if scheme is None:
        nd = ttnn.NdShardSpec(ttnn.Shape(shard_shape), grid, orient)
        mc = ttnn.MemoryConfig(ttnn.BufferType.L1, nd)
    else:
        ss = ttnn.ShardSpec(grid, shard_shape, orient)
        mc = ttnn.MemoryConfig(scheme, ttnn.BufferType.L1, ss)
    t = torch.rand(shape).bfloat16()
    tt_in = ttnn.from_torch(t, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=mc)
    from ttnn.operations.tilize import tilize

    tt_out = tilize(tt_in, memory_config=mc)
    out = ttnn.to_torch(tt_out)
    ok = torch.equal(t, out)
    print(f"{name}: shape={shape} equal={ok} max_diff={(t.float()-out.float()).abs().max().item()}")
    return ok


device = ttnn.open_device(device_id=0)
try:
    _crs = lambda a, b: ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(*a), ttnn.CoreCoord(*b))})
    _ROW = ttnn.ShardOrientation.ROW_MAJOR
    _COL = ttnn.ShardOrientation.COL_MAJOR
    H, W, B = (
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    )
    run("HEIGHT", [1, 1, 512, 64], H, _crs((0, 0), (3, 0)), (128, 64), _ROW)
    run("WIDTH", [1, 1, 64, 512], W, _crs((0, 0), (3, 0)), (64, 128), _ROW)
    run("BLOCK-col", [1, 1, 128, 128], B, _crs((0, 0), (1, 1)), (64, 64), _COL)
    run("nd-r4", [1, 1, 128, 128], None, _crs((0, 0), (1, 1)), (1, 1, 64, 64), _ROW)
    run("nd-r3", [4, 32, 64], None, _crs((0, 0), (1, 0)), (2, 32, 64), _ROW)
finally:
    ttnn.close_device(device)
