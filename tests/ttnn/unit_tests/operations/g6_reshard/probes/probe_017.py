import time, collections, torch, ttnn
from ttnn.operations.g6_reshard import g6_reshard, last_plan

device = ttnn.open_device(device_id=0)
try:

    def cfg(grid, shard, o):
        (x0, y0), (x1, y1) = grid
        cr = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))})
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, ttnn.ShardSpec(cr, list(shard), o)
        )

    RM = ttnn.ShardOrientation.ROW_MAJOR
    x = ttnn.from_torch(
        torch.randn(1, 1, 2048, 1024).to(torch.bfloat16),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=cfg(((0, 0), (10, 9)), (224, 96), RM),
    )
    out_cfg = cfg(((0, 0), (7, 7)), (256, 128), RM)
    t0 = time.perf_counter()
    o1 = g6_reshard(x, out_cfg)
    ttnn.synchronize_device(device)
    t1 = time.perf_counter()
    o2 = g6_reshard(x, out_cfg)
    ttnn.synchronize_device(device)
    t2 = time.perf_counter()
    o3 = g6_reshard(x, out_cfg)
    ttnn.synchronize_device(device)
    t3 = time.perf_counter()
    print(f"T miss(incl compile)={1e3*(t1-t0):.1f}ms hit={1e3*(t2-t1):.2f}ms hit={1e3*(t3-t2):.2f}ms")
    p = last_plan()
    c = collections.Counter((k, i["noc"]) for r in p["roles"] for k, i in enumerate(r["issuers"]))
    print("PLAN", p["regime"], p["noc_policy"], dict(c))
finally:
    ttnn.close_device(device)
