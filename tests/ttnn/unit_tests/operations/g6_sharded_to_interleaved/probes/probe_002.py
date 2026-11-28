# Probe: widen SUPPORTED in-process; check every TARGET scenario bit-exact at num_write_lanes 1 and 2.
import math, torch, ttnn
import ttnn.operations.g6_sharded_to_interleaved.g6_sharded_to_interleaved as op

P2 = dict(
    shape=(1, 5120),
    dtype=ttnn.bfloat16,
    scheme=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    shard=(32, 160),
    grid=[((0, 0), (7, 3))],
    orient=ttnn.ShardOrientation.ROW_MAJOR,
)
P1 = dict(
    shape=(1, 128, 256, 32),
    dtype=ttnn.bfloat8_b,
    scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    shard=(320, 32),
    grid=[((0, 0), (10, 8)), ((0, 9), (3, 9))],
    orient=ttnn.ShardOrientation.ROW_MAJOR,
)
P2O = {**P2, "shard": (32, 96), "grid": [((0, 0), (8, 5))], "orient": ttnn.ShardOrientation.COL_MAJOR}
P1O = {
    **P1,
    "shard": (352, 32),
    "grid": [((0, 0), (8, 9)), ((9, 0), (9, 3))],
    "orient": ttnn.ShardOrientation.COL_MAJOR,
}
# Width-sharded with >1 tile row and a short last shard: exercises the nominal slot stride.
W2 = dict(
    shape=(2, 1, 64, 320),
    dtype=ttnn.bfloat16,
    scheme=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    shard=(128, 96),
    grid=[((0, 0), (3, 0))],
    orient=ttnn.ShardOrientation.ROW_MAJOR,
)
# Height-sharded, 2 tile cols, d2 not a multiple of 32 (per-image tile padding).
H2 = dict(
    shape=(1, 3, 40, 64),
    dtype=ttnn.bfloat16,
    scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    shard=(64, 64),
    grid=[((0, 0), (1, 1))],
    orient=ttnn.ShardOrientation.ROW_MAJOR,
)

op.SUPPORTED["dtype"] = [ttnn.bfloat16, ttnn.bfloat8_b]
op.SUPPORTED["in_scheme"] = ["width_sharded", "height_sharded"]
op.SUPPORTED["buffer"] = ["dram", "l1"]

device = ttnn.open_device(device_id=0)
try:
    for lanes in (1, 2):
        op.NUM_WRITE_LANES = lanes
        for name, s in (("P2", P2), ("P1", P1), ("P2O", P2O), ("P1O", P1O), ("W2", W2), ("H2", H2)):
            for out_mc in (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG):
                crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*a), ttnn.CoreCoord(*b)) for a, b in s["grid"]])
                mc = ttnn.MemoryConfig(s["scheme"], ttnn.BufferType.L1, ttnn.ShardSpec(crs, s["shard"], s["orient"]))
                torch.manual_seed(0)
                rows, cols = math.prod(s["shape"][:-1]), s["shape"][-1]
                x = torch.randn(rows, cols) * torch.pow(
                    2.0, ((torch.arange(rows) // 32)[:, None] * 7 + (torch.arange(cols) // 32)[None, :]) % 16 - 8.0
                )
                x = x.reshape(s["shape"]).bfloat16()
                # dirty the allocator range the output will likely take
                d = ttnn.from_torch(
                    torch.full(s["shape"], 3.0).bfloat16(),
                    dtype=s["dtype"],
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=out_mc,
                )
                ttnn.deallocate(d)
                t = ttnn.from_torch(x, dtype=s["dtype"], layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
                exp = ttnn.to_torch(t)
                y = op.g6_sharded_to_interleaved(t, out_mc)
                got = ttnn.to_torch(y)
                ok = torch.equal(got.view(torch.int16), exp.view(torch.int16))
                print(("OK " if ok else "BAD"), "lanes", lanes, name, out_mc.buffer_type)
                ttnn.deallocate(y)
                ttnn.deallocate(t)
    op.NUM_WRITE_LANES = 2
    t2 = ttnn.from_torch(
        torch.randn(1, 5120).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(
            P2["scheme"],
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]), (32, 160), P2["orient"]
            ),
        ),
    )
    p = op.describe_plan(t2, ttnn.DRAM_MEMORY_CONFIG)
    print("plan", p["plan"], p["num_write_lanes"], p["cores"][0], p["cores"][31])
finally:
    ttnn.close_device(device)
