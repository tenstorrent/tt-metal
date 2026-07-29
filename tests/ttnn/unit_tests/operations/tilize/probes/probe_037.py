import torch, ttnn
from ttnn.operations.tilize import tilize
from ttnn.operations.tilize import tilize_program_descriptor as tpd

_L1 = ttnn.BufferType.L1
_ROW = ttnn.ShardOrientation.ROW_MAJOR


def crs(ex, ey):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(ex, ey))})


def shard(scheme, grid, shape, orient=_ROW):
    return ttnn.MemoryConfig(scheme, _L1, ttnn.ShardSpec(grid, shape, orient))


H = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
W = ttnn.TensorMemoryLayout.WIDTH_SHARDED
B = ttnn.TensorMemoryLayout.BLOCK_SHARDED


def run(device, name, shape, cfg, dtype=ttnn.bfloat16, out_dtype=None, out_cfg="same"):
    n = 1
    for d in shape:
        n *= d
    t = torch.arange(n, dtype=torch.float32).reshape(shape)
    if dtype in (ttnn.uint32, ttnn.int32, ttnn.uint16):
        t = (torch.arange(n) % 5000).reshape(shape)
        tt = ttnn.from_torch(
            t.to(torch.int32) if dtype != ttnn.uint16 else t.to(torch.int32),
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=cfg,
        )
    else:
        tt = ttnn.from_torch(
            t.to(torch.bfloat16) if dtype == ttnn.bfloat16 else t,
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=cfg,
        )
    oc = cfg if out_cfg == "same" else out_cfg
    plan = tpd.build_plan(tt, oc, dtype=out_dtype, use_multicore=True, use_double_buffer=None, device=device)
    out = tilize(tt, oc, dtype=out_dtype, use_multicore=True)
    got = ttnn.to_torch(out)
    ref = ttnn.to_torch(tt)
    ok = torch.equal(got.to(torch.float32), ref.to(torch.float32))
    print(
        f"{name:36s} path={plan['path']:8s} rcb={plan['resident_cb']} blk={plan['blocks_per_core']} "
        f"chk={plan['chunk_wt']} dr={plan['drop_reader']} dw={plan['drop_writer']} exact={ok} "
        f"maxdiff={(got.to(torch.float32)-ref.to(torch.float32)).abs().max().item():.4g}"
    )
    assert ok, name


device = ttnn.open_device(device_id=0)
try:
    # Path B: both sides aliased
    run(device, "B/H 4blk", (1, 1, 512, 64), shard(H, crs(3, 0), (128, 64)))
    run(device, "B/H 1blk tiny", (1, 1, 128, 64), shard(H, crs(3, 0), (32, 64)))
    run(device, "B/BLOCK 8blk", (1, 1, 2048, 512), shard(B, crs(7, 7), (256, 64)))
    run(device, "B/WIDTH", (1, 1, 64, 512), shard(W, crs(7, 0), (64, 64)))
    run(device, "B/H COL orient", (1, 1, 512, 64), shard(H, crs(3, 0), (128, 64), ttnn.ShardOrientation.COL_MAJOR))
    run(device, "B/H multichunk", (1, 1, 128, 256), shard(H, crs(1, 0), (64, 256)))
    # fp32 -> fp32 (slow lossless path) and fp32 -> bf16 (fast) on Path B
    run(device, "B/H fp32", (1, 1, 256, 64), shard(H, crs(1, 0), (128, 64)), dtype=ttnn.float32)
    run(
        device,
        "B/H fp32->bf16",
        (1, 1, 256, 64),
        shard(H, crs(1, 0), (128, 64)),
        dtype=ttnn.float32,
        out_dtype=ttnn.bfloat16,
    )
    run(device, "B/H uint32", (1, 1, 256, 64), shard(H, crs(1, 0), (128, 64)), dtype=ttnn.uint32)
    # alias_out crossover: DRAM interleaved -> BLOCK sharded
    run(device, "alias_out xover", (1, 1, 2048, 512), None, out_cfg=shard(B, crs(7, 7), (256, 64)))
    run(device, "alias_out xover small", (1, 1, 512, 128), None, out_cfg=shard(B, crs(1, 3), (128, 64)))
    # alias_in crossover: sharded -> DRAM (writer stays; input aliased but reader stays)
    run(device, "alias_in xover", (1, 1, 2048, 512), shard(B, crs(7, 7), (256, 64)), out_cfg=ttnn.DRAM_MEMORY_CONFIG)
    # plain interleaved
    run(device, "interleaved", (1, 1, 256, 256), ttnn.DRAM_MEMORY_CONFIG, out_cfg=ttnn.DRAM_MEMORY_CONFIG)
finally:
    ttnn.close_device(device)
print("ALL OK")
