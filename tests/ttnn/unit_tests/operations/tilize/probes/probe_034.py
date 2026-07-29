# Lever 2 (drop the writer kernel on alias_out) — first light + program-cache re-binding.
import torch, ttnn
from ttnn.operations.tilize import tilize
from ttnn.operations.tilize import tilize_program_descriptor as tpd

device = ttnn.open_device(device_id=0)
try:
    _L1 = ttnn.BufferType.L1
    _ROW = ttnn.ShardOrientation.ROW_MAJOR

    def crs(ex, ey):
        return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(ex, ey))})

    def sh(scheme, g, s):
        return ttnn.MemoryConfig(scheme, _L1, ttnn.ShardSpec(g, s, _ROW))

    H = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    W = ttnn.TensorMemoryLayout.WIDTH_SHARDED
    B = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    cases = [
        ("block", (1, 1, 2048, 512), sh(B, crs(7, 7), (256, 64))),
        ("height", (1, 1, 256, 128), sh(H, crs(1, 1), (64, 128))),
        ("width", (1, 1, 128, 256), sh(W, crs(1, 1), (128, 64))),
        ("block_small", (1, 1, 512, 128), sh(B, crs(1, 3), (128, 64))),
    ]
    for name, shape, mc in cases:
        t = torch.arange(int(torch.tensor(shape).prod()), dtype=torch.float32).reshape(shape).bfloat16()
        d = ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        probe = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mc)
        p = tpd.build_plan(d, probe, device)
        out = ttnn.to_torch(tilize(d, mc))
        print(
            f"{name} path={p['path']} drop_writer={p['drop_writer']} blk={p['blocks_per_core']} EQUAL={torch.equal(out,t)}",
            flush=True,
        )
        assert torch.equal(out, t), name
    # re-binding: two calls, different shard addresses, one cache entry, both exact
    shape = (1, 1, 256, 128)
    mc = sh(B, crs(1, 1), (128, 64))
    t = torch.arange(shape[2] * shape[3], dtype=torch.float32).reshape(shape).bfloat16()
    d = ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    first = tilize(d, mc)
    n = device.num_program_cache_entries()
    second = tilize(d, mc)
    print(
        "cache entries stable:",
        device.num_program_cache_entries() == n,
        "| distinct shards:",
        first.buffer_address() != second.buffer_address(),
        "| both exact:",
        torch.equal(ttnn.to_torch(first), t) and torch.equal(ttnn.to_torch(second), t),
        flush=True,
    )
    assert device.num_program_cache_entries() == n and first.buffer_address() != second.buffer_address()
    assert torch.equal(ttnn.to_torch(first), t) and torch.equal(ttnn.to_torch(second), t)
    # 5 repeat launches on the big one — the "CB never needs recycling across launches" claim
    shape = (1, 1, 2048, 512)
    mc = sh(B, crs(7, 7), (256, 64))
    t = torch.arange(shape[2] * shape[3], dtype=torch.float32).reshape(shape).bfloat16()
    d = ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    for i in range(5):
        assert torch.equal(ttnn.to_torch(tilize(d, mc)), t), f"launch {i}"
    print("5 repeat launches bit-exact — no cross-launch CB leak")
    print("ALL OK")
finally:
    ttnn.close_device(device)
