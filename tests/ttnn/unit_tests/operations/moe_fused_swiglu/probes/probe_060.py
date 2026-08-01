import torch, ttnn

device = ttnn.open_device(device_id=0)
try:
    TILE = 32
    EMB, HID = 7168, 2048
    HN_PAD = 6
    w = torch.randn(EMB, HID, dtype=torch.bfloat16)
    wi = ttnn.from_torch(
        w, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    print("interleaved ok", wi.shape, wi.memory_config())
    dg = device.dram_grid_size()
    print("dram grid", dg.x, dg.y)
    spec = ttnn.NdShardSpec(
        shard_shape=ttnn.Shape([TILE, HN_PAD * TILE]),
        grid=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dg.x - 1, dg.y - 1))]),
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    mc = ttnn.MemoryConfig(ttnn.BufferType.DRAM, spec)
    print("memcfg", mc)
    ws = ttnn.to_memory_config(wi, mc)
    print("sharded ok", ws.shape, ws.memory_config())
    print("page size i", wi.buffer_page_size(), "aligned", wi.buffer_aligned_page_size())
    print("page size s", ws.buffer_page_size(), "aligned", ws.buffer_aligned_page_size())
    a = ttnn.to_torch(wi).float()
    b = ttnn.to_torch(ws).float()
    print("roundtrip maxdiff", (a - b).abs().max().item())
    # also try from_torch directly with the sharded memcfg
    ws2 = ttnn.from_torch(w, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    print("from_torch sharded ok", ws2.memory_config())
    print("roundtrip2 maxdiff", (ttnn.to_torch(ws2).float() - a).abs().max().item())
    # accessor CT args
    print("TA args interleaved", ttnn.TensorAccessorArgs(wi).get_compile_time_args())
    print("TA args sharded", ttnn.TensorAccessorArgs(ws).get_compile_time_args())
finally:
    ttnn.close_device(device)
