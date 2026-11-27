import gc, torch, ttnn
from ttnn.operations.g6_interleaved_to_sharded import g6_interleaved_to_sharded
from ttnn.operations.g6_interleaved_to_sharded import g6_interleaved_to_sharded_program_descriptor as d

dev = ttnn.open_device(device_id=0)
try:
    x = ttnn.from_torch(
        torch.randn(1, 5120).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=dev,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    a, b = x.device(), x.device()
    print("device() is device():", a is b, "| id(dev)==id(x.device()):", id(dev) == id(x.device()))
    print("device type", type(a).__name__, "| has .id():", hasattr(a, "id"), a.id() if hasattr(a, "id") else None)
    print(
        "dram banks:",
        dev.num_dram_channels() if hasattr(dev, "num_dram_channels") else "n/a",
        "| dram alignment",
        ttnn.get_dram_alignment(),
    )
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))})
    mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (32, 160), ttnn.ShardOrientation.ROW_MAJOR),
    )
    for i in range(30):
        junk = [object() for _ in range(i)]  # perturb the allocator between calls
        x2 = ttnn.from_torch(
            torch.randn(1, 5120).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        y = g6_interleaved_to_sharded(x2, mc)
        y.deallocate()
        x2.deallocate()
        gc.collect()
    print("python cache entries after 30 calls with fresh tensors:", len(d._PROGRAM_CACHE))
    print("program cache entries:", dev.num_program_cache_entries())
finally:
    ttnn.close_device(dev)
