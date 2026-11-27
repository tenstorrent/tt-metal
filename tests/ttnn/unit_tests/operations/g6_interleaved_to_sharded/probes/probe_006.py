import torch, ttnn, time
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
    print("device() is device():", x.device() is x.device(), "ids:", id(x.device()), id(x.device()), "dev id", id(dev))
    print("device type", type(x.device()), "has id():", hasattr(x.device(), "id"))
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))})
    mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (32, 160), ttnn.ShardOrientation.ROW_MAJOR),
    )
    for i in range(20):
        y = g6_interleaved_to_sharded(x, mc)
        y.deallocate()
    print("python cache entries after 20 calls:", len(d._PROGRAM_CACHE))
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for i in range(200):
        y = g6_interleaved_to_sharded(x, mc)
        y.deallocate()
    ttnn.synchronize_device(dev)
    print("host us/call (eager, incl alloc):", (time.perf_counter() - t0) / 200 * 1e6)
    print("python cache entries:", len(d._PROGRAM_CACHE))
    print("accessor ct args", ttnn.TensorAccessorArgs(x).get_compile_time_args())
finally:
    ttnn.close_device(dev)
