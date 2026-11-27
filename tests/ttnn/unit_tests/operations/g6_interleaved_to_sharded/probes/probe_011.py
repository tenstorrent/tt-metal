# BB-16: capture one call in a trace, replay it, compare with eager; new input values through the same buffer.
import torch, ttnn
from ttnn.operations.g6_interleaved_to_sharded import g6_interleaved_to_sharded

dev = ttnn.open_device(device_id=0, trace_region_size=200000)
try:
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))})
    mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (32, 160), ttnn.ShardOrientation.ROW_MAJOR),
    )
    t0 = torch.randn(1, 5120).to(torch.bfloat16)
    x = ttnn.from_torch(
        t0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    eager = ttnn.to_torch(g6_interleaved_to_sharded(x, mc))  # warm-up compiles
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    y = g6_interleaved_to_sharded(x, mc)
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    r1 = ttnn.to_torch(y)
    print("TRACE replay == eager:", torch.equal(r1.view(torch.int16), eager.view(torch.int16)))
    t1 = torch.randn(1, 5120).to(torch.bfloat16)
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(t1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), x)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    r2 = ttnn.to_torch(y)
    print("TRACE replay picks up new input values:", torch.equal(r2.view(torch.int16), t1.view(torch.int16)))
    ttnn.release_trace(dev, tid)
finally:
    ttnn.close_device(dev)
