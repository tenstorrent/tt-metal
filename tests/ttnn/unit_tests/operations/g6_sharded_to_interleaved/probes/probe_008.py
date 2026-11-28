# Trace-replay ns/op of the production cases at 1 and 2 write lanes (unprofiled, host clock over N replays).
import math, time, torch, ttnn
from ttnn.operations.g6_sharded_to_interleaved.g6_sharded_to_interleaved import _dispatch
from ttnn.operations.g6_sharded_to_interleaved.g6_sharded_to_interleaved_program_descriptor import Knobs

CASES = {
    "p2": (
        (1, 5120),
        ttnn.bfloat16,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        (32, 160),
        [((0, 0), (7, 3))],
        ttnn.DRAM_MEMORY_CONFIG,
    ),
    "p1": (
        (1, 128, 256, 32),
        ttnn.bfloat8_b,
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        (320, 32),
        [((0, 0), (10, 8)), ((0, 9), (3, 9))],
        ttnn.L1_MEMORY_CONFIG,
    ),
}
N = 400
device = ttnn.open_device(device_id=0, trace_region_size=1 << 20)
try:
    for name, (shape, dtype, scheme, shard, grid, out_mc) in CASES.items():
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*a), ttnn.CoreCoord(*b)) for a, b in grid])
        mc = ttnn.MemoryConfig(scheme, ttnn.BufferType.L1, ttnn.ShardSpec(crs, shard, ttnn.ShardOrientation.ROW_MAJOR))
        x = ttnn.from_torch(
            torch.randn(shape).bfloat16(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc
        )
        for lanes in (1, 2):
            knobs = Knobs(num_write_lanes=lanes)
            warm = _dispatch(x, out_mc, knobs)
            ttnn.deallocate(warm)
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            y = _dispatch(x, out_mc, knobs)
            ttnn.end_trace_capture(device, tid, cq_id=0)
            for _ in range(20):
                ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            samples = []
            for _ in range(3):
                t0 = time.perf_counter_ns()
                for _ in range(N):
                    ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(device)
                samples.append((time.perf_counter_ns() - t0) / N)
            ok = torch.equal(
                ttnn.to_torch(y).view(torch.int16) if dtype == ttnn.bfloat16 else ttnn.to_torch(y),
                ttnn.to_torch(x).view(torch.int16) if dtype == ttnn.bfloat16 else ttnn.to_torch(x),
            )
            print(f"REPLAY {name} lanes={lanes} ns/op={[round(s) for s in samples]} exact={ok}")
            ttnn.release_trace(device, tid)
            ttnn.deallocate(y)
        ttnn.deallocate(x)
finally:
    ttnn.close_device(device)
