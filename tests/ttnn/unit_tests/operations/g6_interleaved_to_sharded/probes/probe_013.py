# Unprofiled trace replay: per-call wall time of N captured calls, old default (1 issuer) vs new default.
import time, torch, ttnn
from ttnn.operations.g6_interleaved_to_sharded import g6_interleaved_to_sharded
from ttnn.operations.g6_interleaved_to_sharded import g6_interleaved_to_sharded_program_descriptor as d

dev = ttnn.open_device(device_id=0, trace_region_size=4000000)
N = 200
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
    variants = {
        "1issuer": {"issuers_per_core": 1},
        "default": {},
        "1issuer_b": {"issuers_per_core": 1},
        "default_b": {},
    }
    base = dict(d.KNOBS)
    for name, knobs in variants.items():
        d.KNOBS.clear()
        d.KNOBS.update(base)
        d.KNOBS.update(knobs)
        y = g6_interleaved_to_sharded(x, mc)
        ok = torch.equal(ttnn.to_torch(y).view(torch.int16), t0.view(torch.int16))
        y.deallocate()
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        for i in range(N):
            y = g6_interleaved_to_sharded(x, mc)
            if i < N - 1:
                y.deallocate()  # the next call reuses the address; only the last output stays
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
        ts = []
        for _ in range(7):
            s = time.perf_counter()
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
            ts.append(time.perf_counter() - s)
        ts.sort()
        r = ttnn.to_torch(y)
        ok = ok and torch.equal(r.view(torch.int16), t0.view(torch.int16))
        print(f"RESULT {name}: median {ts[3]/N*1e9:.0f} ns/call, min {ts[0]/N*1e9:.0f} ns/call, bit_exact={ok}")
        ttnn.release_trace(dev, tid)
        y.deallocate()
    d.KNOBS.clear()
    d.KNOBS.update(base)
finally:
    ttnn.close_device(dev)
