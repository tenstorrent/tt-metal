# Trace-replay ns/op with K calls per trace (amortizes execute_trace), plus an empty-kernel program on
# the same core ranges with per-core vs common-only RT args to isolate the per-core RT-arg dispatch cost.
import time, torch, ttnn
from ttnn.operations.g6_sharded_to_interleaved.g6_sharded_to_interleaved import _dispatch
from ttnn.operations.g6_sharded_to_interleaved.g6_sharded_to_interleaved_program_descriptor import Knobs

EMPTY = "generated/verifier/g6_sharded_to_interleaved/empty_kernel/empty.cpp"
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
K, N = 16, 50


def timed(device, run):
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    keep = [run() for _ in range(K)]
    ttnn.end_trace_capture(device, tid, cq_id=0)
    for _ in range(5):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    samples = []
    for _ in range(3):
        t0 = time.perf_counter_ns()
        for _ in range(N):
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        samples.append(round((time.perf_counter_ns() - t0) / (N * K)))
    ttnn.release_trace(device, tid)
    return samples, keep


def empty_program(grid, cores, per_core):
    rt = ttnn.RuntimeArgs()
    if per_core:
        for c in cores:
            rt[c.x][c.y] = [1, 2, 3, 4, 5, 6]
    k = ttnn.KernelDescriptor(
        kernel_source=EMPTY,
        core_ranges=grid,
        compile_time_args=[],
        runtime_args=rt,
        common_runtime_args=[0, 0, 0],
        config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
    )
    return ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[])


device = ttnn.open_device(device_id=0, trace_region_size=4 << 20)
try:
    for name, (shape, dtype, scheme, shard, grid, out_mc) in CASES.items():
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*a), ttnn.CoreCoord(*b)) for a, b in grid])
        mc = ttnn.MemoryConfig(scheme, ttnn.BufferType.L1, ttnn.ShardSpec(crs, shard, ttnn.ShardOrientation.ROW_MAJOR))
        x = ttnn.from_torch(
            torch.randn(shape).bfloat16(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc
        )
        cores = ttnn.corerange_to_cores(crs, None, True)
        warm = _dispatch(x, out_mc)
        ttnn.deallocate(warm)
        s, keep = timed(device, lambda: _dispatch(x, out_mc))
        print(f"REPLAY {name} op lanes=1 ns/op={s}")
        for t in keep:
            ttnn.deallocate(t)
        dummy = ttnn.from_torch(
            torch.zeros(1, 32).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        for per_core in (True, False):
            prog = empty_program(crs, cores, per_core)
            ttnn.generic_op([dummy], prog)
            s, _ = timed(device, lambda: ttnn.generic_op([dummy], prog))
            print(f"REPLAY {name} empty-kernel {'per-core 6 RT args' if per_core else 'common RT args only'} ns/op={s}")
        ttnn.deallocate(dummy)
        ttnn.deallocate(x)
finally:
    ttnn.close_device(device)
