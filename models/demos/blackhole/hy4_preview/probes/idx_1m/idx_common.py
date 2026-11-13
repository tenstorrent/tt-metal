# Common helpers for the Hy4 indexer probes (task idx-1m). Slot 0 of host .44: TT_VISIBLE_DEVICES from slot_env.sh, 1x8 mesh.
import time

import ttnn

HEADS, DIM, TOPK, COLS = 32, 128, 2048, 8


def open_md(trace=400_000_000):
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    return ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=trace)


def close_md(md):
    ttnn.close_mesh_device(md)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def traced_us(md, fn, n=5, reps=3):
    """per-call latency (us) of fn replayed n times inside one trace; fn must be re-runnable. Returns min over reps."""
    out = fn()  # compile
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    for _ in range(n):
        out = fn()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    best = 1e30
    for _ in range(reps):
        t = time.perf_counter()
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        best = min(best, (time.perf_counter() - t) / n * 1e6)
    ttnn.release_trace(md, tid)
    return best


def eager_us(md, fn, n=3):
    fn()
    ttnn.synchronize_device(md)
    t = time.perf_counter()
    for _ in range(n):
        fn()
    ttnn.synchronize_device(md)
    return (time.perf_counter() - t) / n * 1e6


def rep(md):
    return ttnn.ReplicateTensorToMesh(md)


def up(md, t, dt=ttnn.bfloat16, lay=ttnn.TILE_LAYOUT, shard=False):
    mm = ttnn.ShardTensorToMesh(md, dim=0) if shard else ttnn.ReplicateTensorToMesh(md)
    return ttnn.from_torch(
        t.contiguous(), device=md, dtype=dt, layout=lay, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mm
    )


def gid_of(c, l, layout="slice64"):
    """global entry id of local slot l on column c. slice64: pages of 512 tokens, column c holds tokens [64c, 64c+64) of every page."""
    if layout == "slice64":
        return (l >> 6) * 512 + c * 64 + (l & 63)
    return l * COLS + c  # interleave (DSV4)
