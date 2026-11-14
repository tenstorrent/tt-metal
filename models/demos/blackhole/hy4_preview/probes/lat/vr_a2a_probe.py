"""verify-rows: all_to_all of the split-kernel partials: tile owner layout [8,S,32,576] (shipped) vs compact row-major [8,1,8S,544] (8 real head rows only)."""
import json
import os
import statistics
import time

import torch

import ttnn


def R(**kw):
    print("RES " + json.dumps(kw), flush=True)


def traced_us(md, fn, n=7, reps=3):
    def body():
        for _ in range(reps):
            o = fn()
            for t in o if isinstance(o, (tuple, list)) else [o]:
                if isinstance(t, ttnn.Tensor):
                    ttnn.deallocate(t)

    body()
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    body()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        ts.append((time.perf_counter() - t0) / reps * 1e6)
    ttnn.release_trace(md, tid)
    return round(statistics.median(ts), 1)


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=400_000_000)
    try:
        for S in [int(x) for x in os.environ.get("SS", "1,2,4,8").split(",")]:
            for name, shape, lay in (
                ("tile32", [8, S, 32, 576], ttnn.TILE_LAYOUT),
                ("rm8", [8, 1, 8 * S, 544], ttnn.ROW_MAJOR_LAYOUT),
                ("tile8S", [8, 1, max(32, 8 * S), 544], ttnn.TILE_LAYOUT),
            ):
                x = ttnn.from_torch(
                    torch.randn(*shape).bfloat16(),
                    device=md,
                    dtype=ttnn.bfloat16,
                    layout=lay,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(md),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                try:
                    us = traced_us(
                        md,
                        lambda: ttnn.experimental.all_to_all_async_generic(
                            x, in_dim=1, out_dim=0, num_links=2, topology=ttnn.Topology.Ring, cluster_axis=1
                        ),
                    )
                    R(S=S, layout=name, shape=shape, a2a_us=us)
                except Exception as e:
                    R(S=S, layout=name, err=str(e)[:200])
                ttnn.deallocate(x)
    finally:
        ttnn.close_mesh_device(md)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


main()
