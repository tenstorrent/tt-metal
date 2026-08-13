import inspect
import statistics
import time

import torch

import ttnn

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200_000_000, l1_small_size=32768)
TOPO = ttnn.Topology.Ring
cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
sems = [ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(8)]
barrier = ttnn.create_global_semaphore(mesh, cores, 0)
print("P|all_reduce_async sig:", str(inspect.signature(ttnn.experimental.all_reduce_async))[:600])


def timed(fn, reps=17, iters=25):
    o = fn()
    ttnn.synchronize_device(mesh)
    ttnn.deallocate(o)

    def cap(r):
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        for _ in range(r):
            ttnn.deallocate(fn())
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ttnn.synchronize_device(mesh)
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            ts.append((time.perf_counter() - t0) * 1e6)
        ttnn.release_trace(mesh, tid)
        return statistics.median(ts)

    a, b = cap(1), cap(reps)
    return (b - a) / (reps - 1)


def mk(shape):
    return ttnn.from_torch(
        torch.randn(*shape),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


for rows, tag in [(32, "decode b<=32"), (512, "prefill s512")]:
    x = mk((1, 1, rows, 2048))

    # RS + AG composed
    def rsag():
        r = ttnn.experimental.reduce_scatter_minimal_async(
            x,
            persistent_output_buffers=None,
            dim=3,
            multi_device_global_semaphore=sems[0:3],
            barrier_semaphore=barrier,
            num_links=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=TOPO,
        )
        g = ttnn.experimental.all_gather_async(
            r,
            dim=3,
            multi_device_global_semaphore=sems[3:5],
            barrier_semaphore=barrier,
            num_links=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=TOPO,
        )
        ttnn.deallocate(r)
        return g

    try:
        print(f"P|{tag} RS+AG (composed all-reduce) = {timed(rsag):.2f}us")
    except Exception as e:
        print(f"P|{tag} RS+AG ERR {str(e)[:200]}")

    # AG of partials + local sum
    def agsum():
        g = ttnn.experimental.all_gather_async(
            x,
            dim=0,
            multi_device_global_semaphore=sems[5:7],
            barrier_semaphore=barrier,
            num_links=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=TOPO,
        )
        s = ttnn.sum(g, dim=0)
        ttnn.deallocate(g)
        return s

    try:
        print(f"P|{tag} AG(dim0)+local sum      = {timed(agsum):.2f}us")
    except Exception as e:
        print(f"P|{tag} AG+sum ERR {str(e)[:200]}")

    try:

        def ar():
            return ttnn.experimental.all_reduce_async(
                x,
                multi_device_global_semaphore=sems[0],
                num_links=2,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=TOPO,
            )

        print(f"P|{tag} all_reduce_async        = {timed(ar):.2f}us")
    except Exception as e:
        print(f"P|{tag} all_reduce_async ERR {str(e)[:250]}")
    ttnn.deallocate(x)

ttnn.close_mesh_device(mesh)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
