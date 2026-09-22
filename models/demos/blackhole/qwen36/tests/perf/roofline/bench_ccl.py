"""What does a collective cost in ISOLATION at our exact shapes, with nothing to wait for?
In-model we measure AllGather avg 78.8us / min 14.6us. If isolated ~= the min, the 79us is
skew (a scheduling problem); if isolated ~= 79us, it is transfer (a bytes problem). The two
diagnoses call for completely different fixes, so measure before refactoring."""
import time

import torch

import ttnn

T, H, TP = 1024, 2048, 8
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(4, 8), l1_small_size=24576, trace_region_size=200000000)
try:
    sub = mesh.create_submeshes(ttnn.MeshShape(1, TP))[0]
    from models.common.modules.tt_ccl import TT_CCL

    ccl = TT_CCL(sub)
    topo = ttnn.Topology.Linear

    def run(name, fn, shape, dt):
        t = ttnn.from_torch(
            torch.randn(*shape, dtype=torch.bfloat16),
            dtype=dt,
            layout=ttnn.TILE_LAYOUT,
            device=sub,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(sub),
        )
        try:
            o = fn(t)
            ttnn.synchronize_device(sub)
            ttnn.deallocate(o)
            tid = ttnn.begin_trace_capture(sub, cq_id=0)
            o = fn(t)
            ttnn.end_trace_capture(sub, tid, cq_id=0)
            ttnn.synchronize_device(sub)
            best = 1e9
            for _ in range(4):
                t0 = time.time()
                for _ in range(50):
                    ttnn.execute_trace(sub, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(sub)
                best = min(best, (time.time() - t0) / 50 * 1e6)
            mb = 1
            for d in shape:
                mb *= d
            mb = mb * (4 if dt == ttnn.float32 else 2) / 1e6
            print(f"  {name:<34}{str(shape):<22}{str(dt).split('.')[-1][:8]:<9}{mb:7.2f}MB {best:8.1f}us")
            ttnn.release_trace(sub, tid)
            ttnn.deallocate(o)
        except Exception as e:
            print(f"  {name:<34}{str(shape):<22} FAIL {str(e).splitlines()[0][:50]}")
        ttnn.deallocate(t)

    print(f"  {'collective':<34}{'shape':<22}{'dtype':<9}{'bytes':>9} {'isolated':>9}")
    for dt in (ttnn.bfloat16, ttnn.float32):
        run(
            "all_gather dim3 (post-reduce)",
            lambda t: ttnn.experimental.all_gather_async(
                t,
                dim=3,
                multi_device_global_semaphore=ccl.get_and_cycle_ag_semaphore_handles(1),
                num_links=2,
                topology=topo,
                cluster_axis=1,
                mesh_device=sub,
            ),
            (1, 1, T, H // TP),
            dt,
        )
        run(
            "reduce_scatter dim3",
            lambda t: ttnn.experimental.reduce_scatter_minimal_async(
                t,
                dim=3,
                multi_device_global_semaphore=ccl.get_and_cycle_rs_semaphore_handles(1),
                num_links=2,
                topology=topo,
                cluster_axis=1,
            )[0],
            (1, 1, T, H),
            dt,
        )
    run(
        "all_gather dim2 (token, for repl-MLP)",
        lambda t: ttnn.experimental.all_gather_async(
            t,
            dim=2,
            multi_device_global_semaphore=ccl.get_and_cycle_ag_semaphore_handles(1),
            num_links=2,
            topology=topo,
            cluster_axis=1,
            mesh_device=sub,
        ),
        (1, 1, T // TP, H),
        ttnn.bfloat16,
    )
finally:
    try:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    except Exception:
        pass
