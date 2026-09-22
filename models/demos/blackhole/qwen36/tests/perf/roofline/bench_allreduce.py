"""Collectives cost a fixed ~120us each, so COUNT is the only lever. Today each half-layer does
3 of them: reduce_scatter -> norm stats all_gather -> norm output all_gather (the residual stays
hidden-sharded, which forces a DISTRIBUTED norm).  A single fused all_reduce would leave the
residual REPLICATED, making both norm gathers local: 1 collective per half-layer instead of 3.
ttnn.all_reduce has its own device op + program factory, so measure whether it really beats RS+AG."""
import time

import torch

import ttnn

T, H = 1024, 2048
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(4, 8), l1_small_size=24576, trace_region_size=200000000)
try:
    from models.common.modules.tt_ccl import TT_CCL

    topo = ttnn.Topology.Linear
    print(f"  {'TP':>3} {'pattern':<38}{'time':>10}{'per layer x2':>14}{'x24':>9}")
    for TP in (8, 4):
        sub = mesh.create_submeshes(ttnn.MeshShape(1, TP))[0]
        ccl = TT_CCL(sub)
        nl = ccl.get_num_links(1)

        def bench(name, fn, shape):
            t = ttnn.from_torch(
                torch.randn(*shape, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
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
                print(f"  {TP:>3} {name:<38}{best:8.1f}us{2*best:>12.0f}us{2*best*24/1000:>7.1f}ms")
                ttnn.release_trace(sub, tid)
                ttnn.deallocate(o)
                return best
            except Exception as e:
                print(f"  {TP:>3} {name:<38} FAIL {str(e).splitlines()[0][:44]}")
                return None
            finally:
                ttnn.deallocate(t)

        def rs_then_ag(t):
            r = ttnn.experimental.reduce_scatter_minimal_async(
                t,
                dim=3,
                multi_device_global_semaphore=ccl.get_and_cycle_rs_semaphore_handles(1),
                num_links=nl,
                topology=topo,
                cluster_axis=1,
            )[0]
            o = ttnn.experimental.all_gather_async(
                r,
                dim=3,
                multi_device_global_semaphore=ccl.get_and_cycle_ag_semaphore_handles(1),
                num_links=nl,
                topology=topo,
                cluster_axis=1,
                mesh_device=sub,
            )
            ttnn.deallocate(r)
            return o

        bench("reduce_scatter + all_gather (today)", rs_then_ag, (1, 1, T, H))
        bench(
            "fused ttnn.all_reduce",
            lambda t: ttnn.all_reduce(t, cluster_axis=1, num_links=nl, topology=topo),
            (1, 1, T, H),
        )
        bench(
            "norm stats all_gather (tiny, 32 wide)",
            lambda t: ttnn.experimental.all_gather_async(
                t,
                dim=3,
                multi_device_global_semaphore=ccl.get_and_cycle_ag_semaphore_handles(1),
                num_links=nl,
                topology=topo,
                cluster_axis=1,
                mesh_device=sub,
            ),
            (1, 1, T, 32),
        )
finally:
    try:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    except Exception:
        pass
