"""Collectives cost a near-fixed ~120us regardless of bytes. Is that fixed cost HOP LATENCY?
On a Linear topology TP=8 is 7 hops, TP=4 is 3, TP=2 is 1. If cost tracks hops, then the TP
width -- not the message size -- is what sets our collective bill, and the whole SPxTP config
choice has to be re-made."""
import time

import torch

import ttnn

T, H = 1024, 2048
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(4, 8), l1_small_size=24576, trace_region_size=200000000)
try:
    from models.common.modules.tt_ccl import TT_CCL

    print(f"  {'TP':>3}{'hops':>6}{'all_gather':>13}{'reduce_scatter':>16}{'AG+RS/layer x2':>17}{'x24 layers':>12}")
    for TP in (2, 4, 8):
        sub = mesh.create_submeshes(ttnn.MeshShape(1, TP))[0]
        ccl = TT_CCL(sub)
        topo = ttnn.Topology.Linear
        res = {}
        for name, shape, fn in (
            (
                "ag",
                (1, 1, T, H // TP),
                lambda t: ttnn.experimental.all_gather_async(
                    t,
                    dim=3,
                    multi_device_global_semaphore=ccl.get_and_cycle_ag_semaphore_handles(1),
                    num_links=2,
                    topology=topo,
                    cluster_axis=1,
                    mesh_device=sub,
                ),
            ),
            (
                "rs",
                (1, 1, T, H),
                lambda t: ttnn.experimental.reduce_scatter_minimal_async(
                    t,
                    dim=3,
                    multi_device_global_semaphore=ccl.get_and_cycle_rs_semaphore_handles(1),
                    num_links=2,
                    topology=topo,
                    cluster_axis=1,
                )[0],
            ),
        ):
            t = ttnn.from_torch(
                torch.randn(*shape, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=sub,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(sub),
            )
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
            res[name] = best
            ttnn.release_trace(sub, tid)
            ttnn.deallocate(o)
            ttnn.deallocate(t)
        per_layer = 2 * (res["ag"] + res["rs"])
        print(
            f"  {TP:>3}{TP-1:>6}{res['ag']:>11.1f}us{res['rs']:>14.1f}us{per_layer:>15.0f}us{per_layer*24/1000:>10.1f}ms"
        )
finally:
    try:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    except Exception:
        pass
