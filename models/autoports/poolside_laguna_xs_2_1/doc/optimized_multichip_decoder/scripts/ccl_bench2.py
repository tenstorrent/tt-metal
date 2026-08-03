"""CCL family part 2: persistent-buffer all_reduce (OPT-009), RS+AG split, async RS+AG.
Same payload [1,1,32,2048] bf16, 1x4 ring, 2 links. Traced, warmed. us per collective op-group."""
import time

import torch

import ttnn

SHAPE = (1, 1, 32, 2048)
ITERS = 200
NLINKS = 2
AXIS = 1
TOPO = ttnn.Topology.Ring


def open_mesh():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    return ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200_000_000)


def mk(dev):
    t = torch.randn(*SHAPE) * 0.3
    return ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, mesh_mapper=ttnn.ReplicateTensorToMesh(dev)
    )


def time_traced(dev, fn, n=ITERS, ngroup=2):
    fn()
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    fn()
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.synchronize_device(dev)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(n):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    dt = (time.perf_counter() - t0) * 1e6 / n
    ttnn.release_trace(dev, tid)
    return dt / ngroup


def main():
    dev = open_mesh()
    res = {}
    try:
        x = mk(dev)

        # RS + AG composite (what all_reduce lowers to): measure the pair as one "all_reduce-equivalent"
        def rsag():
            for _ in range(2):
                r = ttnn.reduce_scatter(x, dim=3, cluster_axis=AXIS, topology=TOPO, num_links=NLINKS)
                ttnn.all_gather(r, dim=3, cluster_axis=AXIS, topology=TOPO, num_links=NLINKS)

        try:
            res["reduce_scatter+all_gather"] = time_traced(dev, rsag)
        except Exception as e:
            res["reduce_scatter+all_gather"] = f"ERR {type(e).__name__}: {str(e)[:160]}"

        # RS alone (delayed-gather family lower bound: if next op consumes sharded)
        def rs_only():
            for _ in range(2):
                ttnn.reduce_scatter(x, dim=3, cluster_axis=AXIS, topology=TOPO, num_links=NLINKS)

        try:
            res["reduce_scatter_only(delayed-AG lower bound)"] = time_traced(dev, rs_only)
        except Exception as e:
            res["reduce_scatter_only(delayed-AG lower bound)"] = f"ERR {type(e).__name__}: {str(e)[:160]}"

        # async RS minimal + async AG
        def rsag_async():
            for _ in range(2):
                r = ttnn.experimental.reduce_scatter_minimal_async(
                    x, dim=3, cluster_axis=AXIS, mesh_device=dev, num_links=NLINKS, topology=TOPO
                )
                ttnn.experimental.all_gather_async(
                    r, dim=3, cluster_axis=AXIS, mesh_device=dev, num_links=NLINKS, topology=TOPO
                )

        try:
            res["async RS_minimal+AG_async"] = time_traced(dev, rsag_async)
        except Exception as e:
            res["async RS_minimal+AG_async"] = f"ERR {type(e).__name__}: {str(e)[:160]}"

        # persistent-buffer all_reduce_async (OPT-009)
        try:
            crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(6, 6))])
            sem = [ttnn.create_global_semaphore(dev, crs, 0) for _ in range(3)]
            pbuf = ttnn.from_torch(
                torch.zeros(*SHAPE),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                mesh_mapper=ttnn.ReplicateTensorToMesh(dev),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            def ar_persist():
                for _ in range(2):
                    ttnn.experimental.all_reduce_async(
                        x,
                        pbuf,
                        cluster_axis=AXIS,
                        mesh_device=dev,
                        multi_device_global_semaphore=sem,
                        num_links=NLINKS,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        topology=TOPO,
                    )

            res["all_reduce_async_persistent"] = time_traced(dev, ar_persist)
        except Exception as e:
            res["all_reduce_async_persistent"] = f"ERR {type(e).__name__}: {str(e)[:220]}"

        print("=== CCL family part2 (us per all_reduce-equivalent) ===")
        for k, v in res.items():
            print(f"  {k:44s} {v:7.2f} us" if isinstance(v, float) else f"  {k:44s} {v}")
    finally:
        ttnn.close_mesh_device(dev)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
