"""Micro-benchmark the CCL family for the Laguna multichip decode residual all_reduce.
Payload mirrors decode: [1,1,32,2048] bf16, 2 all_reduces/layer, 1x4 ring, 2 links.
Compares: (a) composite ttnn.all_reduce, (b) all_reduce_async (deepseek form),
(c) all_reduce_async persistent-buffer+semaphore form, (d) reduce_scatter+all_gather async.
Traced, warmed. Prints us/all_reduce for each.
"""
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


def mk_input(dev):
    t = torch.randn(*SHAPE) * 0.3
    return ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, mesh_mapper=ttnn.ReplicateTensorToMesh(dev)
    )


def time_traced(dev, fn, n=ITERS):
    # warm + compile
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
    dt = (time.perf_counter() - t0) * 1e6 / n  # us per iter (2 all_reduces)
    ttnn.release_trace(dev, tid)
    return dt / 2.0  # us per single all_reduce


def main():
    dev = open_mesh()
    try:
        x = mk_input(dev)
        results = {}

        # (a) composite ttnn.all_reduce
        def comp():
            a = ttnn.all_reduce(x, cluster_axis=AXIS, topology=TOPO, num_links=NLINKS)
            b = ttnn.all_reduce(x, cluster_axis=AXIS, topology=TOPO, num_links=NLINKS)
            return a, b

        try:
            results["composite_all_reduce"] = time_traced(dev, comp)
        except Exception as e:
            results["composite_all_reduce"] = f"ERR {type(e).__name__}: {str(e)[:200]}"

        # (b) all_reduce_async deepseek form
        def ara():
            a = ttnn.experimental.all_reduce_async(
                x,
                cluster_axis=AXIS,
                mesh_device=dev,
                num_links=NLINKS,
                math_op=ttnn.ReduceType.Sum,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=TOPO,
            )
            b = ttnn.experimental.all_reduce_async(
                x,
                cluster_axis=AXIS,
                mesh_device=dev,
                num_links=NLINKS,
                math_op=ttnn.ReduceType.Sum,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=TOPO,
            )
            return a, b

        try:
            results["all_reduce_async_simple"] = time_traced(dev, ara)
        except Exception as e:
            results["all_reduce_async_simple"] = f"ERR {type(e).__name__}: {str(e)[:200]}"

        # (b-L1) all_reduce_async into L1
        def ara_l1():
            a = ttnn.experimental.all_reduce_async(
                x,
                cluster_axis=AXIS,
                mesh_device=dev,
                num_links=NLINKS,
                math_op=ttnn.ReduceType.Sum,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                topology=TOPO,
            )
            b = ttnn.experimental.all_reduce_async(
                x,
                cluster_axis=AXIS,
                mesh_device=dev,
                num_links=NLINKS,
                math_op=ttnn.ReduceType.Sum,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                topology=TOPO,
            )
            return a, b

        try:
            results["all_reduce_async_L1"] = time_traced(dev, ara_l1)
        except Exception as e:
            results["all_reduce_async_L1"] = f"ERR {type(e).__name__}: {str(e)[:200]}"

        print("=== CCL family (us per single all_reduce, [1,1,32,2048] bf16, 1x4 ring, 2 links) ===")
        for k, v in results.items():
            if isinstance(v, float):
                print(f"  {k:32s} {v:7.2f} us")
            else:
                print(f"  {k:32s} {v}")
    finally:
        ttnn.close_mesh_device(dev)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
