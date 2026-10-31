"""Probe: 1x8 RING vs LINE fabric, stock all_gather_async slope.  Run: RING_PROBE_CFG=line|ring python this_file.py"""
import os
import statistics
import time

M = "models/demos/blackhole/qwen38_flash_next/tools/"
CFG = os.environ.get("RING_PROBE_CFG", "line")
os.environ["TT_MESH_GRAPH_DESC_PATH"] = os.path.join(
    os.environ.get("PYTHONPATH", ".").split(":")[0], M + f"p150_x8_1x8_{CFG}_mesh_graph_descriptor.textproto"
)
import torch

import ttnn


def main():
    ring = CFG == "ring"
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING if ring else ttnn.FabricConfig.FABRIC_1D,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
    )
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 8), l1_small_size=24576, trace_region_size=2000000)
    print("PROBE devices", list(mesh.get_device_ids()), "shape", tuple(mesh.shape), flush=True)
    try:
        grid = mesh.compute_with_storage_grid_size()
        crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        N_SEM = 16
        sems = [ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(2 * N_SEM)]
        bars = [ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(N_SEM)]
        ref = torch.randn(1, 1, 32, 320 * 8).bfloat16()
        x = ttnn.from_torch(
            ref,
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
        )
        state = {"i": 0}

        def ag(topo):
            i = state["i"] % N_SEM
            state["i"] += 1
            return ttnn.experimental.all_gather_async(
                x,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=[sems[2 * i], sems[2 * i + 1]],
                num_links=1,
                cluster_axis=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=topo,
                barrier_semaphore=bars[i],
                chunks_per_sync=1,
                num_workers_per_link=1,
                num_buffers_per_channel=2,
            )

        def check(topo):
            out = ag(topo)
            ttnn.synchronize_device(mesh)
            for t in ttnn.get_device_tensors(out):
                if not torch.equal(ttnn.to_torch(t), ref):
                    return False
            return True

        def slope(topo, name):
            ts = {}
            for n in (4, 12):
                state["i"] = 0
                ag(topo)
                ttnn.synchronize_device(mesh)  # warm compile
                state["i"] = 0
                tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                for _ in range(n):
                    ag(topo)
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
                ttnn.synchronize_device(mesh)
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                v = []
                for _ in range(20):
                    t0 = time.perf_counter()
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                    v.append((time.perf_counter() - t0) * 1e6)
                ttnn.release_trace(mesh, tid)
                ts[n] = statistics.median(v)
            print(
                f"PROBE {CFG}/{name} t4={ts[4]:.1f}us t12={ts[12]:.1f}us slope={(ts[12]-ts[4])/8:.2f}us/call",
                flush=True,
            )

        runs = [(ttnn.Topology.Linear, "Linear")] + ([(ttnn.Topology.Ring, "Ring")] if ring else [])
        runs = runs[::-1] if ring else runs
        for topo, name in runs:
            print(f"PROBE {CFG}/{name} correct={check(topo)}", flush=True)
            slope(topo, name)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def test_ring_probe_device():
    import pytest

    if not os.environ.get("RING_PROBE_RUN"):
        pytest.skip("set RING_PROBE_RUN=1")
    main()


if __name__ == "__main__":
    main()
