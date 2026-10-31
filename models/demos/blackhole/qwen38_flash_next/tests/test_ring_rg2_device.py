"""row_gather2 alone on the 1x8 mesh: bit-exact vs torch gather, traced slope N=4/12.
Run: RG2_CASE=line|ring python this_file.py  (needs PYTHONPATH=<tree>)"""
import os
import statistics
import time

CASE = os.environ.get("RG2_CASE", "line")
RING = CASE.startswith("ring")
os.environ["QWEN38_RING"] = "1" if RING else "0"
M = "models/demos/blackhole/qwen38_flash_next/tools/"
os.environ["TT_MESH_GRAPH_DESC_PATH"] = os.path.join(
    os.environ["PYTHONPATH"].split(":")[0],
    M + f"p150_x8_1x8_{'ring' if RING else 'line'}_mesh_graph_descriptor.textproto",
)
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import fabric_config
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import row_gather2 as rg2

TP, SHARD, ROWS = 8, 320, 1


def main():
    FABRIC_CONFIG = fabric_config()
    ttnn.set_fabric_config(**FABRIC_CONFIG)
    print("RG2 fabric", FABRIC_CONFIG["config"], flush=True)
    mesh = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 8), l1_small_size=24576, trace_region_size=16 * 1024 * 1024
    )
    try:
        mesh.enable_program_cache()
        g = torch.Generator().manual_seed(5)
        x = torch.randn(TP, 1, ROWS, SHARD, generator=g).to(torch.bfloat16)
        src = ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        expect = torch.cat([x[d] for d in range(TP)], dim=-1)

        def call():
            return rg2.row_gather(src, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        def exact(t):
            h = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:, :, :ROWS, :]
            return all(torch.equal(h[d], expect) for d in range(TP))

        ok = exact(call())
        for _ in range(6):
            ok = ok and exact(call())
        ts = {}
        for n in (4, 12):
            call()
            ttnn.synchronize_device(mesh)
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            outs = [call() for _ in range(n)]
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            ttnn.synchronize_device(mesh)
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            ok = ok and all(exact(o) for o in outs)
            v = []
            for _ in range(20):
                t0 = time.perf_counter()
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                v.append((time.perf_counter() - t0) * 1e6)
            ttnn.release_trace(mesh, tid)
            ts[n] = statistics.median(v)
        print(f"RG2 {CASE} t4={ts[4]:.1f} t12={ts[12]:.1f} slope={(ts[12]-ts[4])/8:.2f}us/call exact={ok}", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
