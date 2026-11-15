"""pool-prefill probe A2: token-major q gather (big RM pages), RS over heads (dim 1), to_layout passes."""
import json
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import traced_us
from models.demos.gpt_oss.tt.ccl import CCLManager

N = int(os.environ.get("N", "512"))
R = lambda **kw: print("RES " + json.dumps(kw), flush=True)


def put(md, x, dim, layout, dt=ttnn.bfloat16):
    return ttnn.from_torch(
        x,
        device=md,
        dtype=dt,
        layout=layout,
        mesh_mapper=ttnn.ShardTensorToMesh(md, dim=dim),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=400_000_000)
    try:
        ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
        topo = ttnn.Topology.Ring
        nl = 2
        for lay, nm in ((ttnn.ROW_MAJOR_LAYOUT, "rm"), (ttnn.TILE_LAYOUT, "tile")):
            for dt, dn in ((ttnn.bfloat16, "bf16"), (ttnn.bfloat8_b, "bfp8")):
                if lay == ttnn.ROW_MAJOR_LAYOUT and dn == "bfp8":
                    continue
                q = put(md, torch.randn(8, 1, N, 4608), 0, lay, dt)

                def ag():
                    return ttnn.experimental.all_gather_async(
                        q,
                        dim=3,
                        cluster_axis=1,
                        mesh_device=md,
                        topology=topo,
                        multi_device_global_semaphore=ccl.get_ag_ping_pong_semaphore(),
                        num_links=nl,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        barrier_semaphore=ccl.get_barrier_semaphore(),
                    )

                try:
                    R(op=f"ag_tokmajor_{nm}_{dn}_dim3", n=N, us=round(traced_us(md, ag, n=7, rep_=3), 1))
                except Exception as e:
                    R(op=f"ag_tokmajor_{nm}_{dn}_dim3", err=repr(e)[:300])
                ttnn.deallocate(q)
        qg = put(md, torch.randn(8, 1, N, 4608), 0, ttnn.TILE_LAYOUT)  # stand-in with 8x bigger width below
        big = put(md, torch.randn(8, 1, N, 4608 * 8), 0, ttnn.TILE_LAYOUT)
        R(
            op="tile_to_rm_512x36864",
            us=round(traced_us(md, lambda: ttnn.to_layout(big, ttnn.ROW_MAJOR_LAYOUT), n=7, rep_=3), 1),
        )
        ttnn.deallocate(big)
        for dt, dn in ((ttnn.bfloat16, "bf16"), (ttnn.bfloat8_b, "bfp8")):
            o = put(md, torch.randn(8, 64, N, 512), 0, ttnn.TILE_LAYOUT, dt)

            def rs():
                return ttnn.experimental.reduce_scatter_minimal_async(
                    o,
                    dim=1,
                    multi_device_global_semaphore=ccl.get_rs_ping_pong_semaphore(),
                    num_links=nl,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    topology=topo,
                    cluster_axis=1,
                    barrier_semaphore=ccl.get_barrier_semaphore(),
                )

            try:
                R(op=f"rs_min_tile_dim1_{dn}", n=N, us=round(traced_us(md, rs, n=7, rep_=3), 1))
            except Exception as e:
                R(op=f"rs_min_tile_dim1_{dn}", err=repr(e)[:300])
            ttnn.deallocate(o)
        # n=1024 scaling of the tile AG / RS
        for n2 in (1024,):
            q = put(md, torch.randn(8, 1, n2, 4608), 0, ttnn.TILE_LAYOUT)

            def ag2():
                return ttnn.experimental.all_gather_async(
                    q,
                    dim=3,
                    cluster_axis=1,
                    mesh_device=md,
                    topology=topo,
                    multi_device_global_semaphore=ccl.get_ag_ping_pong_semaphore(),
                    num_links=nl,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    barrier_semaphore=ccl.get_barrier_semaphore(),
                )

            R(op="ag_tokmajor_tile_bf16_dim3", n=n2, us=round(traced_us(md, ag2, n=5, rep_=3), 1))
            ttnn.deallocate(q)
    finally:
        ttnn.close_mesh_device(md)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
