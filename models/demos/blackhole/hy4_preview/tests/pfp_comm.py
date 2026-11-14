"""pool-prefill probe A: CCL / layout costs of the direct-to-pool prefill attention (1x8 slot, traced us, per device).
q all_gather (8 heads -> 64 heads for n rows), partial-output exchange (a2a / reduce_scatter, RM and TILE), tilize + scale passes."""
import json
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import traced_us
from models.demos.gpt_oss.tt.ccl import CCLManager

N = int(os.environ.get("N", "512"))
R = lambda **kw: print("RES " + json.dumps(kw), flush=True)


def put(md, x, dim, layout, dt=ttnn.bfloat16):  # x [8,...] sharded on dim 0 of the host tensor -> one block per device
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
        for nl in (1, 2):
            # --- q all_gather, RM, [1,8,N,576] per device, dim 1 -> [1,64,N,576]
            q = put(md, torch.randn(8, 8, N, 576), 0, ttnn.ROW_MAJOR_LAYOUT)

            def ag():
                return ttnn.experimental.all_gather_async(
                    q,
                    dim=1,
                    cluster_axis=1,
                    mesh_device=md,
                    topology=topo,
                    multi_device_global_semaphore=ccl.get_ag_ping_pong_semaphore(),
                    num_links=nl,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    barrier_semaphore=ccl.get_barrier_semaphore(),
                )

            try:
                R(
                    op="ag_q_rm_dim1",
                    n=N,
                    links=nl,
                    us=round(traced_us(md, ag, n=7, rep_=4), 1),
                    mb_in=round(8 * N * 576 * 2 / 1e6, 2),
                )
            except Exception as e:
                R(op="ag_q_rm_dim1", links=nl, err=repr(e)[:300])
            q_t = put(md, torch.randn(8, 8, N, 576), 0, ttnn.TILE_LAYOUT)

            def ag_t():
                return ttnn.experimental.all_gather_async(
                    q_t,
                    dim=1,
                    cluster_axis=1,
                    mesh_device=md,
                    topology=topo,
                    multi_device_global_semaphore=ccl.get_ag_ping_pong_semaphore(),
                    num_links=nl,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    barrier_semaphore=ccl.get_barrier_semaphore(),
                )

            try:
                R(op="ag_q_tile_dim1", n=N, links=nl, us=round(traced_us(md, ag_t, n=7, rep_=4), 1))
            except Exception as e:
                R(op="ag_q_tile_dim1", links=nl, err=repr(e)[:300])
            ttnn.deallocate(q)
            ttnn.deallocate(q_t)
            # --- partial output [1,64,N,512] per device: reduce_scatter over tokens (dim 2), RM and TILE; a2a heads (TILE)
            for lay, nm in ((ttnn.TILE_LAYOUT, "tile"), (ttnn.ROW_MAJOR_LAYOUT, "rm")):
                o = put(md, torch.randn(8, 64, N, 512), 0, lay)

                def rs():
                    return ttnn.experimental.reduce_scatter_minimal_async(
                        o,
                        dim=2,
                        multi_device_global_semaphore=ccl.get_rs_ping_pong_semaphore(),
                        num_links=nl,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        topology=topo,
                        cluster_axis=1,
                        barrier_semaphore=ccl.get_barrier_semaphore(),
                    )

                try:
                    R(
                        op=f"rs_min_{nm}_dim2",
                        n=N,
                        links=nl,
                        us=round(traced_us(md, rs, n=7, rep_=3), 1),
                        mb=round(64 * N * 512 * 2 / 1e6, 1),
                    )
                except Exception as e:
                    R(op=f"rs_min_{nm}_dim2", links=nl, err=repr(e)[:300])
                ttnn.deallocate(o)
            o = put(md, torch.randn(8, 64, N, 512), 0, ttnn.TILE_LAYOUT)

            def a2a():
                return ttnn.experimental.all_to_all_async_generic(
                    o, in_dim=1, out_dim=0, num_links=nl, topology=topo, cluster_axis=1
                )

            try:
                R(op="a2a_tile_heads_to_src", n=N, links=nl, us=round(traced_us(md, a2a, n=7, rep_=3), 1))
            except Exception as e:
                R(op="a2a_tile_heads_to_src", links=nl, err=repr(e)[:300])
            ttnn.deallocate(o)
        # --- local passes
        orm = put(md, torch.randn(8, 64, N, 544), 0, ttnn.ROW_MAJOR_LAYOUT)
        R(
            op="tilize_rm_544_to_tile",
            us=round(traced_us(md, lambda: ttnn.to_layout(orm, ttnn.TILE_LAYOUT), n=7, rep_=3), 1),
        )
        osl = lambda: ttnn.slice(orm, [0, 0, 0, 0], [1, 64, N, 512])
        R(op="slice_rm_544_to_512", us=round(traced_us(md, osl, n=7, rep_=3), 1))
        ot = put(md, torch.randn(8, 64, N, 512), 0, ttnn.TILE_LAYOUT)
        sc = put(md, torch.rand(8, 64, N, 1), 0, ttnn.TILE_LAYOUT)
        R(op="mul_bcast_tile_o_scale", us=round(traced_us(md, lambda: ttnn.multiply(ot, sc), n=7, rep_=3), 1))
        sc2 = put(md, torch.rand(8, 64, N, 1), 0, ttnn.ROW_MAJOR_LAYOUT)
        orm2 = put(md, torch.randn(8, 64, N, 512), 0, ttnn.ROW_MAJOR_LAYOUT)
        try:
            R(op="mul_bcast_rm_o_scale", us=round(traced_us(md, lambda: ttnn.multiply(orm2, sc2), n=7, rep_=3), 1))
        except Exception as e:
            R(op="mul_bcast_rm_o_scale", err=repr(e)[:200])
    finally:
        ttnn.close_mesh_device(md)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
