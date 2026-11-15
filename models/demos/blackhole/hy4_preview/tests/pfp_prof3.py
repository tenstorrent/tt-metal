import json

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import traced_us
from models.demos.gpt_oss.tt.ccl import CCLManager

R = lambda **kw: print("RES " + json.dumps(kw), flush=True)
n = 512


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
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=200_000_000)
    try:
        ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
        T = lambda name, fn, **kw: R(op=name, us=round(traced_us(md, fn, n=3, rep_=3), 1), **kw)
        ag = lambda t, d, nl=2: ttnn.experimental.all_gather_async(
            t,
            dim=d,
            cluster_axis=1,
            mesh_device=md,
            topology=ttnn.Topology.Ring,
            multi_device_global_semaphore=ccl.get_ag_ping_pong_semaphore(),
            num_links=nl,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            barrier_semaphore=ccl.get_barrier_semaphore(),
        )
        for shape, nm in (([8, 1, n, 128], "512x128"), ([8, 1, 32, 128], "32x128"), ([8, 1, 32, 32], "32x32")):
            for lay, ln in ((ttnn.ROW_MAJOR_LAYOUT, "rm"), (ttnn.TILE_LAYOUT, "tile")):
                t = put(md, torch.rand(*shape), 0, lay)
                for nl in (1, 2):
                    T(f"ag_{nm}_{ln}_dim3_nl{nl}", lambda: ag(t, 3, nl))
        # reduce-scatter small: [1,64,512,32] tile (w column)
        o = put(md, torch.rand(8, 64, n, 544), 0, ttnn.TILE_LAYOUT)
        rs = lambda t: ttnn.experimental.reduce_scatter_minimal_async(
            t,
            dim=1,
            multi_device_global_semaphore=ccl.get_rs_ping_pong_semaphore(),
            num_links=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Ring,
            cluster_axis=1,
            barrier_semaphore=ccl.get_barrier_semaphore(),
        )
        T("rs_544_tile", lambda: rs(o))
    finally:
        ttnn.close_mesh_device(md)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
