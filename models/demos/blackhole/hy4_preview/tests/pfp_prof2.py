"""pool-prefill: op-level profile of the merge-scale chain and post passes (1x8 slot, no model weights)."""
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
        stats = put(md, torch.rand(8, 1, n, 128), 0, ttnn.ROW_MAJOR_LAYOUT)
        ag = lambda t, d: ttnn.experimental.all_gather_async(
            t,
            dim=d,
            cluster_axis=1,
            mesh_device=md,
            topology=ttnn.Topology.Ring,
            multi_device_global_semaphore=ccl.get_ag_ping_pong_semaphore(),
            num_links=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            barrier_semaphore=ccl.get_barrier_semaphore(),
        )
        T("ag_stats_dim3", lambda: ag(stats, 3))
        T("ag_stats_dim0", lambda: ag(stats, 0))
        sg = ag(stats, 3)
        T("reshape", lambda: ttnn.reshape(sg, [n, 8, 2, 64]))
        x = ttnn.reshape(sg, [n, 8, 2, 64])
        T("permute_rm", lambda: ttnn.permute(x, (1, 2, 0, 3)))
        xp = ttnn.permute(x, (1, 2, 0, 3))
        T("tilize_8x2x512x64", lambda: ttnn.to_layout(xp, ttnn.TILE_LAYOUT))
        xt = ttnn.to_layout(xp, ttnn.TILE_LAYOUT)
        T("slice_m", lambda: ttnn.slice(xt, [0, 0, 0, 0], [8, 1, n, 64]))
        m = ttnn.slice(xt, [0, 0, 0, 0], [8, 1, n, 64])
        l = ttnn.slice(xt, [0, 1, 0, 0], [8, 2, n, 64])
        sink = put(md, torch.rand(8, 1, 1, 64), 0, ttnn.TILE_LAYOUT)
        T("sub_sink", lambda: ttnn.subtract(m, sink))
        d = ttnn.subtract(m, sink)
        T("mul_scale", lambda: ttnn.multiply(d, 0.0625))
        T("exp", lambda: ttnn.exp(d))
        e = ttnn.exp(d)
        T("mul_l", lambda: ttnn.multiply(l, e))
        w = ttnn.multiply(l, e)
        T("sum0", lambda: ttnn.sum(w, dim=0, keepdim=True))
        s = ttnn.sum(w, dim=0, keepdim=True)
        T("add1", lambda: ttnn.add(s, 1.0))
        T("recip", lambda: ttnn.reciprocal(s))
        r = ttnn.reciprocal(s)
        T("mul_recip", lambda: ttnn.multiply(w, r))
        f = ttnn.multiply(w, r)
        oh = put(md, torch.rand(64, 1, 1, 1), 0, ttnn.TILE_LAYOUT)
        T("mul_onehot", lambda: ttnn.multiply(f, oh))
        f2 = ttnn.multiply(f, oh)
        T("sum0_own", lambda: ttnn.sum(f2, dim=0, keepdim=True))
        own = ttnn.sum(f2, dim=0, keepdim=True)
        T("own_to_rm", lambda: ttnn.to_layout(own, ttnn.ROW_MAJOR_LAYOUT))
        orm = ttnn.to_layout(own, ttnn.ROW_MAJOR_LAYOUT)
        T("perm_0321", lambda: ttnn.permute(orm, (0, 3, 2, 1)))
        pm = ttnn.permute(orm, (0, 3, 2, 1))
        T("s_to_tile", lambda: ttnn.to_layout(pm, ttnn.TILE_LAYOUT))
        # direct: scale in [1,64,n,1] tile form via transpose op
        T("transpose_own_tile", lambda: ttnn.transpose(own, 2, 3))
        # o passes
        o_rm = put(md, torch.randn(8, 64, n, 512), 0, ttnn.ROW_MAJOR_LAYOUT)
        T("o_tilize", lambda: ttnn.to_layout(o_rm, ttnn.TILE_LAYOUT))
        o_rm_b = put(md, torch.randn(8, 64, n, 512), 0, ttnn.ROW_MAJOR_LAYOUT)
        T("o_tilize_bfp8_out", lambda: ttnn.to_layout(o_rm_b, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b))
        o_t = ttnn.to_layout(o_rm, ttnn.TILE_LAYOUT)
        s_t = ttnn.to_layout(pm, ttnn.TILE_LAYOUT)
        T("o_mul", lambda: ttnn.multiply(o_t, s_t))
        T("o_mul_bfp8_out", lambda: ttnn.multiply(o_t, s_t, dtype=ttnn.bfloat8_b))
    finally:
        ttnn.close_mesh_device(md)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
