import json

import torch

import ttnn

R = lambda **kw: print("RES " + json.dumps(kw), flush=True)


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=100_000_000)
    try:
        g = torch.Generator().manual_seed(1)
        n = 512
        x = torch.rand(8, 1, n, 64, generator=g) * 8 + 75
        x[3, 0, :5, :] = float("-inf")
        mk = lambda t, lay=ttnn.TILE_LAYOUT, dt=ttnn.bfloat16: ttnn.from_torch(
            t,
            device=md,
            dtype=dt,
            layout=lay,
            mesh_mapper=ttnn.ReplicateTensorToMesh(md),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        d0 = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()
        xt = mk(x.bfloat16())
        mxd = ttnn.max(xt, dim=0, keepdim=True)
        ref = x.bfloat16().float().max(0, keepdim=True).values
        R(op="max_dim0", shape=list(mxd.shape), maxerr=float((d0(mxd) - ref).abs().max()))
        sl = torch.randn(1, 1, 1, 64).clamp(-5, 1.5)
        slt = mk(sl.bfloat16())
        mm = ttnn.maximum(mxd, slt)
        refm = torch.maximum(ref, sl.bfloat16().float())
        R(op="maximum_bcast", shape=list(mm.shape), maxerr=float((d0(mm) - refm).abs().max()))
        d = ttnn.subtract(xt, mm)
        R(
            op="sub_bcast_dim0",
            maxerr=float(
                (
                    d0(d)[torch.isfinite(d0(d))]
                    - (x.bfloat16().float() - refm).expand(8, 1, n, 64)[torch.isfinite(d0(d))]
                )
                .abs()
                .max()
            ),
            maxval=float(d0(d)[torch.isfinite(d0(d))].max()),
        )
        e = ttnn.exp(d)
        R(
            op="exp",
            maxerr=float((d0(e) - torch.exp(d0(d))).abs().max()),
            has_nan=bool(torch.isnan(d0(e)).any()),
            minv=float(d0(e).min()),
        )
    finally:
        ttnn.close_mesh_device(md)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


main()
