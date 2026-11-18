"""B47b XPUSH consumer unit (1x8 slot, mode row4): tp24 forward on x_all/route_all (host-built tile/route tensors, the AG path's inputs) vs tp24 with xp=XPush landing: bit equality.
Env: T (tokens per chip, 2 or 4), NCALL, HY4_LAYER, HY4_TP32_CACHE_DIR."""
import json
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tt import hy4_xpush
from models.demos.blackhole.hy4_preview.tt.hy4_moe_tp24 import Tp24Experts

T = int(os.environ.get("T", "4"))
NCALL = int(os.environ.get("NCALL", "4"))
LAYER = int(os.environ.get("HY4_LAYER", "1"))
HID = 6144


def R(**kw):
    print("RES " + json.dumps(kw), flush=True)


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=100_000_000)
    try:
        Tt = 4 * T
        g = torch.Generator().manual_seed(3)
        shd = ttnn.ShardTensorToMesh(md, dim=0)
        tpe = Tp24Experts(md, layer=LAYER)
        xp = hy4_xpush.XPush(md, T)
        cat = lambda t: ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0))
        ok = True
        for it in range(NCALL):
            xs = (torch.randn(8, T, HID, generator=g) * 2).to(torch.bfloat16)  # per chip tokens
            ids = torch.stack([torch.randperm(256, generator=g)[:8] for _ in range(8 * T)]).reshape(8, T, 8)
            sc = torch.rand(8, T, 8, generator=g).to(torch.bfloat16)
            sbits = sc.view(torch.int16).to(torch.int32) & 0xFFFF
            rs = torch.cat([ids.to(torch.int32), sbits], dim=2).reshape(8, 16 * T)  # per chip route words
            xall = torch.zeros(8, 1, Tt, HID, dtype=torch.bfloat16)
            rall = torch.zeros(8, 1, 1, 16 * Tt, dtype=torch.int32)
            for c in range(8):
                gp = c // 4
                xall[c, 0] = xs[4 * gp : 4 * gp + 4].reshape(Tt, HID)
                rall[c, 0, 0] = rs[4 * gp : 4 * gp + 4].reshape(-1)
            xa = ttnn.from_torch(
                xall,
                device=md,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=shd,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ra = ttnn.from_torch(
                rall.reshape(8, 1, 1, 16 * Tt),
                device=md,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=shd,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            xt = ttnn.from_torch(
                xs.reshape(8 * T, 1, 1, HID),
                device=md,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=shd,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            rt = ttnn.from_torch(
                rs.reshape(8, 1, 1, 16 * T),
                device=md,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=shd,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            oa = tpe(xa, ra, Tt)
            ttnn.synchronize_device(md)
            xp(xt, rt)
            ob = tpe(None, None, Tt, xp=xp)
            ttnn.synchronize_device(md)
            A, B = cat(oa), cat(ob)
            eq = bool(torch.equal(A, B))
            ok &= eq
            R(event="call", it=it, equal=eq, nd=int((A != B).sum()), absmax=float(A.float().abs().max()))
        R(event="summary", ok=ok)
    finally:
        ttnn.close_mesh_device(md)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


main()
