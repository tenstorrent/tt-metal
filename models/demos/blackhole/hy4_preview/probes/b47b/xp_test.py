"""B47b XPUSH landing unit (1x8 mesh, mode row4 = groups of 4 chips 4g..4g+3): landing content vs host layout; consecutive calls."""
import json
import os
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tt import hy4_xpush

T = int(os.environ.get("T", "4"))
NCALL = int(os.environ.get("NCALL", "4"))


def R(**kw):
    print("RES " + json.dumps(kw), flush=True)


def expect(xs, rs, T):
    """xs [4,T,6144] bf16 (group sources) -> xland bytes [4*NPX, 2048] bf16, rland [16*4T] int32"""
    npx = 3 * T
    blk = torch.zeros(4, 8, 24, 2, T, 16, dtype=torch.bfloat16)
    for i in range(4):
        v = xs[i].reshape(T, 8, 24, 2, 16)  # r, b, j, f, c
        blk[i] = v.permute(1, 2, 3, 0, 4)
    return blk.reshape(4 * npx, 2048), torch.cat([rs[i].flatten() for i in range(4)])


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=100_000_000)
    try:
        g = torch.Generator().manual_seed(5)
        shd = ttnn.ShardTensorToMesh(md, dim=0)
        xp = hy4_xpush.XPush(md, T)
        cat = lambda t: ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0))
        ok = True
        for it in range(NCALL):
            xs = torch.randn(8, T, 1, 1, 6144, generator=g).to(torch.bfloat16)
            rs = torch.randint(0, 2**30, (8, 1, 1, 1, 16 * T), generator=g, dtype=torch.int32)
            x = ttnn.from_torch(
                xs.reshape(8 * T, 1, 1, 6144),
                device=md,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=shd,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            r = ttnn.from_torch(
                rs.reshape(8, 1, 1, 16 * T),
                device=md,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=shd,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            xl, rl = xp(x, r)
            ttnn.synchronize_device(md)
            XL = cat(xl).reshape(8, 4 * xp.npx, 2048)
            RL = cat(rl).reshape(8, -1)
            for c in range(8):
                grp = c // 4
                ex, er = expect(
                    xs[4 * grp : 4 * grp + 4].reshape(4, T, 6144), rs[4 * grp : 4 * grp + 4].reshape(4, -1), T
                )
                e1, e2 = bool(torch.equal(XL[c], ex)), bool(torch.equal(RL[c], er))
                ok &= e1 and e2
                if not (e1 and e2):
                    R(event="bad", it=it, chip=c, x=e1, route=e2, ndiff=int((XL[c] != ex).sum()))
            R(event="call", it=it, ok=ok)
        R(event="summary", ok=ok)
        if os.environ.get("TIME", "0") == "1":
            NI = 40
            for _ in range(2):
                xp(x, r)
            ttnn.synchronize_device(md)
            tid = ttnn.begin_trace_capture(md, cq_id=0)
            for _ in range(NI):
                xp(x, r)
            ttnn.end_trace_capture(md, tid, cq_id=0)
            ttnn.synchronize_device(md)
            ts = []
            for _ in range(6):
                t0 = time.perf_counter()
                ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
                ts.append((time.perf_counter() - t0) / NI * 1e6)
            R(event="time", us=min(ts))
        if os.environ.get("PROF", "0") == "1":
            for _ in range(2):
                xp(x, r)
            ttnn.synchronize_device(md)
            tid = ttnn.begin_trace_capture(md, cq_id=0)
            xp(x, r)
            ttnn.end_trace_capture(md, tid, cq_id=0)
            ttnn.synchronize_device(md)
            ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
            ttnn.ReadDeviceProfiler(md)
            ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
            ttnn.ReadDeviceProfiler(md)
            R(event="prof")
    finally:
        ttnn.close_mesh_device(md)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


main()
