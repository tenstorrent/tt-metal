"""B47b QFOLD unit (1x8 mesh): fused_q + separate QPush program vs fused_q(push=...) (push comm cores inside the q-prep program).
Bit equality of the landing tensors and of the q rows over varying inputs; consecutive calls; optional traced timing (TIME=1) of both chains."""
import json
import os
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tt import hy4_fuse_attn as FA
from models.demos.blackhole.hy4_preview.tt import hy4_qpush

NCALL = int(os.environ.get("NCALL", "6"))
S = int(os.environ.get("S", "4"))


def R(**kw):
    print("RES " + json.dumps(kw), flush=True)


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=100_000_000)
    try:
        g = torch.Generator().manual_seed(11)
        rep = ttnn.ReplicateTensorToMesh(md)
        shd = ttnn.ShardTensorToMesh(md, dim=0)
        up = lambda t, m: ttnn.from_torch(
            t.to(torch.bfloat16),
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=m,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        wuk = up(torch.randn(1, 8, 192, 512, generator=g) * 0.1, rep)
        tm = up(torch.randn(1, 1, 32, 32, generator=g), rep)
        pa, pb = hy4_qpush.QPush(md, S, mc=False), hy4_qpush.QPush(md, S, mc=os.environ.get("MCB", "0") == "1")
        pc = hy4_qpush.QPush(md, S, mc=True)
        cat = lambda t: ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0))
        allok = True

        def mk():
            q = up(torch.randn(8, 1, S, 2048, generator=g), shd)
            cos = up(torch.randn(1, 1, S, 64, generator=g), rep)
            sin = up(torch.randn(1, 1, S, 64, generator=g), rep)
            return q, cos, sin

        for it in range(NCALL):
            q, cos, sin = mk()
            oa = FA.fused_q(q, wuk, cos, sin, tm)
            la = pa(oa)
            ob = FA.fused_q(q, wuk, cos, sin, tm, push=pb)
            ttnn.synchronize_device(md)
            a, b = cat(la), cat(pb.land)
            qa_, qb_ = cat(oa), cat(ob)
            eq_land, eq_q = bool(torch.equal(a, b)), bool(torch.equal(qa_, qb_))
            allok &= eq_land and eq_q
            R(event="call", it=it, S=S, land_equal=eq_land, q_equal=eq_q, nz=int((a != 0).sum()))
            if not eq_land:
                d = (a != b).nonzero()
                R(event="diff", n=int(d.shape[0]), first=d[:6].tolist())
        R(event="summary", all_equal=allok)
        if os.environ.get("TIME", "0") == "1":
            q, cos, sin = mk()
            fns = {
                "sep": lambda: pa(FA.fused_q(q, wuk, cos, sin, tm)),
                "fold": lambda: FA.fused_q(q, wuk, cos, sin, tm, push=pb),
                "sepmc": lambda: pc(FA.fused_q(q, wuk, cos, sin, tm)),
                "q": lambda: FA.fused_q(q, wuk, cos, sin, tm),
                "push": lambda: pa(oa),
            }
            NI = 40
            for nm in os.environ.get("ARMS", "sep,fold,sep,fold,q,push").split(","):
                fn = fns[nm]
                for _ in range(2):
                    fn()
                ttnn.synchronize_device(md)
                tid = ttnn.begin_trace_capture(md, cq_id=0)
                for _ in range(NI):
                    fn()
                ttnn.end_trace_capture(md, tid, cq_id=0)
                ttnn.synchronize_device(md)
                ts = []
                for _ in range(6):
                    t0 = time.perf_counter()
                    ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
                    ts.append((time.perf_counter() - t0) / NI * 1e6)
                R(event="time", which=nm, us_per_call=min(ts), all=ts)
                ttnn.release_trace(md, tid)
    finally:
        ttnn.close_mesh_device(md)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


main()
