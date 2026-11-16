"""B22 step 4: fused_back_oproj (back phase + skinny o-proj in ONE program) vs fused_back -> SkinnyLinear (two programs): exact equality over varying inputs (all 8 devices) + traced timing."""
import json
import os
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tt import hy4_fuse_attn as FA
from models.demos.blackhole.hy4_preview.tt.hy4_skinny import SkinnyLinear

NCALL = int(os.environ.get("NCALL", "6"))


def R(**kw):
    print("RES " + json.dumps(kw), flush=True)


def main():
    kw = {}
    if int(os.environ.get("HY4_RING_KB", "0")):
        kw["worker_l1_size"] = 1460864 - int(os.environ["HY4_RING_KB"]) * 1024
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=100_000_000, **kw)
    try:
        g = torch.Generator().manual_seed(5)
        rep = ttnn.ReplicateTensorToMesh(md)
        shd = ttnn.ShardTensorToMesh(md, dim=1)
        T_ = lambda t, m=rep: ttnn.from_torch(
            t,
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=m,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        w = torch.randn(8 * 2048, 6144, generator=g) * 0.02
        sk = SkinnyLinear(md, w)
        wuv = T_(torch.randn(1, 64, 512, 256, generator=g) * 0.05, shd)  # per device [1,8,512,256]
        allok = True
        for S in [int(v) for v in os.environ.get("SS", "1,2,4").split(",")]:
            for it in range(NCALL):
                mag = 10.0 ** float(torch.randint(-1, 2, (1,), generator=g))
                olat = T_(torch.randn(1, 64, S, 512, generator=g) * mag, shd)
                gt = T_(torch.randn(1, 1, S, 2048, generator=g))
                for gfold in (True, False):
                    og = FA.fused_back(olat, wuv, gt, gfold)
                    a = sk(og, S)
                    b = FA.fused_back_oproj(olat, wuv, gt, gfold, sk)
                    ttnn.synchronize_device(md)
                    ta = ttnn.to_torch(a, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0))
                    tb = ttnn.to_torch(b, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0))
                    eq = bool(torch.equal(ta, tb))
                    allok &= eq
                    R(
                        event="call",
                        S=S,
                        it=it,
                        mag=mag,
                        gfold=gfold,
                        equal=eq,
                        maxabs=float((ta.float() - tb.float()).abs().max()),
                        nonfinite=int((~torch.isfinite(tb.float())).sum()),
                    )
        R(event="summary", all_equal=allok)
        if os.environ.get("TIME", "0") == "1":
            S = 1
            olat = T_(torch.randn(1, 64, S, 512, generator=g), shd)
            gt = T_(torch.randn(1, 1, S, 2048, generator=g))

            def stock():
                og = FA.fused_back(olat, wuv, gt, True)
                a = sk(og, S)
                ttnn.deallocate(og)
                ttnn.deallocate(a)

            def fused():
                ttnn.deallocate(FA.fused_back_oproj(olat, wuv, gt, True, sk))

            NI = 40
            for nm, fn in (("stock", stock), ("fused", fused)):
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
    finally:
        ttnn.close_mesh_device(md)


main()
