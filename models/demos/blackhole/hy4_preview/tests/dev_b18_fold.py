# B18: Hy4RowsReduce with fold (own shared-expert rows added to the own token block in the reducer): >=5 varying calls vs host sum.
# Env: RP_MESH (1x8 | 4x8), RP_T (1), RP_NRP (16), RP_CALLS (6), RP_FOLD (1), RP_TRACE (1: slope of 10 vs 2 calls)
import json
import os
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import p, pcc
from models.demos.blackhole.hy4_preview.tt.hy4_rows_ar import D, Hy4RowsReduce

T = int(os.environ.get("RP_T", "1"))
NRP = int(os.environ.get("RP_NRP", "16" if T < 4 else "24"))
CALLS = int(os.environ.get("RP_CALLS", "6"))
FOLD = os.environ.get("RP_FOLD", "1") == "1"
OUT = os.environ.get("OUT", "/tmp/b18_fold.json")
shape = tuple(int(v) for v in os.environ.get("RP_MESH", "1x8").split("x"))
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING if shape == (1, 8) else ttnn.FabricConfig.FABRIC_1D)
md = ttnn.open_mesh_device(ttnn.MeshShape(*shape), l1_small_size=16384, trace_region_size=50_000_000)
rows, cols = shape
Tt = 4 * T
res = {"T": T, "NRP": NRP, "fold": FOLD, "pcc": []}
try:
    rr = Hy4RowsReduce(md, T=T, h=2, nrp=NRP)
    g = torch.Generator().manual_seed(11)
    up = lambda P: ttnn.from_torch(
        P,
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, 1), mesh_shape=(rows, cols)),
    )
    groups = (
        [[(r, c) for r in range(4)] for c in range(cols)]
        if rr.mode == "axis0"
        else [[(0, 4 * k + j) for j in range(4)] for k in range(cols // 4)]
    )
    for call in range(CALLS):
        P = torch.randn(rows, cols, Tt, D, generator=g).to(torch.bfloat16)
        F = torch.randn(rows, cols, T, D, generator=g).to(torch.bfloat16)
        part, fold = up(P), (up(F) if FOLD else None)
        o = rr(part, fold=fold)
        ttnn.synchronize_device(md)
        worst = 1.0
        for grp in groups:
            tot = sum(P[r, c].float() for (r, c) in grp)
            for pos, (r, c) in enumerate(grp):
                got = ttnn.to_torch(ttnn.get_device_tensors(o)[r * cols + c])[0, 0].float()[:T]
                exp = tot[pos * T : (pos + 1) * T] + (F[r, c].float() if FOLD else 0)
                worst = min(worst, pcc(got, exp))
                res["pcc"].append(pcc(got, exp))
        p("call", call, "pcc min", worst)
    res["pcc_min"] = min(res["pcc"])
    res["pcc"] = None
    if os.environ.get("RP_TRACE", "1") == "1":
        part = up(torch.randn(rows, cols, Tt, D, generator=g).to(torch.bfloat16))
        fold = up(torch.randn(rows, cols, T, D, generator=g).to(torch.bfloat16)) if FOLD else None

        def tm(n):
            for _ in range(2):
                for _ in range(n):
                    rr(part, fold=fold)
                ttnn.synchronize_device(md)
            tid = ttnn.begin_trace_capture(md, cq_id=0)
            for _ in range(n):
                rr(part, fold=fold)
            ttnn.end_trace_capture(md, tid, cq_id=0)
            ts = []
            for _ in range(15):
                ttnn.synchronize_device(md)
                t0 = time.perf_counter()
                ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(md)
                ts.append((time.perf_counter() - t0) * 1e6)
            ttnn.release_trace(md, tid)
            ts.sort()
            return ts[len(ts) // 2]

        n2, n10 = tm(2), tm(10)
        res["us_per_call"] = round((n10 - n2) / 8, 1)
        p("TRACE us/call", res["us_per_call"])
finally:
    json.dump(res, open(OUT, "w"))
    p("RESULT", json.dumps(res))
    ttnn.close_mesh_device(md)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
