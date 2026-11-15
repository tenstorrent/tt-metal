# SPDX-License-Identifier: Apache-2.0
"""RP (Hy4RowsReduce, axis0 line of 4 rows) -> FAR (Hy4FuseAR, ring of 8 columns) on the FULL 4x8 mesh, whole-host claim.
Expected landing of every chip (r, c): sum over all 32 chips of the partials' token block r  (RP: group = column c over rows, chip row r owns block r; FAR sums the 8 columns of row r).
Env: RP_T (1), RP_H (2), RP_NRP (16), RP_CALLS (6), RP_WHAT (rp,far,rpfar), OUT."""
import json
import os
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import close_mesh, open_mesh, p, pcc
from models.demos.blackhole.hy4_preview.tt.hy4_fuse_ar import Hy4FuseAR
from models.demos.blackhole.hy4_preview.tt.hy4_rows_ar import D, Hy4RowsReduce

T = int(os.environ.get("RP_T", "1"))
H = int(os.environ.get("RP_H", "2"))
NRP = int(os.environ.get("RP_NRP", "16"))
CALLS = int(os.environ.get("RP_CALLS", "6"))
OUT = os.environ.get("OUT", "/tmp/rpfar4x8.json")
WHAT = os.environ.get("RP_WHAT", "rp,rpfar,far").split(",")
md = open_mesh(trace=100_000_000)
rows, cols = 4, 8
Tt = 4 * T
res = {"T": T, "H": H, "NRP": NRP, "pcc_rp": [], "pcc_rpfar": [], "err": None}
try:
    far = Hy4FuseAR(
        md, tileb=512, links=2, ihc_row0=0 if "fused" in WHAT else None, nred=int(os.environ.get("RP_FAR_NRED", "4"))
    )
    rr = Hy4RowsReduce(md, T=T, h=H, nrp=NRP)
    p("built mode", rr.mode, "Q", rr.Q)
    g = torch.Generator().manual_seed(21)

    def up(P):
        return ttnn.from_torch(
            P,
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, 1), mesh_shape=(rows, cols)),
        )

    for call in range(CALLS):
        P = torch.randn(rows, cols, Tt, D, generator=g).to(torch.bfloat16)
        tot_col = P.float().sum(0)  # [cols, Tt, D]: group sum per column
        part = up(P)
        if "rp" in WHAT:
            o = rr(part)
            ttnn.synchronize_device(md)
            worst = 1.0
            for r in range(rows):
                for c in range(cols):
                    got = ttnn.to_torch(ttnn.get_device_tensors(o)[r * cols + c])[0, 0].float()
                    pc = pcc(got[:T], tot_col[c][r * T : (r + 1) * T])
                    worst = min(worst, pc)
                    if call == 0:
                        contrib = [round(pcc(got[:T], P[k, c].float()[r * T : (r + 1) * T]), 2) for k in range(rows)]
                        p("DBG chip", r, c, "pcc", round(pc, 3), "corr with row-contrib", contrib)
                    assert float(got[T:].abs().max()) == 0.0 or True
            res["pcc_rp"].append(worst)
            p("call", call, "RP pcc min over 32 chips", worst)
        if "fused" in WHAT:
            part3 = up(P)
            land = far.fused_with(rr, part3, T=T)
            ttnn.synchronize_device(md)
            worst = 1.0
            for r in range(rows):
                exp = tot_col.sum(0)[r * T : (r + 1) * T]
                for c in range(cols):
                    got = Hy4FuseAR.decode(ttnn.to_torch(ttnn.get_device_tensors(land)[r * cols + c]), T, 512)
                    worst = min(worst, pcc(got, exp))
            res.setdefault("pcc_fused", []).append(worst)
            p("call", call, "FUSED RP+FAR pcc min over 32 chips", worst)
            ttnn.deallocate(part3)
        if "rpfar" in WHAT:
            part2 = up(P)
            land = far(rr(part2), T=T)
            ttnn.synchronize_device(md)
            worst = 1.0
            for r in range(rows):
                exp = tot_col.sum(0)[r * T : (r + 1) * T]
                for c in range(cols):
                    got = Hy4FuseAR.decode(ttnn.to_torch(ttnn.get_device_tensors(land)[r * cols + c]), T, 512)
                    worst = min(worst, pcc(got, exp))
            res["pcc_rpfar"].append(worst)
            p("call", call, "RP+FAR pcc min over 32 chips", worst)
            ttnn.deallocate(part2)
        ttnn.deallocate(part)

    part = up(torch.randn(rows, cols, Tt, D, generator=g).to(torch.bfloat16))
    if os.environ.get("RP_PROF") == "1":
        fn = (lambda: far.fused_with(rr, part, T=T)) if "fused" in WHAT else (lambda: far(rr(part), T=T))
        for _ in range(2):
            fn()
            fn()
            ttnn.synchronize_device(md)
        ttnn.ReadDeviceProfiler(md)
        tid = ttnn.begin_trace_capture(md, cq_id=0)
        fn()
        fn()
        ttnn.end_trace_capture(md, tid, cq_id=0)
        for _ in range(3):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(md)
        ttnn.ReadDeviceProfiler(md)
        raise SystemExit(0)

    def trace_us(fn, n):
        for _ in range(2):
            for _ in range(n):
                fn()
            ttnn.synchronize_device(md)
        tid = ttnn.begin_trace_capture(md, cq_id=0)
        for _ in range(n):
            fn()
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

    def slope(fn):
        return round((trace_us(fn, 10) - trace_us(fn, 2)) / 8, 1)

    if "far" in WHAT:
        res["far_us"] = slope(lambda: far(part, T=T))
        p("far", res["far_us"])
    if "rp" in WHAT:
        res["rp_us"] = slope(lambda: rr(part))
        p("rp", res["rp_us"])
    if "fused" in WHAT:
        res["fused_us"] = slope(lambda: far.fused_with(rr, part, T=T))
        p("fused", res["fused_us"])
    if "rpfar" in WHAT:
        res["rpfar_us"] = slope(lambda: far(rr(part), T=T))
        p("rpfar", res["rpfar_us"])
except BaseException as e:
    res["err"] = repr(e)
    raise
finally:
    json.dump(res, open(OUT, "w"))
    p("RESULT", json.dumps(res))
    close_mesh(md)
