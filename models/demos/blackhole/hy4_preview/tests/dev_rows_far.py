# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4RowsReduce -> Hy4FuseAR composition on ONE 1x8 slot (row4 mode: two groups of 4 chips; FAR sums the 8 per-chip RP outputs, rows 0..T-1).
Expected (host): FAR landing of every chip = sum over the 8 chips c of RPout[c], RPout[c] = (group sum)[pos(c) T : (pos(c)+1) T].
Gates: PCC >= 0.9999 of the landed rows on all 8 chips, >= 6 varying-input calls; trace slope of RP -> FAR vs FAR alone (ping-pong RP, FAR self-ordered).
Env: RP_T, RP_H, RP_NRP, HY4_RP_SUB, RP_CALLS, OUT."""
import json
import os
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import p, pcc
from models.demos.blackhole.hy4_preview.tt.hy4_fuse_ar import Hy4FuseAR
from models.demos.blackhole.hy4_preview.tt.hy4_rows_ar import D, Hy4RowsReduce

T = int(os.environ.get("RP_T", "1"))
H = int(os.environ.get("RP_H", "2"))
NRP = int(os.environ.get("RP_NRP", "16"))
CALLS = int(os.environ.get("RP_CALLS", "6"))
OUT = os.environ.get("OUT", "/tmp/rows_far.json")
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=100_000_000)
rows, cols = 1, 8
Tt = 4 * T
res = {"T": T, "H": H, "NRP": NRP, "pcc": []}
try:
    far = Hy4FuseAR(md, tileb=512, links=2)
    rr = Hy4RowsReduce(md, T=T, h=H, nrp=NRP)
    g = torch.Generator().manual_seed(11)

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
        P = torch.randn(1, 8, Tt, D, generator=g).to(torch.bfloat16)
        part = up(P)
        mid = rr(part)
        land = far(mid, T=T)
        ttnn.synchronize_device(md)
        rpout = torch.zeros(8, T, D)
        for grp in range(2):
            tot = sum(P[0, 4 * grp + k].float() for k in range(4))
            for pos in range(4):
                rpout[4 * grp + pos] = tot[pos * T : (pos + 1) * T]
        exp = rpout.sum(0)
        for c in range(8):
            lh = ttnn.to_torch(ttnn.get_device_tensors(land)[c])
            got = Hy4FuseAR.decode(lh, T, 512)
            res["pcc"].append(pcc(got, exp))
        p("call", call, "pcc min over 8 chips", min(res["pcc"][-8:]))
        ttnn.deallocate(part)
    res["pcc_min"] = min(res["pcc"])
    res["pcc"] = None

    part = up(torch.randn(1, 8, Tt, D, generator=g).to(torch.bfloat16))

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
        for _ in range(12):
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

    res["far_alone_us"] = slope(lambda: far(part, T=T))
    res["rp_us"] = slope(lambda: rr(part))
    res["rp_far_us"] = slope(lambda: far(rr(part), T=T))
    p("TRACE far alone", res["far_alone_us"], "rp alone", res["rp_us"], "rp+far", res["rp_far_us"])
finally:
    json.dump(res, open(OUT, "w"))
    p("RESULT", json.dumps(res))
    ttnn.close_mesh_device(md)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
