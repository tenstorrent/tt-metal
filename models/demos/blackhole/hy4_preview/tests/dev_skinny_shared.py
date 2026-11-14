# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""skinny-kernels X24: Hy4 shared expert partial, custom bank-local kernel (tt/hy4_skinny.SkinnyShared) vs the stock make_shared op, on a 1x8 slot.
Real layer-1 shared weights + real layer-1 MoE input activations; PCC of the column-summed output vs the stock op and vs the torch reference; traced us of both.
Env: SK_T=1,2,4,8 (rows), SK_VARIANTS (comma list name:key=val;key=val), OUT=json."""
import json
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import ACTS, p, pcc, traced_us
from models.demos.blackhole.hy4_preview.tt.hy4_dense import make_shared
from models.demos.blackhole.hy4_preview.tt.hy4_moe_weights import load_moe_layer
from models.demos.blackhole.hy4_preview.tt.hy4_skinny import SkinnyShared

HID = 6144
OUT = os.environ.get("OUT", "/tmp/skinny_shared.json")
TS = [int(t) for t in os.environ.get("SK_T", "1,2,4,8").split(",")]
VARS = os.environ.get("SK_VARIANTS", "base").split("|")


def parse(v):
    name, _, kv = v.partition(":")
    d = {}
    for item in filter(None, kv.split(";")):
        k, val = item.split("=")
        d[k] = val if k == "remap" else int(val)
    return name, d


acts = torch.load(ACTS + "acts_l0_l1.pt")
X = torch.cat([h.reshape(-1, HID) for h, y, r in acts["moe"]])
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=100_000_000)
cols = 8
res = {"v": {}}
try:
    W = load_moe_layer(1, load_experts=False)
    Wsh = [W[k][256] for k in ("shared_w0", "shared_w1", "shared_w2")]
    w0, w1, w2 = [w.reshape(w.shape[-2], w.shape[-1]).float() for w in Wsh]
    stock = make_shared(md, *Wsh, decode=True)
    sks = {}
    for v in VARS:
        name, kw = parse(v)
        sks[name] = SkinnyShared(md, *Wsh, **kw)
        p(name, 'cores', [(c.x, c.y) for c in sks[name].cores], 'phys', [(q.x, q.y) for q in sks[name].phys])
    for T in TS:
        x = X[torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(7))[:T]].to(torch.bfloat16)
        tt_x = ttnn.from_torch(
            x.reshape(1, 1, T, HID), device=md, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(md),
        )
        ref = (torch.nn.functional.silu((x.float() @ w0).clamp(max=10.0)) * (x.float() @ w1).clamp(-10, 10)) @ w2
        comp = ttnn.ConcatMeshToTensor(md, dim=0)

        def summed(o):
            t = ttnn.to_torch(o, mesh_composer=comp).float()  # [8, 1, rows, HID]
            return t[:, 0, :T].sum(0)

        if os.environ.get("SK_PROF"):
            for name, sk in sks.items():
                for _ in range(4):
                    o = sk(tt_x, T)
                    ttnn.synchronize_device(md)
                ttnn.ReadDeviceProfiler(md)
                p("profiler read", name)
            continue
        o = stock.forward(tt_x)
        ttnn.synchronize_device(md)
        s_out = summed(o)
        ttnn.deallocate(o)
        us_s = traced_us(md, lambda: stock.forward(tt_x))
        c_s = pcc(s_out, ref)
        p(f"T={T} stock    {us_s:6.1f} us  PCC vs torch {c_s:.6f}")
        res["v"][f"T{T}_stock"] = {"us": us_s, "pcc_ref": c_s}
        for name, sk in sks.items():
            try:
                o = sk(tt_x, T)
                ttnn.synchronize_device(md)
                k_out = summed(o)
                us = traced_us(md, lambda: sk(tt_x, T))
                r = {"us": us, "pcc_stock": pcc(k_out, s_out), "pcc_ref": pcc(k_out, ref)}
                res["v"][f"T{T}_{name}"] = r
                p(f"T={T} skinny {name:8s} {us:6.1f} us  PCC vs stock {r['pcc_stock']:.6f}  vs torch {r['pcc_ref']:.6f}")
            except Exception as e:
                res["v"][f"T{T}_{name}"] = {"err": str(e)[:300]}
                p(f"T={T} skinny {name} ERROR {str(e)[:300]}")
finally:
    ttnn.close_mesh_device(md)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    json.dump(res, open(OUT, "w"), indent=1)
