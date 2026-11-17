# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""fusion-around: router on the gathered x with the select merge writing the TP32 route words (HY4_TP32_ROUTE=local) vs the production path
(per-row router -> pack -> row all-gather). Bit-exact route bytes on real layer-1 activations (varying calls) + traced us of both paths. Needs HY4_SKINNY=router. Env FA_T=1 (T per row; Tt=4T<=8), NREP=6."""
import json
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import ACTS, close_mesh, open_mesh, p, traced_us
from models.demos.blackhole.hy4_preview.tt.hy4_moe_tp32_block import Hy4MoeTp32Block
from models.demos.blackhole.hy4_preview.tt.hy4_moe_weights import load_moe_layer
from models.demos.blackhole.hy4_preview.tt.hy4_router import Hy4Gate
from models.demos.gpt_oss.tt.ccl import CCLManager

HID = 6144
T = int(os.environ.get("FA_T", "1"))
NREP = int(os.environ.get("NREP", "6"))
acts = torch.load(ACTS + "acts_l0_l1.pt")
X = torch.cat([h.reshape(-1, HID) for h, y, r in acts["moe"]]).to(torch.bfloat16)
md = open_mesh(trace=300_000_000)
rows, cols = tuple(md.shape)
Tt = T * rows
res = {"T": T, "equal": [], "us": {}}
try:
    ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    W = load_moe_layer(1, load_experts=False)
    gate = Hy4Gate(md, W["gate_weight"], W["gate_bias"], mode="fused")
    assert gate._sk is not None, "HY4_SKINNY=router required"
    blk = Hy4MoeTp32Block(md, ccl, gate, type("Tp24Experts", (), {})(), T)
    comp = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).reshape(-1)

    def mk(rep):
        idx = torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(7 * rep))[:Tt]
        x = X[idx]
        tok = ttnn.from_torch(
            x.reshape(Tt, 1, 1, HID),
            device=md,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows, cols)),
        )
        gt = ttnn.from_torch(
            x.reshape(1, 1, Tt, HID),
            device=md,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(2, None), mesh_shape=(rows, cols)),
        )
        return tok, gt

    def path_a(tok, gt):
        w, idx = gate.forward(gt)
        r = blk._pack_route(w, idx)
        ra = blk._ag_rows(r, 3)
        return ra

    def path_b(tok, gt):
        x_all = ttnn.to_layout(ttnn.reshape(blk._ag_rows(tok, 0), [1, 1, Tt, HID]), ttnn.TILE_LAYOUT)
        return gate.route_words(x_all)

    for rep in range(NREP):
        tok, gt = mk(rep)
        a = path_a(tok, gt)
        b = path_b(tok, gt)
        ttnn.synchronize_device(md)
        ok = True
        for d in (0, 5, 13, 31):
            ta = ttnn.to_torch(ttnn.get_device_tensors(a)[d]).reshape(-1).to(torch.int64)
            tb = ttnn.to_torch(ttnn.get_device_tensors(b)[d]).reshape(-1).to(torch.int64)
            ok &= bool(torch.equal(ta, tb))
            if not torch.equal(ta, tb):
                p("MISMATCH", rep, d, ta.tolist(), tb.tolist())
        res["equal"].append(ok)
        p("EQUAL", rep, ok)
    if os.environ.get("PROF") == "1":
        tok, gt = mk(0)
        xa = ttnn.to_layout(ttnn.reshape(blk._ag_rows(tok, 0), [1, 1, Tt, HID]), ttnn.TILE_LAYOUT)
        for _ in range(4):
            gate.route_words(xa)
        ttnn.synchronize_device(md)
        ttnn.ReadDeviceProfiler(md)
        raise SystemExit(0)
    tok, gt = mk(0)
    res["us"]["path_a_router_pack_agroute"] = round(traced_us(md, lambda: path_a(tok, gt)), 1)
    res["us"]["path_b_agx_tilize_routelocal"] = round(traced_us(md, lambda: path_b(tok, gt)), 1)
    res["us"]["agx_tilize_only"] = round(
        traced_us(md, lambda: ttnn.to_layout(ttnn.reshape(blk._ag_rows(tok, 0), [1, 1, Tt, HID]), ttnn.TILE_LAYOUT)), 1
    )
    xa = ttnn.to_layout(ttnn.reshape(blk._ag_rows(tok, 0), [1, 1, Tt, HID]), ttnn.TILE_LAYOUT)
    res["us"]["route_words_only"] = round(traced_us(md, lambda: gate.route_words(xa)), 1)
    res["us"]["router_only"] = round(traced_us(md, lambda: gate.forward(gt)), 1)
    p("US", res["us"])
finally:
    json.dump(res, open(os.environ.get("OUT", "/tmp/route_local.json"), "w"), indent=1)
    close_mesh(md)
