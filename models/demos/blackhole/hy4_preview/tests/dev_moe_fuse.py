# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""B19: shared expert folded into the tp24 program (HY4_MOE_FUSE) vs the current chain (skinny shared on per-row h -> mask-matmul fold) on real layer weights.
Per call: x of Tt = 4T real tokens, both chains run through the full block forward (RS cols, RS rows, AG, select) -> [B, HID]; PCC / max |diff| between the chains + traced us of
forward_partial (the part this task changes). Env: FA_T=1 (T per row), NREP=6, HY4_LAYER=1, HY4_TP32_ROUTE=local, HY4_SKINNY=shared,router, HY4_TP32_CACHE_DIR."""
import json
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import ACTS, close_mesh, open_mesh, p, pcc, traced_us
from models.demos.blackhole.hy4_preview.tt import hy4_dense as D
from models.demos.blackhole.hy4_preview.tt.hy4_moe_tp24 import Tp24Experts
from models.demos.blackhole.hy4_preview.tt.hy4_moe_tp32_block import Hy4MoeTp32Block
from models.demos.blackhole.hy4_preview.tt.hy4_moe_weights import load_moe_layer
from models.demos.blackhole.hy4_preview.tt.hy4_router import Hy4Gate
from models.demos.blackhole.hy4_preview.tt.hy4_skinny import SkinnyShared
from models.demos.gpt_oss.tt.ccl import CCLManager

HID = 6144
T = int(os.environ.get("FA_T", "1"))
NREP = int(os.environ.get("NREP", "6"))
LAYER = int(os.environ.get("HY4_LAYER", "1"))
acts = torch.load(ACTS + "acts_l0_l1.pt")
X = torch.cat([h.reshape(-1, HID) for h, y, r in acts["moe"]]).to(torch.bfloat16)
md = open_mesh(trace=300_000_000)
rows, cols = tuple(md.shape)
Tt = T * rows
res = {"T": T, "pcc": [], "maxdiff": [], "us": {}}
try:
    ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    W = load_moe_layer(LAYER, load_experts=False)
    gate = Hy4Gate(md, W["gate_weight"], W["gate_bias"], mode="fused")
    tpe = Tp24Experts(md, layer=LAYER)
    sw = [W[k][256] for k in ("shared_w0", "shared_w1", "shared_w2")]
    shared_old = D.make_shared(md, *sw, limit=D.NO_CLAMP, decode=True)
    blk = Hy4MoeTp32Block(md, ccl, gate, tpe, T)
    blk_f = Hy4MoeTp32Block(md, ccl, gate, tpe, T)
    blk_f.shared_fuse = SkinnyShared(md, sw[0], sw[1], sw[2] * 0.25, limit=D.NO_CLAMP)
    comp = ttnn.ConcatMesh2dToTensor(md, dims=(2, 3), mesh_shape=(rows, cols))

    def mk(rep):
        idx = torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(11 * rep + 3))[:Tt]
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

    def old(tok, gt):
        part = shared_old.forward(gt)
        o = blk.forward(gt, tok, fold=part)
        ttnn.deallocate(part)
        return o

    def new(tok, gt):
        return blk_f.forward(gt, tok)

    for rep in range(NREP):
        tok, gt = mk(rep)
        a = old(tok, gt)
        b = new(tok, gt)
        ttnn.synchronize_device(md)
        ta = (
            ttnn.to_torch(ttnn.to_memory_config(a, ttnn.DRAM_MEMORY_CONFIG), mesh_composer=comp)
            .reshape(Tt, HID)
            .float()
        )
        tb = (
            ttnn.to_torch(ttnn.to_memory_config(b, ttnn.DRAM_MEMORY_CONFIG), mesh_composer=comp)
            .reshape(Tt, HID)
            .float()
        )
        pc = pcc(ta, tb)
        md_ = float((ta - tb).abs().max())
        res["pcc"].append(pc), res["maxdiff"].append(md_)
        p(f"rep {rep}: PCC fused vs chain {pc:.7f} max|diff| {md_:.4g} (|a| max {float(ta.abs().max()):.4g})")
        ttnn.deallocate(a), ttnn.deallocate(b)
    tok, gt = mk(0)
    part = shared_old.forward(gt)
    res["us"]["shared_skinny"] = round(traced_us(md, lambda: shared_old.forward(gt)), 1)
    res["us"]["chain_forward_partial"] = round(traced_us(md, lambda: blk.forward_partial(gt, tok, fold=part)), 1)
    res["us"]["chain_total_with_shared"] = round(
        traced_us(md, lambda: blk.forward_partial(gt, tok, fold=shared_old.forward(gt))), 1
    )
    res["us"]["fused_forward_partial"] = round(traced_us(md, lambda: blk_f.forward_partial(gt, tok)), 1)
    res["us"]["routed_only_forward_partial"] = round(traced_us(md, lambda: blk.forward_partial(gt, tok)), 1)
    x_all = ttnn.to_layout(ttnn.reshape(blk._ag_rows(tok, 0), [1, 1, Tt, HID]), ttnn.TILE_LAYOUT)
    route_all = gate.route_words(x_all)
    ntb = lambda f: (lambda: f()[0] if isinstance(f(), tuple) else f())
    res["us"]["k_tp24_only"] = round(traced_us(md, lambda: tpe(x_all, route_all, Tt)), 1)
    res["us"]["k_tp24_plus_shared"] = round(
        traced_us(md, lambda: tpe(x_all, route_all, Tt, shared=blk_f.shared_fuse)), 1
    )
    res["us"]["router_local"] = round(traced_us(md, lambda: gate.route_words(x_all)), 1)
    res["us"]["agx_tilize"] = round(
        traced_us(md, lambda: ttnn.to_layout(ttnn.reshape(blk._ag_rows(tok, 0), [1, 1, Tt, HID]), ttnn.TILE_LAYOUT)), 1
    )
    p("US", res["us"])
finally:
    json.dump(res, open(os.environ.get("OUT", "/tmp/moe_fuse.json"), "w"), indent=1)
    close_mesh(md)
