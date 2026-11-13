# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""moe-perf: EP8 x TP{2,4} pseudo-expert layout (tt/hy4_moe_tp.py) vs the stock layout, routed path only (forced real routing, no shared expert): traced us and output PCC between the two.
Env: HY4_MOE_TP=4|2, HY4_T=1, MP_DRAWS=6, MOE_COMPUTE_RING (applies to both blocks), HY4_WEIGHT_CACHE base."""
import json
import os
import statistics

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import ACTS, close_mesh, open_mesh, p, pcc, traced_us
from models.demos.blackhole.hy4_preview.tt import hy4_moe_tp as TPM
from models.demos.blackhole.hy4_preview.tt import hy4_moe_weights as MW
from models.demos.blackhole.hy4_preview.tt.hy4_moe import Hy4MoEBlock

T = int(os.environ.get("HY4_T", "1"))
TP = int(os.environ.get("HY4_MOE_TP", "4"))
DRAWS = int(os.environ.get("MP_DRAWS", "6"))
OUT = os.environ.get("OUT", "/tmp/moetp.json")
HID = 6144
BASE = MW.WEIGHT_CACHE

acts = torch.load(ACTS + "acts_l0_l1.pt")
X = torch.cat([h.reshape(-1, HID) for h, y, r in acts["moe"]])
md = open_mesh()
rows, cols = tuple(md.shape)
res = {
    "T": T,
    "TP": TP,
    "env": {k: v for k, v in os.environ.items() if k.startswith(("MOE_", "DSV41_", "HY4_", "MP_"))},
    "draws": [],
}
try:
    MW.WEIGHT_CACHE = BASE
    Wb = MW.load_moe_layer(1)
    blk = Hy4MoEBlock(md, Wb, batch_per_device=T)
    blk.warmup()
    p("base block ready")
    MW.WEIGHT_CACHE = BASE + f"/tp{TP}"
    os.makedirs(MW.WEIGHT_CACHE, exist_ok=True)
    Wc = MW.load_moe_layer(1)
    Wp = TPM.slice_weights(Wc, TP) if Wc["w0"] is not None else Wc
    p(f"tp weights ready (cold={Wc['w0'] is not None}) cache_dir={Wp['cache_dir']}")
    os.environ["HY4_MOE_TP"] = str(TP)
    blkp = TPM.Hy4MoETPBlock(md, Wp, batch_per_device=T)
    blkp.warmup()
    p("tp block ready")
    B = T * rows
    shard = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows, cols))
    mk = lambda t, dt: ttnn.from_torch(
        t, device=md, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=dt, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=shard
    )
    comp = ttnn.ConcatMesh2dToTensor(md, dims=(2, 3), mesh_shape=(rows, cols))
    for d in range(DRAWS):
        sel = torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(500 + d))[:B]
        x = X[sel].to(torch.bfloat16)
        tt_tok = mk(x.reshape(B, 1, 1, HID), ttnn.bfloat16)
        tt_gate = ttnn.from_torch(
            x.reshape(1, 1, B, HID),
            device=md,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(2, None), mesh_shape=(rows, cols)),
        )
        sc, ix = blk.gate.forward(tt_gate)
        devs_ix, devs_sc = ttnn.get_device_tensors(ix), ttnn.get_device_tensors(sc)
        ids = torch.cat([ttnn.to_torch(devs_ix[r * cols]).reshape(-1, 8).long() for r in range(rows)])
        w = torch.cat([ttnn.to_torch(devs_sc[r * cols]).reshape(-1, 8).float() for r in range(rows)])
        pw, pid = TPM.pseudo_routing(w, ids, TP)
        t_s, t_i = mk(w.reshape(B, 1, 1, 8).to(torch.bfloat16), ttnn.bfloat16), mk(
            ids.reshape(B, 1, 1, 8).to(torch.int32), ttnn.uint16
        )
        t_ps, t_pi = mk(pw.reshape(B, 1, 1, 8 * TP).to(torch.bfloat16), ttnn.bfloat16), mk(
            pid.reshape(B, 1, 1, 8 * TP).to(torch.int32), ttnn.uint16
        )
        run_b = lambda: blk.forward(tt_gate, tt_tok, forced_routing=(t_s, t_i))
        run_p = lambda: blkp.forward(tt_gate, tt_tok, forced_routing=(t_ps, t_pi))
        ob, op_ = run_b(), run_p()
        ttnn.synchronize_device(md)
        gb = (
            ttnn.to_torch(ttnn.to_memory_config(ob, ttnn.DRAM_MEMORY_CONFIG), mesh_composer=comp)
            .reshape(B, HID)
            .float()
        )
        gp = (
            ttnn.to_torch(ttnn.to_memory_config(op_, ttnn.DRAM_MEMORY_CONFIG), mesh_composer=comp)
            .reshape(B, HID)
            .float()
        )
        pc = pcc(gb, gp)
        us_b, us_p = traced_us(md, run_b), traced_us(md, run_p)
        # distinct experts on the busiest chip for this routing
        from models.demos.blackhole.hy4_preview.tt.hy4_moe_place import chip_of

        cnt = {}
        for e in set(ids.flatten().tolist()):
            cnt[chip_of(e)] = cnt.get(chip_of(e), 0) + 1
        mx = max(cnt.values())
        res["draws"].append({"busiest_experts": mx, "base_us": us_b, "tp_us": us_p, "pcc_tp_vs_base": pc})
        p(f"draw {d}: busiest-chip experts {mx}  base {us_b:.1f} us  TP{TP} {us_p:.1f} us  PCC(TP vs base) {pc:.5f}")
        ttnn.deallocate(ob)
        ttnn.deallocate(op_)
    b = [r["base_us"] for r in res["draws"]]
    q = [r["tp_us"] for r in res["draws"]]
    p(f"MEAN base {statistics.mean(b):.1f} us  TP{TP} {statistics.mean(q):.1f} us")
finally:
    close_mesh(md)
    json.dump(res, open(OUT, "w"), indent=1)
