# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""skinny-kernels wiring check: the integrated entry points (hy4_dense.make_shared, Hy4Gate) under HY4_SKINNY=<flags> (env of the process): router + shared expert of one layer-1 token batch in ONE trace
(the decode work before the dispatch), output check vs the torch reference (shared partial summed over the 8 columns; router top-8 set vs the bf16 CPU emulation), traced us.
Run twice: HY4_SKINNY unset (stock, the default) and HY4_SKINNY=shared,router. Env: SK_T=1,4."""
import json
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import ACTS, p, pcc, traced_us
from models.demos.blackhole.hy4_preview.tt.hy4_dense import make_shared
from models.demos.blackhole.hy4_preview.tt.hy4_moe_weights import load_moe_layer
from models.demos.blackhole.hy4_preview.tt.hy4_router import Hy4Gate, route_cpu, set_match

HID = 6144
OUT = os.environ.get("OUT", "/tmp/skinny_wired.json")
TS = [int(t) for t in os.environ.get("SK_T", "1,4").split(",")]
acts = torch.load(ACTS + "acts_l0_l1.pt")
gs = torch.load(ACTS + "l1_gate_shared.pt")
X = torch.cat([h.reshape(-1, HID) for h, y, r in acts["moe"]]).to(torch.bfloat16)
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=100_000_000)
res = {"flags": os.environ.get("HY4_SKINNY", "")}
try:
    W = load_moe_layer(1, load_experts=False)
    Wsh = [W[k][256] for k in ("shared_w0", "shared_w1", "shared_w2")]
    w0, w1, w2 = [w.reshape(w.shape[-2], w.shape[-1]).float() for w in Wsh]
    shared = make_shared(md, *Wsh, decode=True)
    gate = Hy4Gate(md, W["gate_weight"], W["gate_bias"], mode="fused")
    p("flags", res["flags"], "shared", type(shared).__name__, "router skinny", gate._sk is not None)
    _, ids_ref, _ = route_cpu(X, gs["mlp.gate.weight"], gs["mlp.gate.e_score_correction_bias"], emulate_device=True)
    for T in TS:
        n = (X.shape[0] // T) * T
        got_i, errs = [], []
        for s in range(0, n, T):
            t = ttnn.from_torch(X[s : s + T].reshape(1, 1, T, HID), device=md, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(md))
            w_, i_ = gate.forward(t)
            got_i.append(ttnn.to_torch(ttnn.get_device_tensors(i_)[0]).reshape(-1, 8).long())
            if s < 8 * T:
                o = shared.forward(t)
                part = ttnn.to_torch(o, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0)).float()[:, 0, :T].sum(0)
                x = X[s : s + T].float()
                ref = (torch.nn.functional.silu((x @ w0).clamp(max=10.0)) * (x @ w1).clamp(-10, 10)) @ w2
                errs.append(pcc(part, ref))
                ttnn.deallocate(o)
            for q in (w_, i_, t):
                ttnn.deallocate(q)
        sm = set_match(ids_ref[:n], torch.cat(got_i))
        t0 = ttnn.from_torch(X[:T].reshape(1, 1, T, HID), device=md, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(md))
        us_sh = traced_us(md, lambda: shared.forward(t0))
        us_rt = traced_us(md, lambda: gate.forward(t0))
        us_both = traced_us(md, lambda: (gate.forward(t0), shared.forward(t0)))
        res[f"T{T}"] = {"shared_us": us_sh, "router_us": us_rt, "both_us": us_both, "set_match": sm, "shared_pcc_min": min(errs), "shared_pcc_mean": sum(errs) / len(errs)}
        p(f"T={T}: shared {us_sh:.1f} us, router(+select) {us_rt:.1f} us, both in one trace {us_both:.1f} us; router set-match {sm:.4f}; shared PCC vs torch min {min(errs):.6f} mean {sum(errs)/len(errs):.6f}")
finally:
    ttnn.close_mesh_device(md)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    json.dump(res, open(OUT, "w"), indent=1)
