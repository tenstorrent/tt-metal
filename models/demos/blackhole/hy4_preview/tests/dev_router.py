# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""moe-perf X22: router variants (matmul fidelity, grid) -- traced us of gate.forward and set-match of the top-8 sets vs the CPU emulation over all real layer-1 tokens (B=4 tokens per call).
Env: SH_VARIANTS=fid names. Only HiFi4 is the exact shipped config."""
import json
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import ACTS, close_mesh, open_mesh, p, traced_us
from models.demos.blackhole.hy4_preview.tt.hy4_moe_weights import load_moe_layer
from models.demos.blackhole.hy4_preview.tt.hy4_router import Hy4Gate, route_cpu, set_match

HID = 6144
OUT = os.environ.get("OUT", "/tmp/router.json")
acts = torch.load(ACTS + "acts_l0_l1.pt")
gs = torch.load(ACTS + "l1_gate_shared.pt")
X = torch.cat([h.reshape(-1, HID) for h, y, r in acts["moe"]]).to(torch.bfloat16)
md = open_mesh()
rows, cols = tuple(md.shape)
res = {}
try:
    W = load_moe_layer(1, load_experts=False)
    gate = Hy4Gate(md, W["gate_weight"], W["gate_bias"], mode="fused")
    _, ids_ref, _ = route_cpu(X, gs["mlp.gate.weight"], gs["mlp.gate.e_score_correction_bias"], emulate_device=True)
    n = (X.shape[0] // rows) * rows
    for name in os.environ.get("SH_VARIANTS", "HiFi4,HiFi2,LoFi").split(","):
        for gridname, grid in (("1x8", (1, 8)),):
            gate.ckc = ttnn.init_device_compute_kernel_config(
                md.arch(),
                math_fidelity=getattr(ttnn.MathFidelity, name),
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )
            got = []
            for s in range(0, n, rows):
                x = X[s : s + rows]
                t = ttnn.from_torch(
                    x.reshape(1, 1, rows, HID),
                    device=md,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(2, None), mesh_shape=(rows, cols)),
                )
                w_, i_ = gate.forward(t, grid=grid)
                dv = ttnn.get_device_tensors(i_)
                got.append(torch.cat([ttnn.to_torch(dv[r * cols]).reshape(-1, 8).long() for r in range(rows)]))
                ttnn.deallocate(w_)
                ttnn.deallocate(i_)
                ttnn.deallocate(t)
            got = torch.cat(got)
            sm = set_match(ids_ref[:n], got)
            x = X[:rows]
            t = ttnn.from_torch(
                x.reshape(1, 1, rows, HID),
                device=md,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(2, None), mesh_shape=(rows, cols)),
            )
            us = traced_us(md, lambda: gate.forward(t, grid=grid))
            res[f"{name}_{gridname}"] = {"us": us, "set_match": sm}
            p(f"router {name} {gridname}: {us:.1f} us  set-match vs bf16-emulation CPU ({n} tokens) {sm:.4f}")
finally:
    close_mesh(md)
    json.dump(res, open(OUT, "w"), indent=1)
