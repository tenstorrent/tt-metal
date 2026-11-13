# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""S1: dense layer 0 (SwiGLU 6144->18432->6144, TP8 bfp8) and layer-1 shared expert on REAL activations, decode (T=1,4,16) and prefill (T=512) vs the fp32 oracle."""
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import ACTS, close_mesh, dev0, open_mesh, p, pcc, rep
from models.demos.blackhole.hy4_preview.tt import hy4_dense as D

acts = torch.load(ACTS + "acts_l0_l1.pt")
md = open_mesh()
try:
    # ---- dense layer 0
    w = torch.load(ACTS + "l0_dense.pt")
    t = lambda m: m.T.contiguous().to(torch.bfloat16)[None, None]
    w0, w1, w2 = t(w["mlp.gate_proj.weight"]), t(w["mlp.up_proj.weight"]), t(w["mlp.down_proj.weight"])
    h_all = torch.cat([h.reshape(-1, 6144) for h, y in acts["dense"]])
    y_all = torch.cat([y.reshape(-1, 6144) for h, y in acts["dense"]])
    longest = max(acts["dense"], key=lambda hy: hy[0].shape[1])
    hl, yl = longest[0].reshape(-1, 6144), longest[1].reshape(-1, 6144)
    for decode, Ts in ((True, (1, 4, 16)), (False, (512,))):
        t0 = time.time()
        dense = D.make_dense(md, w0, w1, w2, decode=decode)
        p(f"dense decode={decode} built {time.time()-t0:.1f}s")
        for T in Ts:
            src_h, src_y = (h_all, y_all) if T <= 32 else (hl, yl)
            n = min(T, src_h.shape[0])
            idx = (
                torch.arange(0, n)
                if T > 32
                else torch.randperm(src_h.shape[0], generator=torch.Generator().manual_seed(1))[:T]
            )
            x = src_h[idx].to(torch.bfloat16)
            if x.shape[0] < T:
                p(f"dense T={T}: only {x.shape[0]} real tokens available")
                T = x.shape[0]
            tx = rep(md, x.reshape(1, 1, T, 6144))
            for it in range(3):
                part = dense.forward(tx)
                full = D.allreduce_cols(part)
                ttnn.synchronize_device(md)
            got = dev0(full).reshape(T, 6144).float()
            p(
                f"dense layer0 decode={decode} T={T}: PCC {pcc(got, src_y[idx]):.6f}  maxabs {(got-src_y[idx]).abs().max():.4f}  ref rms {src_y[idx].pow(2).mean().sqrt():.4f}"
            )
            # all devices identical?
            devs = ttnn.get_device_tensors(full)
            p(
                f"   replicated check: dev31 vs dev0 maxabs {(ttnn.to_torch(devs[31]).float().reshape(T,6144)-got).abs().max():.4f}"
            )
            ttnn.deallocate(tx)
        del dense
    # ---- layer-1 shared expert (limit 10)
    g = torch.load(ACTS + "l1_gate_shared.pt")
    s0, s1, s2 = (
        t(g["mlp.shared_experts.gate_proj.weight"]),
        t(g["mlp.shared_experts.up_proj.weight"]),
        t(g["mlp.shared_experts.down_proj.weight"]),
    )
    X = torch.cat([h.reshape(-1, 6144) for h, y, r in acts["moe"]])
    ref = lambda x: (torch.nn.functional.silu(x.float() @ s0[0, 0].float().clamp(-1e9, 1e9)) if False else None)

    def sh_ref(x):
        gg = (x.float() @ g["mlp.shared_experts.gate_proj.weight"].T).clamp(max=10.0)
        uu = (x.float() @ g["mlp.shared_experts.up_proj.weight"].T).clamp(-10, 10)
        return (torch.nn.functional.silu(gg) * uu) @ g["mlp.shared_experts.down_proj.weight"].T

    for decode, Ts in ((True, (1, 4, 16)), (False, (512,))):
        sh = D.make_shared(md, s0, s1, s2, decode=decode)
        for T in Ts:
            idx = torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(2))[:T]
            x = X[idx].to(torch.bfloat16)
            tx = rep(md, x.reshape(1, 1, T, 6144))
            for it in range(3):
                full = D.allreduce_cols(sh.forward(tx))
                ttnn.synchronize_device(md)
            got = dev0(full).reshape(T, 6144).float()
            r_ = sh_ref(x)
            p(
                f"shared expert L1 decode={decode} T={T}: PCC {pcc(got, r_):.6f} maxabs {(got-r_).abs().max():.4f} ref rms {r_.pow(2).mean().sqrt():.4f}"
            )
            ttnn.deallocate(tx)
        del sh
finally:
    close_mesh(md)
