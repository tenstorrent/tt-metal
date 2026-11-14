# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""skinny-kernels probe: what does the LoFi custom_mm truncate?  SkinnyLinear (o_proj shape, K=2048 per chip, N=6144) with exactly-representable operands vs an fp32 reference:
(A) x full bf16 mantissa, W small-integer / 64 (bfp8-exact); (B) x k/8 (<= 4 significant bits), W same; (C) x full bf16, W k/4 (<= 3 bits); (D) x = x_hi (top 4 bits) and x_lo = x - x_hi: two kernel calls, sum.
Prints relative L2 error of the chip-0 partial vs fp32 and vs the bf16-rounded-product-free reference."""
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import p
from models.demos.blackhole.hy4_preview.tt.hy4_skinny import SkinnyLinear

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=100_000_000)
g = torch.Generator().manual_seed(11)
T = 4
try:
    cases = {}
    Wi = (torch.randint(-63, 64, (16384, 6144), generator=g).float() / 64.0)
    Wc = (torch.randint(-3, 4, (16384, 6144), generator=g).float() / 4.0)
    xa = torch.randn(T, 16384, generator=g).to(torch.bfloat16).float()
    xb = (torch.randint(-16, 17, (T, 16384), generator=g).float() / 8.0)

    def run(W, x):
        sk = SkinnyLinear(md, W)
        tt_x = ttnn.from_torch(x.to(torch.bfloat16).reshape(1, 1, T, 16384), device=md, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(None, 3), mesh_shape=(1, 8)))
        o = sk(tt_x, T)
        ttnn.synchronize_device(md)
        got = ttnn.to_torch(o, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0)).float()[:, 0, :T]  # [8, T, N]
        ref = torch.stack([x[:, 2048 * c : 2048 * (c + 1)].double() @ W[2048 * c : 2048 * (c + 1)].double() for c in range(8)]).float()
        return got, ref

    for name, W, x in (("A x full-bf16, W k/64", Wi, xa), ("B x k/8, W k/64", Wi, xb), ("C x full-bf16, W k/4", Wc, xa)):
        got, ref = run(W, x)
        rel = ((got - ref).norm() / ref.norm()).item()
        p(f"{name}: rel L2 err {rel:.3e}   max abs {(got - ref).abs().max().item():.3e}  (ref rms {ref.pow(2).mean().sqrt().item():.3e})")
    # D: x split
    xh = ((xa * 8).round() / 8)  # top ~4 bits
    xl = xa - xh
    g1, ref = run(Wi, xh)
    g2, _ = run(Wi, xl)
    refa = torch.stack([xa[:, 2048 * c : 2048 * (c + 1)].double() @ Wi[2048 * c : 2048 * (c + 1)].double() for c in range(8)]).float()
    p(f"D x split hi+lo (two calls, W k/64): rel L2 err {(((g1 + g2) - refa).norm() / refa.norm()).item():.3e}; hi alone vs ref(hi) {((g1 - ref).norm() / ref.norm()).item():.3e}")
finally:
    ttnn.close_mesh_device(md)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
