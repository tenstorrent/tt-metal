# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_1C_ATTN_NR: q/k RMSNorm + RoPE as one op (ops/attn_nr) against today's LEAN+QGVIEW chain (rms_norm on
[H,1,D], permute+reshape, k reshape + rms_norm, rope_lean_ttnn x2) and a float64 reference from the same bf16
inputs. The op keeps fp32 between norm and RoPE, so it is not bit-exact; the bar is that it is at least as close to
float64 as the chain (max and mean |d|), with output magnitude and shapes printed and asserted. Runs twice on the
same persistent outputs to check they are fully overwritten, with k_out in the model's height-sharded L1 layout.

    pytest models/demos/blackhole/bonsai2/tp1/test_attn_nr.py -s
"""
import pytest
import torch

import ttnn
from models.demos.blackhole.bonsai2.tp1.ops.attn_nr.op import alloc_outputs, attn_nr
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_attention import (
    _rope_full,
    _rope_rotate_matrix,
    rope_lean_ttnn,
)

H, HK, D, RD, EPS = 24, 4, 256, 64, 1e-6


def ref64(x, w, cos, sin):
    """x [h, D] -> RMSNorm(x) * w, then RoPE on the first RD dims (rotate_half), in float64."""
    x, w, cos, sin = x.double(), w.double(), cos.double(), sin.double()
    n = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w
    r, p = n[:, :RD], n[:, RD:]
    rh = torch.cat([-r[:, RD // 2 :], r[:, : RD // 2]], -1)
    return torch.cat([r * cos + rh * sin, p], -1)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("pos,scale", [(0, 1.0), (7, 1.0), (3001, 1.0), (3001, 30.0)])
def test_attn_nr(mesh_device, pos, scale):
    L1 = ttnn.L1_MEMORY_CONFIG
    kmc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}), [32, D],
                       ttnn.ShardOrientation.ROW_MAJOR),
    )
    outs = alloc_outputs(mesh_device, H, HK, D, kmc)

    def dev(t, mc=L1):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=mc)

    for rep in range(2):
        torch.manual_seed(pos * 10 + rep)
        qg_t = (torch.randn(1, 1, H * 2 * D) * scale).to(torch.bfloat16)
        k_t = (torch.randn(1, 1, HK * D) * scale).to(torch.bfloat16)
        wq_t, wk_t = (torch.rand(D) + 0.5).to(torch.bfloat16), (torch.rand(D) + 0.5).to(torch.bfloat16)
        inv = 1.0 / (1e7 ** (torch.arange(0, RD, 2).float() / RD))
        emb = torch.cat([pos * inv, pos * inv])
        cos_t, sin_t = emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)

        qg = dev(qg_t)
        qg_h = ttnn.experimental.view(qg, ttnn.Shape([H, 1, 2 * D]), ttnn.Shape([H, ttnn.TILE_SIZE, 2 * D]))
        q_h = ttnn.slice(qg_h, (0, 0, 0), (H, 1, D))
        k = dev(k_t)
        wq, wk = dev(wq_t, ttnn.DRAM_MEMORY_CONFIG), dev(wk_t, ttnn.DRAM_MEMORY_CONFIG)
        cos, sin = dev(cos_t.reshape(1, 1, RD)), dev(sin_t.reshape(1, 1, RD))

        # today's chain
        q0 = ttnn.reshape(ttnn.permute(ttnn.rms_norm(q_h, weight=wq, epsilon=EPS), (1, 0, 2)), [1, 1, H, D])
        k0 = ttnn.rms_norm(ttnn.reshape(k, [1, 1, HK, D]), weight=wk, epsilon=EPS)
        cf, sf = _rope_full(cos, sin, D)
        P = _rope_rotate_matrix(mesh_device, RD, D)
        q0 = ttnn.to_torch(rope_lean_ttnn(q0, cf, sf, P, memory_config=ttnn.DRAM_MEMORY_CONFIG)).float()
        k0 = ttnn.to_torch(rope_lean_ttnn(k0, cf, sf, P)).float()

        q1, k1 = attn_nr(q_h, k, wq, wk, cos, sin, *outs, EPS, RD)
        assert q1.memory_config().buffer_type == ttnn.BufferType.DRAM and k1.memory_config() == kmc
        q1, k1 = ttnn.to_torch(q1).float(), ttnn.to_torch(k1).float()

        qr = ref64(qg_t.float().reshape(H, 2 * D)[:, :D], wq_t, cos_t.float(), sin_t.float()).reshape(1, 1, H, D)
        kr = ref64(k_t.float().reshape(HK, D), wk_t, cos_t.float(), sin_t.float()).reshape(1, 1, HK, D)
        for name, r, a, b in (("q", qr, q0, q1), ("k", kr, k0, k1)):
            assert a.shape == b.shape == r.shape, (a.shape, b.shape, r.shape)
            ea, eb = (a.double() - r).abs(), (b.double() - r).abs()
            print(f"pos {pos} scale {scale} rep {rep} {name}: |ref| max {r.abs().max():.3f} mean {r.abs().mean():.3f}  "
                  f"|fused| max {b.abs().max():.3f}  chain-vs-f64 max {ea.max():.3e} mean {ea.mean():.3e}  "
                  f"fused-vs-f64 max {eb.max():.3e} mean {eb.mean():.3e}  fused==chain {100 * (a == b).float().mean():.1f}%")
            assert torch.isfinite(b).all()
            assert (b.abs().mean() / r.abs().mean() - 1).abs() < 0.01  # magnitude: not a scaled or zero output
            assert eb.mean() <= ea.mean() * 1.05 and eb.max() <= ea.max() * 1.05
