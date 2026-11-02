# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_1C_ATTN_LEAN equivalence: RoPE on [1,1,H,D] via rope_lean_ttnn (x*cos_full + (x@P)*sin_full) against
today's path (transpose to [1,H,1,D], apply_rotary_pos_emb_ttnn slice/neg/concat, transpose back), for the
model's q (24 heads) and k (4 heads), head_dim 256, rotary_dim 64. Expect bit-exact.

    pytest models/demos/blackhole/bonsai2/tp1/test_attn_lean.py -s
"""
import pytest
import torch

import ttnn
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_attention import (
    _rope_full,
    _rope_rotate_matrix,
    apply_rotary_pos_emb_ttnn,
    rope_lean_ttnn,
)

D, RD = 256, 64


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("pos", [0, 7, 3001])
def test_attn_lean(mesh_device, pos):
    torch.manual_seed(pos)

    def dev(t):
        return ttnn.from_torch(t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)

    inv = 1.0 / (1e7 ** (torch.arange(0, RD, 2).float() / RD))
    f = pos * inv
    emb = torch.cat([f, f])
    cos, sin = dev(emb.cos().reshape(1, 1, RD)), dev(emb.sin().reshape(1, 1, RD))
    q, k = dev(torch.randn(1, 1, 24, D)), dev(torch.randn(1, 1, 4, D))

    qt, kt = ttnn.transpose(q, 1, 2), ttnn.transpose(k, 1, 2)
    q0, k0 = apply_rotary_pos_emb_ttnn(qt, kt, ttnn.reshape(cos, [1, 1, 1, RD]), ttnn.reshape(sin, [1, 1, 1, RD]))
    q0, k0 = ttnn.to_torch(ttnn.transpose(q0, 1, 2)).float(), ttnn.to_torch(ttnn.transpose(k0, 1, 2)).float()

    cf, sf = _rope_full(cos, sin, D)
    P = _rope_rotate_matrix(mesh_device, RD, D)
    q1 = ttnn.to_torch(rope_lean_ttnn(q, cf, sf, P)).float()
    k1 = ttnn.to_torch(rope_lean_ttnn(k, cf, sf, P)).float()
    for name, a, b in (("q", q0, q1), ("k", k0, k1)):
        exact = (a == b).float().mean().item()
        print(f"pos {pos} {name}: shape {tuple(b.shape)} bit-exact {100 * exact:.2f}%  max|d| {(a - b).abs().max().item():.3e}")
        assert exact == 1.0


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_attn_qgview(mesh_device):
    """QWEN36_1C_ATTN_QGVIEW: the q|gate split by view + slices + per-row norm + permute, against reshape + chunk +
    norm + reshape. Expect q and gate bit-exact."""
    torch.manual_seed(1)
    H = 24
    L1 = ttnn.L1_MEMORY_CONFIG
    qg_t = torch.randn(1, 1, H * 2 * D).to(torch.bfloat16)
    w = ttnn.from_torch(torch.rand(D) + 0.5, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    qg = ttnn.from_torch(qg_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=L1)

    q0, g0 = ttnn.chunk(ttnn.reshape(qg, [1, 1, H, 2 * D]), 2, dim=-1)
    q0 = ttnn.to_torch(ttnn.rms_norm(q0, weight=w, epsilon=1e-6)).float()
    g0 = ttnn.to_torch(ttnn.reshape(g0, [1, 1, H * D])).float()

    qg_h = ttnn.experimental.view(qg, ttnn.Shape([H, 1, 2 * D]), ttnn.Shape([H, ttnn.TILE_SIZE, 2 * D]))
    q_h = ttnn.slice(qg_h, (0, 0, 0), (H, 1, D))
    g_h = ttnn.slice(qg_h, (0, 0, D), (H, 1, 2 * D))
    g1 = ttnn.to_torch(ttnn.experimental.view(g_h, ttnn.Shape([1, 1, H * D]), ttnn.Shape([1, ttnn.TILE_SIZE, H * D]))).float()
    q1 = ttnn.rms_norm(q_h, weight=w, epsilon=1e-6)
    q1 = ttnn.to_torch(ttnn.reshape(ttnn.permute(q1, (1, 0, 2)), [1, 1, H, D])).float()
    for name, a, b in (("q", q0, q1), ("gate", g0, g1)):
        exact = (a == b).float().mean().item()
        print(f"qgview {name}: shape {tuple(b.shape)} bit-exact {100 * exact:.2f}%  max|d| {(a - b).abs().max().item():.3e}")
        assert exact == 1.0
