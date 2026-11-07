# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
# gdn_prefill.forward.GateTiles against torch fp32 and against the ttnn chain it replaces.
import torch

import ttnn
from models.demos.blackhole.bonsai2.tt.gdn_prefill.forward import GateTiles, _to_bht


def test_gdn_gates(device):
    torch.manual_seed(0)
    T, H = 2048, 48
    ab = (torch.randn(1, T, 2 * H) * 3).bfloat16()
    dt = torch.randn(H) * 0.5
    an = -torch.rand(H) * 4
    abt = ttnn.from_torch(ab, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    beta_bh, g_bh = GateTiles(device, dt, an)(abt, T)
    a, b = ab.float()[0, :, :H], ab.float()[0, :, H:]
    beta_ref = torch.sigmoid(b).T.reshape(H, T, 1)
    g_ref = (an * torch.nn.functional.softplus(a + dt)).T
    beta = ttnn.to_torch(beta_bh).float()
    g = ttnn.to_torch(g_bh).float()
    rb = ((beta - beta_ref).norm() / beta_ref.norm()).item()
    rg = ((g - g_ref).norm() / g_ref.norm()).item()
    # the ttnn chain (forward.py's non-fused path, bf16 at each step)
    a_t = ttnn.from_torch(ab[:, :, :H], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    b_t = ttnn.from_torch(ab[:, :, H:], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    dt_t = ttnn.from_torch(dt.reshape(1, 1, H).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    an_t = ttnn.from_torch(an.reshape(1, 1, H).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    g_old = ttnn.to_torch(_to_bht(ttnn.multiply(an_t, ttnn.softplus(ttnn.add(a_t, dt_t))), 1, T, H)).float()
    b_old = ttnn.to_torch(ttnn.reshape(_to_bht(ttnn.sigmoid(b_t), 1, T, H), [H, T, 1])).float()
    ro_g = ((g_old - g_ref).norm() / g_ref.norm()).item()
    ro_b = ((b_old - beta_ref).norm() / beta_ref.norm()).item()
    print(f"GATES fused: beta rel {rb:.6f} g rel {rg:.6f} | ttnn chain: beta rel {ro_b:.6f} g rel {ro_g:.6f}")
    assert rb < 0.005 and rg < 0.005
