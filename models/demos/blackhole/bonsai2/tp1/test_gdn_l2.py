# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_1C_GDN_L2: q/k l2-normalization and the query scale inside the recurrence op (fused_l2norm), against
today's host path (rms_norm q, rms_norm k * K^-0.5, the composite's q * scale), both against float64.

Probes isolate the normalization (o and state are otherwise dominated by the op's internal bf16 rounding of q):
  q: state = identity per head, beta = 0, decay = 1  ->  o = q_n = q * scale / ||q||
  k: state = 0, beta = 1, decay = 1, v = ones        ->  every column of the new state = k_n = k / ||k||
The fused values must be at least as close to float64 as today's.

    pytest models/demos/blackhole/bonsai2/tp1/test_gdn_l2.py -s
"""
import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tt.gdn.tp import fused_recurrent_trim

NK, NV, D, GQA = 16, 48, 128, 3


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_gdn_l2(mesh_device):
    torch.manual_seed(0)
    f32 = ttnn.float32

    def dev(t, dt=f32):
        return ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh_device)

    # head-major q/k/v as FCONV hands them over: [heads, 1, D] fp32 (head_major_input)
    q_t, k_t = torch.randn(NK, 1, D) * 3, torch.randn(NK, 1, D) * 3
    q, k = dev(q_t), dev(k_t)
    scale = D**-0.5
    ones_v = dev(torch.ones(NV, 1, D))
    zero_v = dev(torch.zeros(NV, 1, D))
    g0 = dev(torch.zeros(1, 1, NV))  # decay = exp(0) = 1
    b0, b1 = dev(torch.zeros(1, 1, NV)), dev(torch.ones(1, 1, NV))
    eye = torch.eye(D).expand(NV, D, D).contiguous()
    qn64 = (q_t.double().reshape(NK, D) * scale / q_t.double().reshape(NK, D).norm(dim=-1, keepdim=True))
    kn64 = k_t.double().reshape(NK, D) / k_t.double().reshape(NK, D).norm(dim=-1, keepdim=True)
    hq = torch.arange(NV) // GQA

    res = {}
    for name, kw in (("today", {}), ("fused_l2norm", {"fused_l2norm": True})):
        o, _ = fused_recurrent_trim(q, k, zero_v, b0, g0, scale, initial_state=dev(eye), head_major_input=True, **kw)
        q_eff = ttnn.to_torch(o).double().reshape(NV, D)
        _, st = fused_recurrent_trim(q, k, ones_v, b1, g0, scale, initial_state=dev(torch.zeros(NV, D, D)),
                                     head_major_input=True, **kw)
        k_eff = ttnn.to_torch(st).double().reshape(NV, D, D)[:, :, 0]
        eq = ((q_eff - qn64[hq]).abs().max() / qn64.abs().max()).item()
        ek = ((k_eff - kn64[hq]).abs().max() / kn64.abs().max()).item()
        res[name] = (eq, ek)
        print(f"{name:<13}: q_n max rel err vs float64 {eq:.3e}   k_n {ek:.3e}")
    (eq0, ek0), (eq1, ek1) = res["today"], res["fused_l2norm"]
    assert eq1 <= eq0 * 1.05 and ek1 <= ek0 * 1.05, res
