# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_1C_GDN_HMI and QWEN36_1C_GDN_RECT equivalence, from the head-major conv output [1, 80, 128] bf16:
  today (CONV_HM + TRIM + HMO): three bf16 row slices -> [1,1,H,D] views -> the op typecasts and head-splits
  HMI: one fp32 typecast + one permute to [80,1,128] -> [H,1,D] slices -> head_major_input=True
Expect o and state bit-exact, with RECT on or off (RECT only changes how the same cores are dispatched).

    pytest models/demos/blackhole/bonsai2/tp1/test_gdn_hmi.py -s
"""
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tt.gdn.tp import fused_recurrent_trim

NK, NV, D = 16, 48, 128


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("rect", ["0", "1"])
def test_gdn_hmi(mesh_device, rect):
    os.environ["QWEN36_1C_GDN_RECT"] = rect
    torch.manual_seed(0)
    L1 = ttnn.L1_MEMORY_CONFIG

    def dev(t, dt=ttnn.bfloat16):
        return ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=L1)

    conv = dev(torch.randn(1, 2 * NK + NV, D))
    beta, g = dev(torch.rand(1, 1, NV)), dev(-torch.rand(1, 1, NV) * 2, ttnn.float32)
    s_t = torch.randn(NV, D, D) * 0.1

    q = ttnn.reshape(ttnn.slice(conv, (0, 0, 0), (1, NK, D)), (1, 1, NK, D))
    k = ttnn.reshape(ttnn.slice(conv, (0, NK, 0), (1, 2 * NK, D)), (1, 1, NK, D))
    v = ttnn.reshape(ttnn.slice(conv, (0, 2 * NK, 0), (1, 2 * NK + NV, D)), (1, 1, NV, D))
    o0, s0 = fused_recurrent_trim(q, k, v, beta, g, D**-0.5, initial_state=dev(s_t, ttnn.float32),
                                  head_major_output=True)

    ch = ttnn.permute(ttnn.typecast(conv, ttnn.float32), (1, 0, 2))
    q1, k1, v1 = (ttnn.slice(ch, (a, 0, 0), (b, 1, D)) for a, b in ((0, NK), (NK, 2 * NK), (2 * NK, 2 * NK + NV)))
    o1, s1 = fused_recurrent_trim(q1, k1, v1, beta, g, D**-0.5, initial_state=dev(s_t, ttnn.float32),
                                  head_major_output=True, head_major_input=True)
    for name, a, b in (("o", o0, o1), ("state", s0, s1)):
        a, b = ttnn.to_torch(a).float(), ttnn.to_torch(b).float()
        exact = (a == b).float().mean().item()
        print(f"rect={rect} {name}: shape {tuple(b.shape)} bit-exact {100 * exact:.2f}%  max|d| {(a - b).abs().max().item():.3e}")
        assert exact == 1.0
