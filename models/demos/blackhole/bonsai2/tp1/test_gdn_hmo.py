# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_1C_GDN_HMO equivalence: recurrence o head-major ([Nv,1,Dv], head_major_output=True) -> gated
rms_norm -> flat view, against today's o [1,1,Nv,Dv] -> reshape -> rms_norm -> reshape. Expect bit-exact out_f
and a zero-copy flatten (same buffer address).

    pytest models/demos/blackhole/bonsai2/tp1/test_gdn_hmo.py -s
"""
import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tt.gdn.tp import fused_recurrent_trim, head_major_flat_view

NK, NV, D = 16, 48, 128


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_gdn_hmo(mesh_device):
    torch.manual_seed(0)
    L1 = ttnn.L1_MEMORY_CONFIG

    def dev(t, dt=ttnn.bfloat16):
        return ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=L1)

    q, k, v = (dev(torch.randn(1, 1, n, D)) for n in (NK, NK, NV))
    beta, g = dev(torch.rand(1, 1, NV)), dev(-torch.rand(1, 1, NV) * 2, ttnn.float32)
    s_t = torch.randn(NV, D, D) * 0.1
    w = ttnn.from_torch(torch.rand(D) + 0.5, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device)

    o_old, _ = fused_recurrent_trim(q, k, v, beta, g, D**-0.5, initial_state=dev(s_t, ttnn.float32))
    n_old = ttnn.rms_norm(ttnn.reshape(o_old, (1, NV, D)), weight=w, epsilon=1e-6, memory_config=L1)
    f_old = ttnn.to_torch(ttnn.reshape(n_old, (1, 1, NV * D))).float()

    o_new, _ = fused_recurrent_trim(q, k, v, beta, g, D**-0.5, initial_state=dev(s_t, ttnn.float32),
                                    head_major_output=True)
    print("o head-major shape", tuple(o_new.shape), "padded", tuple(o_new.padded_shape))
    n_new = ttnn.rms_norm(o_new, weight=w, epsilon=1e-6, memory_config=L1)
    f_new_t = head_major_flat_view(n_new)
    print("view shape", tuple(f_new_t.shape), "same buffer", f_new_t.buffer_address() == n_new.buffer_address())
    f_new = ttnn.to_torch(f_new_t).float()
    assert f_new.shape == f_old.shape, (f_new.shape, f_old.shape)
    exact = (f_new == f_old).float().mean().item()
    print(f"out_f bit-exact {100 * exact:.2f}%  max|d| {(f_new - f_old).abs().max().item():.3e}")
    assert exact == 1.0
