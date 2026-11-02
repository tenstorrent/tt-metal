# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_1C_GDN_VSPLIT: the recurrence op with each head on 2 cores (one per half of V) against 1 core per head.
Every step separates by V column, so o and the updated state must be BIT-EXACT. Run in the model's form: ring
mode (state updated in place), GB (fused gating) and L2 (fused l2norm), head-major q/k/v.

The factory reads the env var when it builds the program, and the program cache key does not include it, so the
cache is cleared between the two arms.

    pytest models/demos/blackhole/bonsai2/tp1/test_gdn_vsplit.py -s
"""
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tt.gdn.tp import fused_recurrent_trim

NK, NV, D = 16, 48, 128


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_gdn_vsplit(mesh_device):
    torch.manual_seed(0)
    f32 = ttnn.float32

    def dev(t, dt=f32):
        return ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh_device)

    q_t, k_t, v_t = torch.randn(NK, 1, D) * 2, torch.randn(NK, 1, D) * 2, torch.randn(NV, 1, D)
    ab_t = torch.randn(1, 1, 2 * NV) * 2
    consts_t = torch.cat([-torch.rand(NV) * 2, torch.randn(NV) * 0.5]).reshape(1, 1, 2 * NV)
    s_t = torch.randn(NV, D, D) * 0.1
    idx = ttnn.from_torch(torch.arange(NV, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                          device=mesh_device)
    out = {}
    for arm in ("0", "1"):
        os.environ["QWEN36_1C_GDN_VSPLIT"] = arm
        mesh_device.clear_program_cache()
        ring = dev(s_t)
        o, st = fused_recurrent_trim(
            dev(q_t), dev(k_t), dev(v_t), dev(consts_t), dev(ab_t), D**-0.5,
            initial_state=ring, initial_state_block_idx=idx, output_per_token_state=True,
            head_major_input=True, head_major_output=True, fused_gating=True, fused_l2norm=True,
        )
        out[arm] = (ttnn.to_torch(o).float(), ttnn.to_torch(st).float())
    os.environ.pop("QWEN36_1C_GDN_VSPLIT")
    for name, i in (("o", 0), ("state", 1)):
        a, b = out["0"][i], out["1"][i]
        assert a.shape == b.shape, (name, a.shape, b.shape)
        exact = (a == b).float().mean().item()
        print(f"{name}: shape {tuple(b.shape)} bit-exact {100 * exact:.2f}%  max|d| {(a - b).abs().max().item():.3e}"
              f"  finite {bool(torch.isfinite(b).all())}")
        assert exact == 1.0
