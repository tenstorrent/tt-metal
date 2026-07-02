# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PCC: ACE-Step v1.5 flow-matching solver step vs reference elementwise math.

Reference:
  xt_next = xt - vt * dt          (Euler step in generate_audio)
  x0_pred = zt - vt * t           (get_x0_from_noise)

Validates the solver's on-device elementwise ops that close the inference loop around the DiT.
"""

import pytest
import torch


from models.experimental.acestep.tt.flow_match import FlowMatchStep
from models.experimental.acestep.tests.test_utils import (
    assert_pcc,
    require_single_device,
    to_torch,
    to_ttnn_tensor,
)

OUT_CHANNELS = 64  # audio latent acoustic dim
SEQ_LENS = [128, 512]


@pytest.mark.parametrize("seq_len", SEQ_LENS, ids=[f"S{s}" for s in SEQ_LENS])
def test_flow_match_step_vs_ref(device, seq_len):
    require_single_device(device)
    torch.manual_seed(0)

    xt = torch.randn(1, 1, seq_len, OUT_CHANNELS, dtype=torch.float32)
    vt = torch.randn(1, 1, seq_len, OUT_CHANNELS, dtype=torch.float32)
    zt = torch.randn(1, 1, seq_len, OUT_CHANNELS, dtype=torch.float32)
    dt = 0.037  # representative step size (1/infer_steps range)
    t = 0.6  # representative timestep

    ref_euler = xt - vt * dt
    ref_x0 = zt - vt * t

    solver = FlowMatchStep(mesh_device=device)
    xt_tt = to_ttnn_tensor(xt, device)
    vt_tt = to_ttnn_tensor(vt, device)
    zt_tt = to_ttnn_tensor(zt, device)

    euler_tt = solver.euler_step(xt_tt, vt_tt, dt)
    x0_tt = solver.x0_from_noise(zt_tt, vt_tt, t)

    euler = to_torch(euler_tt, expected_shape=(1, 1, seq_len, OUT_CHANNELS))
    x0 = to_torch(x0_tt, expected_shape=(1, 1, seq_len, OUT_CHANNELS))

    assert_pcc(ref_euler, euler, 0.999)
    assert_pcc(ref_x0, x0, 0.999)
