# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host-only parity check of `TtUniAD.velo_update` against the reference `UniAD.velo_update`.

Both move the reference points of active tracks by their velocity and the ego motion between two frames; the device
version takes ttnn tensors and does the few rows of arithmetic on the host.
"""

from types import SimpleNamespace

import torch
import ttnn

from models.experimental.uniad.reference.uniad import UniAD
from models.experimental.uniad.tt.ttnn_uniad import TtUniAD

PC_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]


def rot_z(yaw):
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_velo_update_matches_reference():
    torch.manual_seed(0)
    ref_pts = torch.randn(7, 3)
    velocity = torch.randn(7, 2)
    l2g_r1 = rot_z(torch.tensor(0.3))[None]
    l2g_r2 = rot_z(torch.tensor(0.35))[None]
    l2g_t1 = torch.tensor([[600.0, 1640.0, 1.8]])
    l2g_t2 = l2g_t1 + torch.tensor([[4.0, 1.0, 0.0]])
    time_delta = torch.tensor([0.5])
    args = (ref_pts, velocity, l2g_r1, l2g_t1, l2g_r2, l2g_t2, time_delta)

    expected = UniAD.velo_update(SimpleNamespace(pc_range=PC_RANGE), *args)
    actual = TtUniAD.velo_update(SimpleNamespace(pc_range=PC_RANGE), *(ttnn.from_torch(a) for a in args))

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=1e-4)
