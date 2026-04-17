# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Residual block PCC test — isolate ttnn.add precision."""

from __future__ import annotations

import pytest
import torch
import ttnn

from models.experimental.droid_slam.reference.droid_net_ref import ResidualBlock
from models.experimental.droid_slam.tt.droid_encoder_tt import _TtResidualBlock


def _pcc(a, b):
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp(min=1e-12)
    return float((a @ b) / denom)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("stride", [1, 2])
def test_resblock_cnet_style(device, stride):
    torch.manual_seed(17)
    in_c, out_c = (32, 32) if stride == 1 else (32, 64)
    ref = ResidualBlock(in_c, out_c, norm_fn="none", stride=stride).eval()
    x = torch.randn(1, in_c, 120, 160)
    with torch.no_grad():
        y_ref = ref(x)

    tt_block = _TtResidualBlock(ref, norm_fn="none", device=device)
    x_nhwc = x.permute(0, 2, 3, 1).reshape(1, 1, 120 * 160, in_c)
    x_tt = ttnn.from_torch(x_nhwc, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    y_tt, h, w = tt_block(x_tt, device, 1, 120, 160)
    y = ttnn.to_torch(y_tt).float().reshape(1, h, w, out_c).permute(0, 3, 1, 2)
    pcc = _pcc(y_ref, y)
    print(f"stride={stride} PCC={pcc:.5f}")
    assert pcc > 0.99, f"PCC {pcc:.5f}"
