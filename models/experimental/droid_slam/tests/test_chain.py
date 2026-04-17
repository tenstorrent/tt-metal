# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Chain-conv accuracy drill-down."""

from __future__ import annotations

import pytest
import torch
import ttnn

from models.experimental.droid_slam.tt.ttnn_layers import RELU, TtConv2d


def _pcc(a, b):
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp(min=1e-12)
    return float((a @ b) / denom)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("num_convs", [1, 2, 4, 8, 16])
def test_chain_conv_pcc(device, num_convs):
    torch.manual_seed(99)
    c = 32
    convs = [torch.nn.Conv2d(c, c, kernel_size=3, padding=1).eval() for _ in range(num_convs)]
    x = torch.randn(1, c, 120, 160)

    with torch.no_grad():
        y_ref = x
        for cv in convs:
            y_ref = torch.relu(cv(y_ref))

    tt_convs = [TtConv2d(cv, activation=RELU) for cv in convs]
    x_nhwc = x.permute(0, 2, 3, 1).reshape(1, 1, 120 * 160, c)
    x_tt = ttnn.from_torch(x_nhwc, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    h, w = 120, 160
    for tc in tt_convs:
        x_tt, h, w = tc(x_tt, device, 1, h, w)

    y = ttnn.to_torch(x_tt).float().reshape(1, h, w, c).permute(0, 3, 1, 2)
    pcc = _pcc(y_ref, y)
    print(f"num_convs={num_convs} PCC={pcc:.5f}")
    assert pcc > 0.999 if num_convs == 1 else pcc > 0.99, (
        f"num_convs={num_convs} PCC={pcc:.5f} below target"
    )
