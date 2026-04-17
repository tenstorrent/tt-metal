# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the ttnn layer wrappers (T0)."""

from __future__ import annotations

import pytest
import torch
import ttnn

from models.experimental.droid_slam.tt.ttnn_layers import (
    RELU,
    TtConv2d,
    TtInstanceNorm2d,
    from_tile_nhwc,
    to_tile_nhwc,
)


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp(min=1e-12)
    return float((a @ b) / denom)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch, in_c, out_c, h, w, kernel, stride, padding",
    [
        (1, 3, 32, 240, 320, (7, 7), (2, 2), (3, 3)),  # fnet/cnet conv1 (batch=1)
        (2, 3, 32, 240, 320, (7, 7), (2, 2), (3, 3)),  # batch=2
        (1, 32, 32, 120, 160, (3, 3), (1, 1), (1, 1)),  # residual block conv (layer1)
        (1, 64, 128, 60, 80, (3, 3), (2, 2), (1, 1)),  # layer3 stride-2
        (4, 128, 128, 30, 40, (3, 3), (1, 1), (1, 1)),  # UpdateModule inner conv (batch=4)
        (4, 196, 128, 30, 40, (1, 1), (1, 1), (0, 0)),  # corr_encoder first conv
    ],
)
def test_ttconv2d_matches_torch(device, batch, in_c, out_c, h, w, kernel, stride, padding):
    torch.manual_seed(42)
    ref = torch.nn.Conv2d(
        in_c,
        out_c,
        kernel_size=kernel,
        stride=stride,
        padding=padding,
        bias=True,
    ).eval()
    x = torch.randn(batch, in_c, h, w)

    with torch.no_grad():
        ref_out = ref(x)

    tt_conv = TtConv2d(ref, activation=None)
    x_tt = to_tile_nhwc(x, device)
    y_tt, out_h, out_w = tt_conv(x_tt, device, batch, h, w)
    assert out_h == ref_out.shape[-2]
    assert out_w == ref_out.shape[-1]
    y = from_tile_nhwc(y_tt, batch, out_h, out_w, out_c)
    pcc = _pcc(ref_out, y)
    assert pcc > 0.999, f"conv PCC {pcc:.5f} < 0.999"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_ttconv2d_with_relu(device):
    torch.manual_seed(7)
    ref = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1).eval()
    x = torch.randn(1, 32, 120, 160)
    with torch.no_grad():
        ref_out = torch.relu(ref(x))

    tt_conv = TtConv2d(ref, activation=RELU)
    x_tt = to_tile_nhwc(x, device)
    y_tt, h, w = tt_conv(x_tt, device, 1, 120, 160)
    y = from_tile_nhwc(y_tt, 1, h, w, 32)
    pcc = _pcc(ref_out, y)
    assert pcc > 0.999, f"conv+relu PCC {pcc:.5f} < 0.999"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_ttinstance_norm_matches_torch(device):
    channels = 32
    torch.manual_seed(3)
    x = torch.randn(1, channels, 120, 160)
    ref = torch.nn.InstanceNorm2d(channels, affine=False).eval()
    with torch.no_grad():
        ref_out = ref(x)

    tt_norm = TtInstanceNorm2d(channels, device)
    # Pack torch NCHW into [1, 1, N*H*W, C] tile, matching what ttnn.conv2d emits.
    x_packed = x.permute(0, 2, 3, 1).reshape(1, 1, 120 * 160, channels)
    x_tt = ttnn.from_torch(
        x_packed, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    y_tt = tt_norm(x_tt, batch_size=1, spatial=120 * 160)
    y = ttnn.to_torch(y_tt).float()
    y_nchw = y.reshape(1, 120, 160, channels).permute(0, 3, 1, 2).contiguous()
    pcc = _pcc(ref_out, y_nchw)
    assert pcc > 0.99, f"instance_norm PCC {pcc:.5f} < 0.99"
