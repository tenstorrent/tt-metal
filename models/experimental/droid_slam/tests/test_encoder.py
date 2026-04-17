# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TtBasicEncoder (T1)."""

from __future__ import annotations

import pytest
import torch
import ttnn

from models.experimental.droid_slam.reference.droid_net_ref import BasicEncoder
from models.experimental.droid_slam.tt.droid_encoder_tt import TtBasicEncoder


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp(min=1e-12)
    return float((a @ b) / denom)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("norm_fn, output_dim", [("none", 256), ("instance", 128)])
def test_ttbasic_encoder(device, norm_fn, output_dim):
    torch.manual_seed(123)
    ref = BasicEncoder(output_dim=output_dim, norm_fn=norm_fn).eval()
    batch, frames = 1, 2
    h, w = 240, 320
    x = torch.randn(batch, frames, 3, h, w)

    with torch.no_grad():
        ref_out = ref(x)  # (B, N, C_out, H/8, W/8)

    tt = TtBasicEncoder(ref, device)

    # Flatten batch + frames and move to NHWC tile.
    x_bn = x.view(batch * frames, 3, h, w)
    x_nhwc = x_bn.permute(0, 2, 3, 1).reshape(1, 1, batch * frames * h * w, 3)
    x_tt = ttnn.from_torch(
        x_nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )

    y_tt, h_out, w_out = tt(x_tt, batch_size=batch * frames, h=h, w=w)
    y = ttnn.to_torch(y_tt).float()
    y_nchw = (
        y.reshape(batch * frames, h_out, w_out, tt.out_channels)
        .permute(0, 3, 1, 2)
        .contiguous()
        .view(batch, frames, tt.out_channels, h_out, w_out)
    )

    pcc = _pcc(ref_out, y_nchw)
    assert pcc > 0.99, f"encoder(norm={norm_fn}) PCC {pcc:.5f} < 0.99"
