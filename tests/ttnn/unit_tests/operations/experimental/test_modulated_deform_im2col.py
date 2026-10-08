# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ttnn.experimental.modulated_deform_im2col against a float64 host reference, and its columns
times the weight against torchvision's deform_conv2d, which pins the (kk, c) column order."""

import pytest
import torch
import torch.nn.functional as F
import torchvision

import ttnn
from models.common.utility_functions import comp_pcc


def reference_im2col(x, offset, mask, kernel_size, stride, padding, dilation):
    """``x`` (B, H, W, C), ``offset`` (B, Ho, Wo, 2*K*K) as (y, x) pixels, ``mask`` (B, Ho, Wo, K*K)
    -> (B*Ho*Wo, K*K*C), bilinear with zero padding outside the input."""
    batch, in_h, in_w, channels = x.shape
    _, out_h, out_w, _ = offset.shape
    k_h, k_w = kernel_size
    h_o = torch.arange(out_h, dtype=x.dtype).view(1, -1, 1)
    w_o = torch.arange(out_w, dtype=x.dtype).view(1, 1, -1)
    columns = []
    for kk in range(k_h * k_w):
        y = h_o * stride[0] - padding[0] + (kk // k_w) * dilation[0] + offset[..., 2 * kk]
        x_ = w_o * stride[1] - padding[1] + (kk % k_w) * dilation[1] + offset[..., 2 * kk + 1]
        # grid_sample with align_corners=False puts pixel p at (2p + 1) / size - 1.
        grid = torch.stack([(2 * x_ + 1) / in_w - 1, (2 * y + 1) / in_h - 1], dim=-1)
        sampled = F.grid_sample(x.permute(0, 3, 1, 2), grid, mode="bilinear", align_corners=False)
        columns.append(sampled.permute(0, 2, 3, 1) * mask[..., kk : kk + 1])
    return torch.cat(columns, dim=-1).reshape(batch * out_h * out_w, -1)


def run_im2col(device, batch, in_h, in_w, channels, kernel_size, stride, padding, dilation, offset_scale):
    torch.manual_seed(0)
    k_h, k_w = kernel_size
    out_h = (in_h + 2 * padding[0] - dilation[0] * (k_h - 1) - 1) // stride[0] + 1
    out_w = (in_w + 2 * padding[1] - dilation[1] * (k_w - 1) - 1) // stride[1] + 1
    # Rounded to bfloat16 first, so the reference sees the device's inputs.
    x = torch.randn(batch, in_h, in_w, channels).bfloat16().double()
    offset = (offset_scale * torch.randn(batch, out_h, out_w, 2 * k_h * k_w)).bfloat16().double()
    mask = torch.rand(batch, out_h, out_w, k_h * k_w).bfloat16().double()

    def to_device(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    tt_columns = ttnn.experimental.modulated_deform_im2col(
        to_device(x),
        to_device(offset),
        to_device(mask),
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    rows = batch * out_h * out_w
    assert tuple(tt_columns.shape) == (1, 1, rows, k_h * k_w * channels)
    columns = ttnn.to_torch(tt_columns).double().reshape(rows, -1)
    return x, offset, mask, columns, (out_h, out_w)


@pytest.mark.parametrize(
    "batch, in_h, in_w, channels, kernel_size, stride, padding, dilation, offset_scale",
    [
        (1, 8, 8, 32, (3, 3), (1, 1), (1, 1), (1, 1), 2.0),  # offsets push corners outside
        (2, 5, 7, 64, (3, 3), (1, 1), (1, 1), (1, 1), 1.0),  # 70 rows: a partial last block
        (1, 9, 11, 32, (3, 3), (2, 2), (1, 1), (1, 1), 1.0),
        (1, 12, 12, 32, (3, 3), (1, 1), (2, 2), (2, 2), 1.0),
        (1, 6, 10, 96, (1, 3), (1, 1), (0, 1), (1, 1), 1.0),
        (6, 58, 100, 256, (3, 3), (1, 1), (1, 1), (1, 1), 4.0),  # BEVFormer-base layer3
        (6, 29, 50, 512, (3, 3), (1, 1), (1, 1), (1, 1), 4.0),  # BEVFormer-base layer4
    ],
)
def test_modulated_deform_im2col(
    device, batch, in_h, in_w, channels, kernel_size, stride, padding, dilation, offset_scale
):
    x, offset, mask, columns, _ = run_im2col(
        device, batch, in_h, in_w, channels, kernel_size, stride, padding, dilation, offset_scale
    )
    expected = reference_im2col(x, offset, mask, kernel_size, stride, padding, dilation)
    passed, pcc = comp_pcc(expected, columns, 0.999)
    assert passed, f"PCC {pcc}"
    assert torch.isfinite(columns).all()


def test_modulated_deform_im2col_conv(device):
    """The columns times the reshaped weight are torchvision's modulated deformable conv."""
    kernel_size, stride, padding, dilation = (3, 3), (1, 1), (1, 1), (1, 1)
    x, offset, mask, columns, (out_h, out_w) = run_im2col(
        device, 2, 16, 20, 64, kernel_size, stride, padding, dilation, 3.0
    )
    weight = torch.randn(48, 64, 3, 3).double()
    expected = torchvision.ops.deform_conv2d(
        x.permute(0, 3, 1, 2),
        offset.permute(0, 3, 1, 2),
        weight,
        stride=stride,
        padding=padding,
        dilation=dilation,
        mask=mask.permute(0, 3, 1, 2),
    )
    # Columns are ordered (kk, c), so the weight goes (k_h, k_w, C_in) x C_out.
    conv = columns @ weight.permute(2, 3, 1, 0).reshape(-1, weight.shape[0])
    conv = conv.reshape(2, out_h, out_w, -1).permute(0, 3, 1, 2)
    passed, pcc = comp_pcc(expected, conv, 0.999)
    assert passed, f"PCC {pcc}"
