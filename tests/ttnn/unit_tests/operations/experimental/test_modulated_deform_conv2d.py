# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ttnn.experimental.modulated_deform_conv2d against a float64 host reference: bilinear columns
times the weight, with torchvision's deform_conv2d pinning the reference's (kk, c) column order."""

import pytest
import torch
import torch.nn.functional as F
import torchvision
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

# Relative error along the sign of the reference, which PCC cannot see: a uniform scale error. The
# op's HiFi2 products and its bfloat16 sum over kernel positions shrink outputs by about 0.55%; a
# coefficient truncated instead of rounded to bfloat16 adds about 0.2% more.
MAX_RELATIVE_BIAS = 1e-2


def out_size(size, kernel, stride, padding, dilation):
    return (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def reference_columns(x, offset, mask, kernel_size, stride, padding, dilation):
    """``x`` (B, H, W, C), ``offset`` (B, Ho, Wo, 2*K*K) as (y, x) pixels, ``mask`` (B, Ho, Wo, K*K)
    -> (B*Ho*Wo, K*K*C), bilinear with zero padding outside the input."""
    batch, in_h, in_w, _ = x.shape
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


def random_inputs(batch, in_h, in_w, channels, c_out, kernel_size, stride, padding, dilation, offset_scale):
    """Inputs rounded to bfloat16 first, so the reference sees the device's values: the input, the
    offsets, the mask logits, the (K*K*C, C_out) weight and the bias."""
    torch.manual_seed(0)
    k_h, k_w = kernel_size
    out_h = out_size(in_h, k_h, stride[0], padding[0], dilation[0])
    out_w = out_size(in_w, k_w, stride[1], padding[1], dilation[1])
    kk = k_h * k_w
    x = torch.randn(batch, in_h, in_w, channels).bfloat16().double()
    offset = (offset_scale * torch.randn(batch, out_h, out_w, 2 * kk)).bfloat16().double()
    logits = (2.0 * torch.randn(batch, out_h, out_w, kk)).bfloat16().double()
    weight = (torch.randn(kk * channels, c_out) / (kk * channels) ** 0.5).bfloat16().double()
    bias = torch.randn(1, c_out).bfloat16().double()
    return x, offset, logits, weight, bias


def run_conv(device, x, offset, logits, weight, bias, kernel_size, stride, padding, dilation, relu):
    """The op's output as a (B*Ho*Wo, C_out) float64 tensor; ``bias`` may be None."""
    row_major = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tiled = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.experimental.modulated_deform_conv2d(
        row_major(x),
        row_major(torch.cat([offset, logits], dim=-1)),
        tiled(weight),
        None if bias is None else tiled(bias),
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        relu=relu,
    )
    rows = offset.shape[0] * offset.shape[1] * offset.shape[2]
    assert tuple(out.shape) == (1, 1, rows, weight.shape[1])
    return ttnn.to_torch(out).double().reshape(rows, -1)


def assert_close(expected, actual):
    passed, pcc = comp_pcc(expected, actual, 0.999)
    assert torch.isfinite(actual).all()
    assert passed, f"PCC {pcc}"
    bias = ((actual - expected) * expected.sign()).mean() / expected.abs().mean()
    logger.info(f"PCC {pcc}, relative bias {bias:.2e}")
    assert abs(bias) < MAX_RELATIVE_BIAS, f"relative bias {bias:.2e}"


@pytest.mark.parametrize(
    "batch, in_h, in_w, channels, c_out, has_bias, relu, offset_scale, kernel_size, stride, padding, dilation",
    [
        (1, 8, 8, 32, 32, False, False, 2.0, (3, 3), (1, 1), (1, 1), (1, 1)),  # corners pushed outside
        (2, 5, 7, 64, 96, True, False, 1.0, (3, 3), (1, 1), (1, 1), (1, 1)),  # a partial last block
        (1, 24, 24, 32, 64, True, True, 20.0, (3, 3), (1, 1), (1, 1), (1, 1)),  # many corners far outside
        (1, 20, 40, 256, 256, True, True, 4.0, (3, 3), (1, 1), (1, 1), (1, 1)),
        # 22 blocks: two full rows of cores, a rectangle, so the weight multicast runs on a small grid.
        (1, 16, 44, 64, 64, True, True, 1.0, (3, 3), (1, 1), (1, 1), (1, 1)),
        # C_out over one 8-tile dest pass: several, and a partial last one.
        (1, 16, 20, 64, 320, True, True, 2.0, (3, 3), (1, 1), (1, 1), (1, 1)),
        (1, 16, 20, 64, 512, True, False, 2.0, (3, 3), (1, 1), (1, 1), (1, 1)),
        (1, 19, 21, 64, 64, True, False, 1.0, (3, 3), (2, 2), (1, 1), (1, 1)),
        (1, 16, 16, 32, 64, False, True, 1.0, (3, 3), (1, 1), (2, 2), (2, 2)),
        (1, 6, 10, 96, 64, True, False, 1.0, (1, 3), (1, 1), (0, 1), (1, 1)),
        # Inputs over one 8-tile chunk of channels: several passes per kernel position.
        (1, 12, 20, 512, 64, True, True, 4.0, (3, 3), (1, 1), (1, 1), (1, 1)),
        (1, 10, 14, 384, 96, True, False, 2.0, (3, 3), (1, 1), (1, 1), (1, 1)),
        (1, 10, 14, 320, 64, False, True, 2.0, (3, 3), (1, 1), (1, 1), (1, 1)),
        (6, 58, 100, 256, 256, True, True, 4.0, (3, 3), (1, 1), (1, 1), (1, 1)),  # BEVFormer-base layer3
        (6, 29, 50, 512, 512, True, True, 4.0, (3, 3), (1, 1), (1, 1), (1, 1)),  # BEVFormer-base layer4
    ],
)
def test_modulated_deform_conv2d(
    device, batch, in_h, in_w, channels, c_out, has_bias, relu, offset_scale, kernel_size, stride, padding, dilation
):
    x, offset, logits, weight, bias = random_inputs(
        batch, in_h, in_w, channels, c_out, kernel_size, stride, padding, dilation, offset_scale
    )
    bias = bias if has_bias else None
    actual = run_conv(device, x, offset, logits, weight, bias, kernel_size, stride, padding, dilation, relu)
    expected = reference_columns(x, offset, logits.sigmoid(), kernel_size, stride, padding, dilation) @ weight
    if bias is not None:
        expected = expected + bias
    if relu:
        expected = expected.clamp(min=0)
    assert_close(expected, actual)


def test_modulated_deform_conv2d_matches_torchvision(device):
    """The op is torchvision's modulated deformable conv, with the weight laid out (kk, c) x C_out."""
    kernel_size, stride, padding, dilation = (3, 3), (1, 1), (1, 1), (1, 1)
    x, offset, logits, _, bias = random_inputs(2, 16, 20, 64, 64, kernel_size, stride, padding, dilation, 3.0)
    torch_weight = (torch.randn(64, 64, 3, 3) / 24).bfloat16().double()
    expected = torchvision.ops.deform_conv2d(
        x.permute(0, 3, 1, 2),
        offset.permute(0, 3, 1, 2),
        torch_weight,
        bias[0],
        stride=stride,
        padding=padding,
        dilation=dilation,
        mask=logits.sigmoid().permute(0, 3, 1, 2),
    )
    weight = torch_weight.permute(2, 3, 1, 0).reshape(-1, torch_weight.shape[0])
    actual = run_conv(device, x, offset, logits, weight, bias, kernel_size, stride, padding, dilation, False)
    assert_close(expected.permute(0, 2, 3, 1).reshape(actual.shape), actual)


def test_modulated_deform_conv2d_l1_too_small(device, expect_error):
    """Each core keeps all its blocks' outputs and two weight slices; past L1 the op says so."""
    x, offset, logits, weight, bias = random_inputs(6, 58, 100, 256, 1024, (3, 3), (1, 1), (1, 1), (1, 1), 1.0)
    with expect_error(RuntimeError, "needs .* KB of L1 per core"):
        run_conv(device, x, offset, logits, weight, bias, (3, 3), (1, 1), (1, 1), (1, 1), False)


def test_modulated_deform_conv2d_kernel_too_large(device, expect_error):
    x, offset, logits, weight, bias = random_inputs(1, 16, 16, 32, 32, (7, 7), (1, 1), (3, 3), (1, 1), 1.0)
    with expect_error(RuntimeError, "more than 32 positions"):
        run_conv(device, x, offset, logits, weight, None, (7, 7), (1, 1), (3, 3), (1, 1), False)
