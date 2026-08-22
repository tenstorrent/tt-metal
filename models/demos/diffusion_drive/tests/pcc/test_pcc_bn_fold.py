# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the Conv+BN fold helper (no device required).

Validates fold_bn() against running the unfused Conv+BN pair in eval mode.
Tests cover:
  - Standard 3×3 conv (bias=False, typical ResNet block)
  - 1×1 stride-2 conv (downsampling shortcut)
  - 7×7 stem conv (bias=False)
  - Conv with explicit bias
"""

import pytest
import torch
import torch.nn as nn

from models.demos.diffusion_drive.tt.common import fold_bn


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    if denom < 1e-12:
        return 1.0
    return (a @ b).item() / denom


def _run_case(conv: nn.Conv2d, bn: nn.BatchNorm2d, x: torch.Tensor) -> None:
    """Test fold_bn formula correctness.

    Two checks:
    1. fp32 fold (no quantisation): max abs error < 1e-3 (fold math correct)
    2. bfloat16 fold (actual output of fold_bn): PCC >= 0.99
       (bfloat16 cast introduces ~0.01 abs error — that is expected and fine)
    """
    conv.eval()
    bn.eval()

    # Reference: unfused fp32
    with torch.no_grad():
        ref = bn(conv(x))

    # --- Check 1: formula correctness in fp32 (no bfloat16 cast) -------------
    # Replicate fold_bn logic but stay in fp32
    w_fp32 = conv.weight.float()
    b_fp32 = conv.bias.float() if conv.bias is not None else torch.zeros(conv.out_channels)
    g = bn.weight.float()
    mu = bn.running_mean.float()
    var = bn.running_var.float()
    eps = bn.eps
    beta = bn.bias.float()
    scale = g / torch.sqrt(var + eps)
    w_fold_fp32 = w_fp32 * scale.reshape(-1, 1, 1, 1)
    b_fold_fp32 = (b_fp32 - mu) * scale + beta

    conv_fold_fp32 = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
    )
    conv_fold_fp32.weight.data = w_fold_fp32
    conv_fold_fp32.bias.data = b_fold_fp32
    conv_fold_fp32.eval()

    with torch.no_grad():
        out_fp32 = conv_fold_fp32(x)

    max_err_fp32 = (out_fp32 - ref).abs().max().item()
    assert max_err_fp32 < 1e-3, f"fp32 fold max abs error {max_err_fp32:.2e} >= 1e-3 — formula bug"

    # --- Check 2: bfloat16 output of fold_bn — PCC >= 0.99 -------------------
    w_fold, b_fold = fold_bn(conv, bn)

    conv_fold_bf16 = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
    )
    conv_fold_bf16.weight.data = w_fold.float()  # bfloat16 → float for conv
    conv_fold_bf16.bias.data = b_fold.float()
    conv_fold_bf16.eval()

    with torch.no_grad():
        out_bf16 = conv_fold_bf16(x)

    pcc_val = _pcc(out_bf16, ref)
    assert pcc_val >= 0.99, f"bfloat16 fold PCC {pcc_val:.6f} < 0.99"


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch", [1, 2])
def test_fold_bn_3x3_no_bias(batch: int) -> None:
    """Standard ResNet BasicBlock 3×3 conv (bias=False)."""
    torch.manual_seed(0)
    conv = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
    bn = nn.BatchNorm2d(64)
    # Seed running stats so they are non-trivial
    bn.running_mean.data = torch.randn(64) * 0.5
    bn.running_var.data = torch.rand(64) + 0.1
    x = torch.randn(batch, 64, 32, 32)
    _run_case(conv, bn, x)


def test_fold_bn_1x1_downsample() -> None:
    """1×1 stride-2 downsampling shortcut in ResNet BasicBlock."""
    torch.manual_seed(1)
    conv = nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False)
    bn = nn.BatchNorm2d(128)
    bn.running_mean.data = torch.randn(128)
    bn.running_var.data = torch.rand(128) + 0.1
    x = torch.randn(1, 64, 32, 32)
    _run_case(conv, bn, x)


def test_fold_bn_7x7_stem() -> None:
    """ResNet-34 stem: 7×7 conv, stride=2, bias=False."""
    torch.manual_seed(2)
    conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    bn = nn.BatchNorm2d(64)
    bn.running_mean.data = torch.randn(64) * 0.3
    bn.running_var.data = torch.rand(64) + 0.2
    x = torch.randn(1, 3, 256, 1024)
    _run_case(conv, bn, x)


def test_fold_bn_with_explicit_bias() -> None:
    """Conv2d with explicit bias (less common, but should work)."""
    torch.manual_seed(3)
    conv = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True)
    bn = nn.BatchNorm2d(32)
    bn.running_mean.data = torch.randn(32)
    bn.running_var.data = torch.rand(32) + 0.1
    x = torch.randn(2, 32, 16, 16)
    _run_case(conv, bn, x)


def test_fold_bn_precision_vs_fp16() -> None:
    """Demonstrate that folding in fp32 is correct; PCC >= 0.99 after bfloat16 cast."""
    torch.manual_seed(4)
    conv = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
    bn = nn.BatchNorm2d(512)
    bn.running_mean.data = torch.randn(512) * 2
    bn.running_var.data = torch.rand(512) * 5 + 0.1
    x = torch.randn(1, 512, 8, 8)
    conv.eval()
    bn.eval()
    with torch.no_grad():
        ref = bn(conv(x))
    w_fold, b_fold = fold_bn(conv, bn)
    # Cast is at the end, fold is in fp32 — should satisfy PCC >= 0.99
    conv_fold = nn.Conv2d(512, 512, 3, padding=1, bias=True)
    conv_fold.weight.data = w_fold.float()
    conv_fold.bias.data = b_fold.float()
    conv_fold.eval()
    with torch.no_grad():
        out = conv_fold(x)
    pcc_val = _pcc(out, ref)
    assert pcc_val >= 0.99, f"PCC {pcc_val:.6f} < 0.99"
