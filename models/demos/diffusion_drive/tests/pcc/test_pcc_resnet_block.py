# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Per-BasicBlock PCC tests for the DiffusionDrive ResNet-34 backbone.

Tests individual BasicBlock implementations (TTNN conv2d + BN-folded weights)
against the PyTorch reference. Two block types are covered:
  - Identity-shortcut block (no downsampling): test_basic_block_identity_shortcut
  - Downsampling block (1×1 stride-2 + BN): test_basic_block_downsample

These tests validate the BN-fold + TTNN conv2d pipeline in isolation (gap 18)
without needing the full model loaded.

Input tensors: random with fixed seed; no pretrained weights required.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import ttnn
from models.demos.diffusion_drive.tt.ttnn_resnet34 import TtnnBasicBlock, prepare_basic_block_params


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    if denom < 1e-12:
        return 1.0
    return (a @ b).item() / denom


def _make_basic_block(in_ch: int, out_ch: int, stride: int) -> nn.Module:
    """Construct a timm-style BasicBlock (conv1+bn1+relu + conv2+bn2 + optional ds)."""
    import timm

    m = timm.create_model("resnet34", pretrained=False, features_only=True)
    # Find any block that has the right (in_ch, out_ch, stride)
    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        layer = getattr(m, layer_name)
        for block in layer:
            c1 = block.conv1.in_channels
            c2 = block.conv2.out_channels
            s = block.stride
            if c1 == in_ch and c2 == out_ch and s == stride:
                return block
    raise RuntimeError(f"No BasicBlock found with in={in_ch} out={out_ch} stride={stride}")


def _torch_run_block(block: nn.Module, x: torch.Tensor) -> torch.Tensor:
    block.eval()
    with torch.no_grad():
        return block(x)


def _ttnn_run_block(
    block_params: dict,
    x_torch: torch.Tensor,
    stride: int,
    device: ttnn.Device,
) -> torch.Tensor:
    """Run TtnnBasicBlock on the given input; return output as NCHW float32 tensor."""
    B, C_in, H, W = x_torch.shape

    # Convert to TTNN format: NCHW → NHWC → (1, 1, B*H*W, C)
    x_nhwc = x_torch.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
    x_flat = x_nhwc.reshape(1, 1, B * H * W, C_in).to(torch.bfloat16)
    x_ttnn = ttnn.from_torch(x_flat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    ttnn_block = TtnnBasicBlock(block_params, stride=stride, device=device)
    out_ttnn, (B_out, H_out, W_out, C_out) = ttnn_block(x_ttnn, (B, H, W, C_in))

    # Convert back: TTNN (1, 1, B*H*W, C_out) → NCHW float32
    out_torch = ttnn.to_torch(out_ttnn)  # (1, 1, B*H_out*W_out, C_out)
    out_torch = out_torch.reshape(B_out, H_out, W_out, C_out)
    out_torch = out_torch.permute(0, 3, 1, 2).float()  # NCHW
    return out_torch


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch", [1, 2])
def test_basic_block_identity_shortcut(device, batch: int) -> None:
    """BasicBlock with identity shortcut (in_ch == out_ch, stride=1): PCC ≥ 0.99."""
    torch.manual_seed(42)

    # Layer1 block[0]: conv(64→64 s=1), no downsample
    block = _make_basic_block(in_ch=64, out_ch=64, stride=1)
    block.eval()

    # Prepare TTNN params (fold BN)
    params = prepare_basic_block_params(block.conv1, block.bn1, block.conv2, block.bn2, downsample=block.downsample)

    # Random input — deliberately small spatial size to keep test fast
    x = torch.randn(batch, 64, 16, 32)

    ref_out = _torch_run_block(block, x)
    ttnn_out = _ttnn_run_block(params, x, stride=1, device=device)

    pcc = _pcc(ttnn_out, ref_out)
    assert pcc >= 0.99, f"Identity-shortcut block PCC {pcc:.6f} < 0.99"


def test_basic_block_downsample(device) -> None:
    """BasicBlock with 1×1 stride-2 downsampling shortcut: PCC ≥ 0.99."""
    torch.manual_seed(0)

    # Layer2 block[0]: conv(64→128 s=2), with downsample
    block = _make_basic_block(in_ch=64, out_ch=128, stride=2)
    block.eval()

    params = prepare_basic_block_params(block.conv1, block.bn1, block.conv2, block.bn2, downsample=block.downsample)

    x = torch.randn(1, 64, 16, 32)

    ref_out = _torch_run_block(block, x)
    ttnn_out = _ttnn_run_block(params, x, stride=2, device=device)

    pcc = _pcc(ttnn_out, ref_out)
    assert pcc >= 0.99, f"Downsampling block PCC {pcc:.6f} < 0.99"
