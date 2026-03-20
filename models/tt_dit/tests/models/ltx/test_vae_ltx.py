# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Tests for LTX-2 Video VAE components.
"""

import sys

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.models.vae.ltx.vae_ltx import LTXCausalConv3d, LTXResnetBlock3D
from models.tt_dit.utils.check import assert_quality
from models.tt_dit.utils.conv3d import conv_pad_in_channels

sys.path.insert(0, "LTX-2/packages/ltx-core/src")


@pytest.mark.parametrize(
    "in_c, out_c, kernel_size, stride, T, H, W",
    [
        (128, 128, 3, 1, 3, 16, 16),  # Standard residual block conv
        (128, 256, 3, 1, 3, 16, 16),  # Channel expansion
        (48, 128, 3, 1, 5, 32, 32),  # conv_in (after patchify)
    ],
    ids=["res_128_128", "expand_128_256", "conv_in_48_128"],
)
@pytest.mark.parametrize(
    "mesh_device",
    [(1, 1)],
    ids=["1x1"],
    indirect=["mesh_device"],
)
def test_ltx_causal_conv3d(
    mesh_device: ttnn.MeshDevice, in_c: int, out_c: int, kernel_size: int, stride: int, T: int, H: int, W: int
):
    """
    Test LTXCausalConv3d against PyTorch CausalConv3d reference.
    """
    from ltx_core.model.video_vae.convolution import CausalConv3d as TorchCausalConv3d

    B = 1
    torch.manual_seed(42)

    # PyTorch reference
    torch_model = TorchCausalConv3d(
        in_channels=in_c,
        out_channels=out_c,
        kernel_size=kernel_size,
        stride=stride,
    )
    torch_model.eval()

    # TT model
    tt_model = LTXCausalConv3d(
        in_channels=in_c,
        out_channels=out_c,
        kernel_size=kernel_size,
        stride=stride,
        mesh_device=mesh_device,
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())

    # Input
    x = torch.randn(B, in_c, T, H, W, dtype=torch.float32)

    # PyTorch forward (BCTHW format)
    with torch.no_grad():
        torch_out = torch_model(x)  # (B, out_c, T_out, H_out, W_out)

    # TT forward (BTHWC format)
    x_bthwc = x.permute(0, 2, 3, 4, 1)  # (B, T, H, W, C)
    x_bthwc = conv_pad_in_channels(x_bthwc)  # Pad C to alignment
    x_tt = ttnn.from_torch(x_bthwc, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)

    tt_out = tt_model(x_tt)
    tt_out_torch = ttnn.to_torch(tt_out)  # (B, T_out, H_out, W_out, C_out)
    tt_out_torch = tt_out_torch[:, :, :, :, :out_c]  # Trim padded channels
    tt_out_torch = tt_out_torch.permute(0, 4, 1, 2, 3)  # Back to BCTHW

    logger.info(f"PyTorch out: {torch_out.shape}, TT out: {tt_out_torch.shape}")
    assert_quality(torch_out, tt_out_torch, pcc=0.999)
    logger.info(f"PASSED: LTXCausalConv3d ({in_c}->{out_c}) matches reference")


@pytest.mark.parametrize(
    "in_c, out_c, T, H, W",
    [
        (128, 128, 3, 16, 16),  # Same channels (no shortcut)
        (128, 256, 3, 16, 16),  # Channel expansion (with shortcut)
    ],
    ids=["same_channels", "expand_channels"],
)
@pytest.mark.parametrize(
    "mesh_device",
    [(1, 1)],
    ids=["1x1"],
    indirect=["mesh_device"],
)
def test_ltx_resnet_block(mesh_device: ttnn.MeshDevice, in_c: int, out_c: int, T: int, H: int, W: int):
    """
    Test LTXResnetBlock3D against PyTorch ResnetBlock3D reference.
    """
    from ltx_core.model.video_vae.enums import NormLayerType
    from ltx_core.model.video_vae.resnet import ResnetBlock3D as TorchResnetBlock3D

    B = 1
    torch.manual_seed(42)

    # PyTorch reference
    torch_model = TorchResnetBlock3D(
        dims=3,
        in_channels=in_c,
        out_channels=out_c,
        norm_layer=NormLayerType.PIXEL_NORM,
    )
    torch_model.eval()

    # TT model
    tt_model = LTXResnetBlock3D(
        in_channels=in_c,
        out_channels=out_c,
        mesh_device=mesh_device,
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())

    # Input
    x = torch.randn(B, in_c, T, H, W, dtype=torch.float32)

    # PyTorch forward
    with torch.no_grad():
        torch_out = torch_model(x)

    # TT forward
    x_bthwc = x.permute(0, 2, 3, 4, 1)
    x_bthwc = conv_pad_in_channels(x_bthwc)
    x_tt = ttnn.from_torch(x_bthwc, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)

    tt_out = tt_model(x_tt)
    tt_out_torch = ttnn.to_torch(tt_out)
    tt_out_torch = tt_out_torch[:, :, :, :, :out_c]
    tt_out_torch = tt_out_torch.permute(0, 4, 1, 2, 3)

    logger.info(f"PyTorch out: {torch_out.shape}, TT out: {tt_out_torch.shape}")
    assert_quality(torch_out, tt_out_torch, pcc=0.999)
    logger.info(f"PASSED: LTXResnetBlock3D ({in_c}->{out_c}) matches reference")
