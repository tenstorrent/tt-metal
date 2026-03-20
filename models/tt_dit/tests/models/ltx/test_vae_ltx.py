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
from models.tt_dit.models.vae.ltx.vae_ltx import (
    LTXCausalConv3d,
    LTXDepthToSpaceUpsample,
    LTXResnetBlock3D,
    LTXVideoDecoder,
    LTXVideoDecoderTorch,
    LTXVideoEncoderTorch,
)
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
    # Lower threshold for channel-expansion blocks (GroupNorm(1) vs layer_norm precision)
    min_pcc = 0.995 if in_c != out_c else 0.999
    assert_quality(torch_out, tt_out_torch, pcc=min_pcc)
    logger.info(f"PASSED: LTXResnetBlock3D ({in_c}->{out_c}) matches reference")


@pytest.mark.parametrize(
    "in_c, stride, T, H, W",
    [
        (128, (2, 2, 2), 4, 8, 8),  # Full 3D upsample (compress_all)
        (128, (1, 2, 2), 3, 8, 8),  # Spatial only (compress_space)
        (128, (2, 1, 1), 4, 8, 8),  # Temporal only (compress_time)
    ],
    ids=["upsample_all", "upsample_space", "upsample_time"],
)
@pytest.mark.parametrize(
    "mesh_device",
    [(1, 1)],
    ids=["1x1"],
    indirect=["mesh_device"],
)
def test_ltx_depth_to_space_upsample(mesh_device: ttnn.MeshDevice, in_c: int, stride: tuple, T: int, H: int, W: int):
    """Test LTXDepthToSpaceUpsample against PyTorch reference."""
    from ltx_core.model.video_vae.sampling import DepthToSpaceUpsample as TorchDTS

    B = 1
    torch.manual_seed(42)

    torch_model = TorchDTS(dims=3, in_channels=in_c, stride=stride)
    torch_model.eval()

    tt_model = LTXDepthToSpaceUpsample(in_channels=in_c, stride=stride, mesh_device=mesh_device)
    tt_model.load_torch_state_dict(torch_model.state_dict())

    x = torch.randn(B, in_c, T, H, W, dtype=torch.float32)
    with torch.no_grad():
        torch_out = torch_model(x)

    x_bthwc = x.permute(0, 2, 3, 4, 1)
    x_bthwc = conv_pad_in_channels(x_bthwc)
    x_tt = ttnn.from_torch(x_bthwc, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)

    tt_out = tt_model(x_tt)
    tt_out_torch = ttnn.to_torch(tt_out)
    out_c = torch_out.shape[1]
    tt_out_torch = tt_out_torch[:, :, :, :, :out_c].permute(0, 4, 1, 2, 3)

    logger.info(f"Upsample stride={stride}: {x.shape} -> torch {torch_out.shape}, tt {tt_out_torch.shape}")
    assert_quality(torch_out, tt_out_torch, pcc=0.999)
    logger.info(f"PASSED: DepthToSpaceUpsample stride={stride}")


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 1)],
    ids=["1x1"],
    indirect=["mesh_device"],
)
def test_ltx_video_decoder(mesh_device: ttnn.MeshDevice):
    """
    Test full LTXVideoDecoder against PyTorch VideoDecoder reference.
    Uses small dimensions for fast testing.
    """
    from ltx_core.model.video_vae.enums import NormLayerType
    from ltx_core.model.video_vae.video_vae import VideoDecoder as TorchVideoDecoder

    B = 1
    torch.manual_seed(42)

    decoder_blocks = [
        ("compress_all", {"multiplier": 2}),
        ("compress_all", {"multiplier": 2}),
        ("compress_time", {"multiplier": 2}),
        ("compress_space", {"multiplier": 2}),
    ]

    # PyTorch reference
    torch_decoder = TorchVideoDecoder(
        convolution_dimensions=3,
        in_channels=128,
        out_channels=3,
        decoder_blocks=decoder_blocks,
        patch_size=4,
        norm_layer=NormLayerType.PIXEL_NORM,
        causal=True,
        timestep_conditioning=False,
        base_channels=128,
        # Default: reflect spatial padding (matches TTNN implementation)
    )
    torch_decoder.eval()

    # Set per-channel stats to identity so denormalization is a no-op
    # (random weights produce garbage stats)
    torch_state = torch_decoder.state_dict()
    torch_state["per_channel_statistics.mean-of-means"] = torch.zeros(128)
    torch_state["per_channel_statistics.std-of-means"] = torch.ones(128)
    torch_decoder.load_state_dict(torch_state)

    # TT decoder
    tt_decoder = LTXVideoDecoder(
        decoder_blocks=decoder_blocks,
        in_channels=128,
        out_channels=3,
        patch_size=4,
        base_channels=128,
        mesh_device=mesh_device,
    )
    tt_decoder.load_torch_state_dict(torch_decoder.state_dict())

    # Small latent input
    latent = torch.randn(B, 128, 3, 4, 4, dtype=torch.float32)

    # PyTorch forward
    with torch.no_grad():
        torch_out = torch_decoder(latent)

    # TT forward
    tt_out = tt_decoder(latent)

    logger.info(f"Decoder: {latent.shape} -> torch {torch_out.shape}, tt {tt_out.shape}")
    assert_quality(torch_out, tt_out, pcc=0.99)
    logger.info("PASSED: LTXVideoDecoder matches PyTorch reference")


def test_ltx_vae_roundtrip():
    """Test VAE encode → decode round-trip (torch-only, no device needed)."""
    torch.manual_seed(42)

    encoder_blocks = [
        ("compress_space_res", {}),
        ("compress_time_res", {}),
        ("compress_all_res", {}),
        ("compress_all_res", {}),
    ]
    decoder_blocks = [
        ("compress_all", {"multiplier": 2}),
        ("compress_all", {"multiplier": 2}),
        ("compress_time", {"multiplier": 2}),
        ("compress_space", {"multiplier": 2}),
    ]

    encoder = LTXVideoEncoderTorch.from_config(encoder_blocks)
    decoder = LTXVideoDecoderTorch.from_config(decoder_blocks)

    # Random video: (B, 3, F, H, W) — F must be 1 + 8k
    video = torch.randn(1, 3, 17, 128, 128)

    latent = encoder.encode(video)
    logger.info(f"Encode: {video.shape} -> {latent.shape}")

    reconstructed = decoder.decode(latent)
    logger.info(f"Decode: {latent.shape} -> {reconstructed.shape}")

    assert reconstructed.shape == video.shape, f"Shape mismatch: {reconstructed.shape} != {video.shape}"
    logger.info("PASSED: VAE encode->decode round-trip shapes correct")
