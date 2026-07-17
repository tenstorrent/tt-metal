# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Tests for LTX-2 Video VAE components.

Torch references use diffusers LTX-2 VAE modules (mirrors test_vae_wan2_1.py / test_audio_components_ltx.py).
"""

from __future__ import annotations

import os
from typing import Any

import pytest
import torch
import torch.nn as nn
from loguru import logger

import ttnn
from models.tt_dit.models.vae.vae_ltx import LTXCausalConv3d, LTXDepthToSpaceUpsample, LTXResnetBlock3D, LTXVideoDecoder
from models.tt_dit.parallel.config import ParallelFactor, VaeHWParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.check import assert_quality
from models.tt_dit.utils.conv3d import conv_pad_in_channels


def _require_diffusers_ltx_vae():
    """Return diffusers LTX-2 VAE building blocks (preferred) or LTX-Video fallbacks."""
    pytest.importorskip("diffusers")
    try:
        from diffusers.models.autoencoders.autoencoder_kl_ltx2 import (
            LTX2VideoCausalConv3d,
            LTX2VideoResnetBlock3d,
            LTX2VideoUpsampler3d,
            PerChannelRMSNorm,
        )

        return {
            "causal_conv": LTX2VideoCausalConv3d,
            "resnet": LTX2VideoResnetBlock3d,
            "upsample": LTX2VideoUpsampler3d,
            "pixel_norm": PerChannelRMSNorm,
            "ltx2": True,
        }
    except ImportError:
        from diffusers.models.autoencoders.autoencoder_kl_ltx import (
            LTXVideoCausalConv3d,
            LTXVideoResnetBlock3d,
            LTXVideoUpsampler3d,
        )
        from diffusers.models.normalization import RMSNorm

        return {
            "causal_conv": LTXVideoCausalConv3d,
            "resnet": LTXVideoResnetBlock3d,
            "upsample": LTXVideoUpsampler3d,
            "pixel_norm": RMSNorm,
            "ltx2": False,
        }


def _diffusers_resnet_state_to_tt(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remap diffusers LTX-2 resnet keys to the TT/ltx_core layout."""
    out = dict(state)
    if "conv_shortcut.weight" in out:
        out["conv_shortcut.conv.weight"] = out.pop("conv_shortcut.weight")
        if "conv_shortcut.bias" in out:
            out["conv_shortcut.conv.bias"] = out.pop("conv_shortcut.bias")
    return out


def _run_diffusers_causal_conv(conv: nn.Module, x: torch.Tensor, *, causal: bool, ltx2: bool) -> torch.Tensor:
    if ltx2:
        return conv(x, causal=causal)
    return conv(x)


def _run_diffusers_resnet(block: nn.Module, x: torch.Tensor, *, causal: bool, ltx2: bool) -> torch.Tensor:
    if ltx2:
        return block(x, causal=causal)
    return block(x)


def _run_diffusers_upsample(block: nn.Module, x: torch.Tensor, *, causal: bool, ltx2: bool) -> torch.Tensor:
    if ltx2:
        return block(x, causal=causal)
    return block(x)


class _PerChannelStatistics(nn.Module):
    """Minimal per-channel latent stats (ltx_core key names for TT weight load)."""

    def __init__(self, latent_channels: int) -> None:
        super().__init__()
        self.register_buffer("mean-of-means", torch.zeros(latent_channels))
        self.register_buffer("std-of-means", torch.ones(latent_channels))

    def un_normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.get_buffer("mean-of-means").view(1, -1, 1, 1, 1)
        std = self.get_buffer("std-of-means").view(1, -1, 1, 1, 1)
        return x * std + mean


def _diffusers_decoder_state_to_tt(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remap diffusers LTX-2 decoder keys (incl. nested resnets) to TT layout."""
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.endswith("conv_shortcut.weight"):
            out[key.replace("conv_shortcut.weight", "conv_shortcut.conv.weight")] = value
        elif key.endswith("conv_shortcut.bias"):
            out[key.replace("conv_shortcut.bias", "conv_shortcut.conv.bias")] = value
        else:
            out[key] = value
    return out


class _TorchUNetMidBlock3D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_layers: int,
        resnet_cls: type,
        ltx2: bool,
        spatial_padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        kwargs: dict[str, Any] = {"in_channels": in_channels, "out_channels": in_channels}
        if ltx2 and spatial_padding_mode != "zeros":
            kwargs["spatial_padding_mode"] = spatial_padding_mode
        self.res_blocks = nn.ModuleList([resnet_cls(**kwargs) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, *, causal: bool, ltx2: bool) -> torch.Tensor:
        for block in self.res_blocks:
            x = _run_diffusers_resnet(block, x, causal=causal, ltx2=ltx2)
        return x


def _unpatchify(x: torch.Tensor, *, patch_size_hw: int = 4, patch_size_t: int = 1) -> torch.Tensor:
    batch_size, num_channels, num_frames, height, width = x.shape
    p = patch_size_hw
    p_t = patch_size_t
    x = x.reshape(batch_size, -1, p_t, p, p, num_frames, height, width)
    return x.permute(0, 1, 5, 2, 6, 4, 7, 3).flatten(6, 7).flatten(4, 5).flatten(2, 3)


class _TorchLTXVideoDecoder(nn.Module):
    """Block-list LTX decoder built from diffusers VAE primitives (matches TT/ltx_core layout)."""

    def __init__(
        self,
        *,
        decoder_blocks: list[tuple[str, dict | int]],
        in_channels: int = 128,
        out_channels: int = 3,
        patch_size: int = 4,
        base_channels: int = 128,
        causal: bool = False,
        spatial_padding_mode: str = "zeros",
        vae_mods: dict,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.causal = causal
        self.ltx2 = vae_mods["ltx2"]
        causal_conv = vae_mods["causal_conv"]
        resnet_cls = vae_mods["resnet"]
        upsample_cls = vae_mods["upsample"]
        pixel_norm = vae_mods["pixel_norm"]

        out_channels_patched = out_channels * patch_size**2
        feature_channels = base_channels * 8

        conv_kwargs: dict[str, Any] = {}
        if self.ltx2:
            if spatial_padding_mode != "zeros":
                conv_kwargs["spatial_padding_mode"] = spatial_padding_mode
        else:
            conv_kwargs["is_causal"] = causal

        self.per_channel_statistics = _PerChannelStatistics(in_channels)
        self.conv_in = causal_conv(
            in_channels=in_channels,
            out_channels=feature_channels,
            kernel_size=3,
            stride=1,
            **conv_kwargs,
        )

        stride_map = {
            "compress_all": (2, 2, 2),
            "compress_space": (1, 2, 2),
            "compress_time": (2, 1, 1),
        }

        self.up_blocks = nn.ModuleList()
        ch = feature_channels
        for block_name, block_params in reversed(decoder_blocks):
            block_config = {"num_layers": block_params} if isinstance(block_params, int) else block_params
            if block_name == "res_x":
                self.up_blocks.append(
                    _TorchUNetMidBlock3D(
                        in_channels=ch,
                        num_layers=block_config["num_layers"],
                        resnet_cls=resnet_cls,
                        ltx2=self.ltx2,
                        spatial_padding_mode=spatial_padding_mode,
                    )
                )
            elif block_name in stride_map:
                multiplier = block_config.get("multiplier", 1)
                upsample_kwargs: dict[str, Any] = {
                    "in_channels": ch,
                    "stride": stride_map[block_name],
                    "residual": block_config.get("residual", False),
                    "upscale_factor": multiplier,
                }
                self.up_blocks.append(upsample_cls(**upsample_kwargs))
                ch = ch // multiplier
            else:
                raise ValueError(f"Unknown decoder block: {block_name}")

        self.conv_norm_out = pixel_norm()
        self.conv_act = nn.SiLU()
        self.conv_out = causal_conv(
            in_channels=ch,
            out_channels=out_channels_patched,
            kernel_size=3,
            stride=1,
            **conv_kwargs,
        )

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.per_channel_statistics.un_normalize(sample)
        sample = _run_diffusers_causal_conv(self.conv_in, sample, causal=self.causal, ltx2=self.ltx2)
        for up_block in self.up_blocks:
            if isinstance(up_block, _TorchUNetMidBlock3D):
                sample = up_block(sample, causal=self.causal, ltx2=self.ltx2)
            else:
                sample = _run_diffusers_upsample(up_block, sample, causal=self.causal, ltx2=self.ltx2)
        if self.ltx2:
            sample = self.conv_norm_out(sample)
        else:
            sample = self.conv_norm_out(sample.movedim(1, -1)).movedim(-1, 1)
        sample = self.conv_act(sample)
        sample = _run_diffusers_causal_conv(self.conv_out, sample, causal=self.causal, ltx2=self.ltx2)
        return _unpatchify(sample, patch_size_hw=self.patch_size)


def _vae_parallel_kwargs(mesh_device: ttnn.MeshDevice) -> dict:
    """Build a VaeHWParallelConfig + CCLManager for a test mesh.

    Maps mesh axis 0 → height shard, axis 1 → width shard. For a (1, 1) mesh
    both factors are 1 and ``LTXCausalConv3d`` skips halo exchange.
    """
    mesh_shape = tuple(mesh_device.shape)
    parallel_config = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=mesh_shape[0], mesh_axis=0),
        width_parallel=ParallelFactor(factor=mesh_shape[1], mesh_axis=1),
    )
    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    return {"parallel_config": parallel_config, "ccl_manager": ccl_manager}


# Fabric required for multi-device decoder tests (mirrors wan2_2 VAE tests).
_LTX_VAE_FABRIC_DEVICE_PARAMS = [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}]
_LTX_SKIP_SMALL_MESH_FABRIC = os.environ.get("LTX_VAE_FORCE_SMALL_MESH_FABRIC", "0") != "1"

# Single-chip: no fabric (avoids BH fabric router handshake when only one device is opened).
# Stand-alone conv/res/upsample tests use raw (unsharded) tensors — 1x1 only.
_LTX_VAE_MESH_DEVICE_PARAMS = [
    ((1, 1), {}),
]
_LTX_VAE_MESH_DEVICE_IDS = ["1x1"]

# Multi-mesh configs for full LTXVideoDecoder parity (mirrors test_vae_wan2_1.py).
_LTX_DECODER_MESH_PARAMS = [
    pytest.param(
        (1, 1),
        0,
        1,
        1,
        marks=pytest.mark.skipif(
            _LTX_SKIP_SMALL_MESH_FABRIC,
            reason=("Known flaky fabric handshake on 1x1 mesh; set " "LTX_VAE_FORCE_SMALL_MESH_FABRIC=1 to force-run."),
        ),
    ),
    ((2, 4), 0, 1, 1),
    ((2, 4), 1, 0, 1),
    ((1, 8), 0, 1, 1),
    pytest.param(
        (1, 4),
        1,
        0,
        1,
        marks=pytest.mark.skipif(
            _LTX_SKIP_SMALL_MESH_FABRIC,
            reason=("Known flaky fabric handshake on 1x4 mesh; set " "LTX_VAE_FORCE_SMALL_MESH_FABRIC=1 to force-run."),
        ),
    ),
    ((4, 8), 0, 1, 2),
]
_LTX_DECODER_MESH_IDS = [
    "1x1_h0_w1",
    "2x4_h0_w1",
    "2x4_h1_w0",
    "1x8_h0_w1",
    "1x4_h1_w0",
    "4x8_h0_w1",
]

# 2K decode does not fit on a single device — multi-device meshes only.
_LTX_DECODER_MESH_MULTI_ONLY_PARAMS = [
    ((2, 4), 0, 1, 1),
    ((2, 4), 1, 0, 1),
    ((4, 8), 0, 1, 2),
]
_LTX_DECODER_MESH_MULTI_ONLY_IDS = [
    "2x4_h0_w1",
    "2x4_h1_w0",
    "4x8_h0_w1",
]

# (num_frames, height, width): H/W divisible by 64; (num_frames - 1) % 8 == 0 for VAE.
_LTX_DECODER_SHAPE_PARAMS = [
    pytest.param(17, 128, 256, id="17f_128x256"),  # smoke — fast on 2x4
    pytest.param(9, 512, 832, id="9f_512x832"),  # ~480p latent grid
    pytest.param(17, 544, 960, id="17f_544x960"),  # stage-1 half of 1080p (544x960)
    pytest.param(9, 1088, 1920, id="9f_1088x1920"),  # 1080p production (modest T)
    pytest.param(141, 1088, 1920, id="141f_1088x1920"),  # full production T — hits the tuned blocking table
]

_LTX_DECODER_2K_SHAPE_PARAMS = [
    pytest.param(9, 1088, 2048, id="9f_1088x2048"),  # 2K (1088x2048, latent 34x64)
]

# Smoke resolution — only shape exercised on a 1x1 mesh in test_ltx_video_decoder.
_LTX_DECODER_1X1_MAX_HW = (256, 256)


def _skip_ltx_decoder_if_single_chip_too_large(mesh_device: ttnn.MeshDevice, height: int, width: int) -> None:
    if tuple(mesh_device.shape) == (1, 1) and (
        height > _LTX_DECODER_1X1_MAX_HW[0] or width > _LTX_DECODER_1X1_MAX_HW[1]
    ):
        pytest.skip(f"{height}x{width} LTX decode requires a multi-device mesh")


def _unwrap_tt_output(tt_out):
    if isinstance(tt_out, (tuple, list)):
        # Newer TT path may return auxiliary tensors alongside the primary output.
        return tt_out[0]
    return tt_out


# Full-decoder test — exercises the production decoder_blocks under sharding.
_LTX_PROD_DECODER_BLOCKS = [
    ("res_x", {"num_layers": 4}),
    ("compress_space", {"multiplier": 2}),
    ("res_x", {"num_layers": 6}),
    ("compress_time", {"multiplier": 2}),
    ("res_x", {"num_layers": 4}),
    ("compress_all", {"multiplier": 1}),
    ("res_x", {"num_layers": 2}),
    ("compress_all", {"multiplier": 2}),
    ("res_x", {"num_layers": 2}),
]


@pytest.mark.parametrize(
    "in_c, out_c, kernel_size, stride, T, H, W",
    [
        (128, 128, 3, 1, 3, 16, 16),  # small residual block conv
        (128, 256, 3, 1, 3, 16, 16),  # channel expansion
        (48, 128, 3, 1, 5, 32, 32),  # conv_in (after patchify)
        (128, 1024, 3, 1, 21, 17, 15),  # ltx_s0_conv_in @ 2x4 per-device (1080p path)
        (1024, 1024, 3, 1, 21, 17, 15),  # ltx_s0_res
        (1024, 4096, 3, 1, 21, 17, 15),  # ltx_s0_up (upsample projection)
        (512, 512, 3, 1, 39, 34, 30),  # ltx_s1_res
        (128, 48, 3, 1, 147, 136, 120),  # ltx_s4_out
    ],
    ids=[
        "res_128_128",
        "expand_128_256",
        "conv_in_48_128",
        "ltx_s0_conv_in",
        "ltx_s0_res",
        "ltx_s0_up",
        "ltx_s1_res",
        "ltx_s4_out",
    ],
)
@pytest.mark.parametrize("mean, std", [(0, 1), (2, 3)], ids=["std1", "std3"])
@pytest.mark.parametrize(
    "mesh_device, device_params",
    _LTX_VAE_MESH_DEVICE_PARAMS,
    ids=_LTX_VAE_MESH_DEVICE_IDS,
    indirect=["mesh_device", "device_params"],
)
def test_ltx_causal_conv3d(
    mesh_device: ttnn.MeshDevice,
    in_c: int,
    out_c: int,
    kernel_size: int,
    stride: int,
    T: int,
    H: int,
    W: int,
    mean: float,
    std: float,
):
    """
    Test LTXCausalConv3d against diffusers LTX causal conv reference.
    """
    vae_mods = _require_diffusers_ltx_vae()
    TorchCausalConv3d = vae_mods["causal_conv"]
    ltx2 = vae_mods["ltx2"]

    B = 1
    torch.manual_seed(42)

    conv_kwargs: dict[str, Any] = {
        "in_channels": in_c,
        "out_channels": out_c,
        "kernel_size": kernel_size,
        "stride": stride if ltx2 else (stride, stride, stride),
    }
    if not ltx2:
        conv_kwargs["is_causal"] = True

    torch_model = TorchCausalConv3d(**conv_kwargs)
    torch_model.eval()

    # TT model
    tt_model = LTXCausalConv3d(
        in_channels=in_c,
        out_channels=out_c,
        kernel_size=kernel_size,
        stride=stride,
        mesh_device=mesh_device,
        **_vae_parallel_kwargs(mesh_device),
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())

    # Input
    x = torch.randn(B, in_c, T, H, W, dtype=torch.float32) * std + mean

    # PyTorch forward (BCTHW format)
    with torch.no_grad():
        torch_out = _run_diffusers_causal_conv(torch_model, x, causal=True, ltx2=ltx2)

    # TT forward (BTHWC format)
    x_bthwc = x.permute(0, 2, 3, 4, 1)  # (B, T, H, W, C)
    x_bthwc = conv_pad_in_channels(x_bthwc)  # Pad C to alignment
    x_tt = ttnn.from_torch(x_bthwc, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)

    tt_out = _unwrap_tt_output(tt_model(x_tt))
    tt_out_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0])  # (B, T_out, H_out, W_out, C_out)
    tt_out_torch = tt_out_torch[:, :, :, :, :out_c]  # Trim padded channels
    tt_out_torch = tt_out_torch.permute(0, 4, 1, 2, 3)  # Back to BCTHW

    logger.info(f"PyTorch out: {torch_out.shape}, TT out: {tt_out_torch.shape}")
    assert_quality(torch_out, tt_out_torch, pcc=0.999)
    logger.info(f"PASSED: LTXCausalConv3d ({in_c}->{out_c}) matches reference")


@pytest.mark.parametrize(
    "in_c, out_c, T, H, W",
    [
        (128, 128, 3, 16, 16),  # same channels (no shortcut)
        (128, 256, 3, 16, 16),  # channel expansion (with shortcut)
        (1024, 1024, 21, 17, 15),  # ltx_s0_res @ 2x4 per-device
        (512, 512, 39, 34, 30),  # ltx_s1_res
        (256, 256, 147, 70, 66),  # ltx_s3_res @ 2K per-device
        (128, 128, 147, 138, 130),  # ltx_s4_res @ 2K per-device
    ],
    ids=[
        "same_channels",
        "expand_channels",
        "ltx_s0_res",
        "ltx_s1_res",
        "ltx_s3_res_2k",
        "ltx_s4_res_2k",
    ],
)
@pytest.mark.parametrize("mean, std", [(0, 1), (2, 3)], ids=["std1", "std3"])
@pytest.mark.parametrize(
    "mesh_device, device_params",
    _LTX_VAE_MESH_DEVICE_PARAMS,
    ids=_LTX_VAE_MESH_DEVICE_IDS,
    indirect=["mesh_device", "device_params"],
)
def test_ltx_resnet_block(
    mesh_device: ttnn.MeshDevice, in_c: int, out_c: int, T: int, H: int, W: int, mean: float, std: float
):
    """
    Test LTXResnetBlock3D against diffusers LTX ResnetBlock3D reference.
    """
    vae_mods = _require_diffusers_ltx_vae()
    TorchResnetBlock3D = vae_mods["resnet"]
    ltx2 = vae_mods["ltx2"]

    B = 1
    torch.manual_seed(42)

    torch_model = TorchResnetBlock3D(in_channels=in_c, out_channels=out_c)
    torch_model.eval()

    # TT model
    tt_model = LTXResnetBlock3D(
        in_channels=in_c,
        out_channels=out_c,
        mesh_device=mesh_device,
        **_vae_parallel_kwargs(mesh_device),
    )
    tt_model.load_torch_state_dict(_diffusers_resnet_state_to_tt(torch_model.state_dict()))

    # Input
    x = torch.randn(B, in_c, T, H, W, dtype=torch.float32) * std + mean

    # PyTorch forward
    with torch.no_grad():
        torch_out = _run_diffusers_resnet(torch_model, x, causal=True, ltx2=ltx2)

    # TT forward
    x_bthwc = x.permute(0, 2, 3, 4, 1)
    x_bthwc = conv_pad_in_channels(x_bthwc)
    x_tt = ttnn.from_torch(x_bthwc, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)

    tt_out = _unwrap_tt_output(tt_model(x_tt))
    tt_out_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0])
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
        (128, (2, 2, 2), 4, 8, 8),  # full 3D upsample (compress_all)
        (128, (1, 2, 2), 3, 8, 8),  # spatial only (compress_space)
        (128, (2, 1, 1), 4, 8, 8),  # temporal only (compress_time)
        (512, (2, 2, 2), 39, 34, 30),  # compress_all @ ltx_s1 stage
        (1024, (1, 2, 2), 21, 17, 15),  # compress_space @ ltx_s0 stage
        (256, (2, 1, 1), 75, 68, 60),  # compress_time @ ltx_s2 stage
    ],
    ids=[
        "upsample_all",
        "upsample_space",
        "upsample_time",
        "upsample_all_s1",
        "upsample_space_s0",
        "upsample_time_s2",
    ],
)
@pytest.mark.parametrize("mean, std", [(0, 1), (2, 3)], ids=["std1", "std3"])
@pytest.mark.parametrize(
    "mesh_device, device_params",
    _LTX_VAE_MESH_DEVICE_PARAMS,
    ids=_LTX_VAE_MESH_DEVICE_IDS,
    indirect=["mesh_device", "device_params"],
)
def test_ltx_depth_to_space_upsample(
    mesh_device: ttnn.MeshDevice, in_c: int, stride: tuple, T: int, H: int, W: int, mean: float, std: float
):
    """Test LTXDepthToSpaceUpsample against diffusers LTX upsampler reference."""
    vae_mods = _require_diffusers_ltx_vae()
    TorchUpsample = vae_mods["upsample"]
    ltx2 = vae_mods["ltx2"]

    B = 1
    torch.manual_seed(42)

    torch_model = TorchUpsample(in_channels=in_c, stride=stride, upscale_factor=1)
    torch_model.eval()

    tt_model = LTXDepthToSpaceUpsample(
        in_channels=in_c,
        stride=stride,
        mesh_device=mesh_device,
        **_vae_parallel_kwargs(mesh_device),
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())

    x = torch.randn(B, in_c, T, H, W, dtype=torch.float32) * std + mean
    with torch.no_grad():
        torch_out = _run_diffusers_upsample(torch_model, x, causal=True, ltx2=ltx2)

    x_bthwc = x.permute(0, 2, 3, 4, 1)
    x_bthwc = conv_pad_in_channels(x_bthwc)
    x_tt = ttnn.from_torch(x_bthwc, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)

    tt_out = _unwrap_tt_output(tt_model(x_tt))
    tt_out_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0])
    out_c = torch_out.shape[1]
    tt_out_torch = tt_out_torch[:, :, :, :, :out_c].permute(0, 4, 1, 2, 3)

    logger.info(f"Upsample stride={stride}: {x.shape} -> torch {torch_out.shape}, tt {tt_out_torch.shape}")
    assert_quality(torch_out, tt_out_torch, pcc=0.999)
    logger.info(f"PASSED: DepthToSpaceUpsample stride={stride}")


def _run_ltx_decoder_parity(
    mesh_device: ttnn.MeshDevice,
    h_axis: int,
    w_axis: int,
    num_links: int,
    num_frames: int,
    height: int,
    width: int,
    *,
    mean: float = 0.0,
    std: float = 1.0,
    pcc: float = 0.99,
):
    """Build torch + TT production decoders at ``(num_frames, height, width)`` and
    assert numerical parity. Shared by the standard multi-resolution decoder test
    and the 2K decoder test. The TT decoder is given num_frames/height/width so the
    dim walker fires and ``_BLOCKINGS`` gets an exact-match lookup for every conv3d.
    """
    vae_mods = _require_diffusers_ltx_vae()
    if not vae_mods["ltx2"]:
        pytest.skip("Full decoder parity requires diffusers autoencoder_kl_ltx2 (runtime causal= support)")

    B = 1
    torch.manual_seed(42)
    spatial_compression = 32
    latent_frames = (num_frames - 1) // 8 + 1
    latent_h = height // spatial_compression
    latent_w = width // spatial_compression

    # PyTorch reference — full production decoder shape (diffusers primitives).
    torch_decoder = _TorchLTXVideoDecoder(
        decoder_blocks=_LTX_PROD_DECODER_BLOCKS,
        in_channels=128,
        out_channels=3,
        patch_size=4,
        base_channels=128,
        causal=False,
        spatial_padding_mode="zeros",
        vae_mods=vae_mods,
    )
    torch_decoder.eval()

    # Per-channel stats set to identity so denormalization is a no-op (random
    # weights produce garbage stats otherwise).
    torch_state = torch_decoder.state_dict()
    torch_state["per_channel_statistics.mean-of-means"] = torch.zeros(128)
    torch_state["per_channel_statistics.std-of-means"] = torch.ones(128)
    torch_decoder.load_state_dict(torch_state)

    # Build VAE parallel config from h_axis / w_axis (mirrors test_wan_decoder).
    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear, num_links=num_links)
    parallel_config = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[h_axis], mesh_axis=h_axis),
        width_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[w_axis], mesh_axis=w_axis),
    )

    # TT decoder — pass num_frames/height/width so the dim walker fires and
    # _BLOCKINGS gets an exact-match lookup for every conv3d.
    tt_decoder = LTXVideoDecoder(
        decoder_blocks=_LTX_PROD_DECODER_BLOCKS,
        in_channels=128,
        out_channels=3,
        patch_size=4,
        base_channels=128,
        causal=False,
        num_frames=num_frames,
        height=height,
        width=width,
        mesh_device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )
    tt_decoder.load_torch_state_dict(_diffusers_decoder_state_to_tt(torch_decoder.state_dict()))

    # Latent shape matches what the pipeline would produce at this resolution.
    latent = torch.randn(B, 128, latent_frames, latent_h, latent_w, dtype=torch.float32) * std + mean

    with torch.no_grad():
        torch_out = torch_decoder(latent)

    tt_out = tt_decoder(latent)

    logger.info(f"Decoder: {latent.shape} -> torch {torch_out.shape}, tt {tt_out.shape}")
    assert_quality(torch_out, tt_out, pcc=pcc)
    logger.info("PASSED: LTXVideoDecoder matches PyTorch reference")


@pytest.mark.parametrize("num_frames, height, width", _LTX_DECODER_SHAPE_PARAMS)
@pytest.mark.parametrize("mean, std", [(0, 1), (2, 3)], ids=["std1", "std3"])
@pytest.mark.parametrize(
    "mesh_device, h_axis, w_axis, num_links",
    _LTX_DECODER_MESH_PARAMS,
    ids=_LTX_DECODER_MESH_IDS,
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("device_params", _LTX_VAE_FABRIC_DEVICE_PARAMS, indirect=True)
def test_ltx_video_decoder(
    mesh_device: ttnn.MeshDevice,
    num_frames: int,
    height: int,
    width: int,
    mean: float,
    std: float,
    h_axis: int,
    w_axis: int,
    num_links: int,
):
    """
    Test full LTXVideoDecoder against PyTorch VideoDecoder reference using the
    production LTX-2.3 22B decoder_blocks. Random weights — checks numerical
    parity, not absolute output quality.

    For 2x4 this validates the parallel halo-exchange path end-to-end, which
    is otherwise only covered by the slow pipeline-level test.
    """
    _skip_ltx_decoder_if_single_chip_too_large(mesh_device, height, width)
    _run_ltx_decoder_parity(mesh_device, h_axis, w_axis, num_links, num_frames, height, width, mean=mean, std=std)


@pytest.mark.parametrize("num_frames, height, width", _LTX_DECODER_2K_SHAPE_PARAMS)
@pytest.mark.parametrize("mean, std", [(0, 1)])
@pytest.mark.parametrize(
    "mesh_device, h_axis, w_axis, num_links",
    _LTX_DECODER_MESH_MULTI_ONLY_PARAMS,
    ids=_LTX_DECODER_MESH_MULTI_ONLY_IDS,
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("device_params", _LTX_VAE_FABRIC_DEVICE_PARAMS, indirect=True)
def test_ltx_video_decoder_2k(
    mesh_device: ttnn.MeshDevice,
    num_frames: int,
    height: int,
    width: int,
    mean: float,
    std: float,
    h_axis: int,
    w_axis: int,
    num_links: int,
):
    """Full LTXVideoDecoder at the 2K production resolution (2048x1080 -> 1088x2048).

    Multi-device only (2x4 is the production layout; 4x8 also exercised). The two
    512->4096 / 1024->4096 upsample-projection convs are not in the 2K ``_BLOCKINGS``
    table (they OOM a 2x4 mesh at full production T), so at 2K they fall back to the
    generic blockings — correct, but not yet perf-tuned. See conv3d.py.
    """
    _run_ltx_decoder_parity(mesh_device, h_axis, w_axis, num_links, num_frames, height, width, mean=mean, std=std)
