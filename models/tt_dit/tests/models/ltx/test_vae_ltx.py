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
from models.tt_dit.models.vae.vae_ltx import (
    LTXCausalConv3d,
    LTXDepthToSpaceUpsample,
    LTXResnetBlock3D,
    LTXVideoDecoder,
    LTXVideoDecoderTorch,
    LTXVideoEncoderTorch,
)
from models.tt_dit.parallel.config import ParallelFactor, VaeHWParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.check import assert_quality
from models.tt_dit.utils.conv3d import conv_pad_in_channels


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


sys.path.insert(0, "LTX-2/packages/ltx-core/src")

# Single-chip: no fabric (avoids BH fabric router handshake when only one device is opened).
# 2x4: production target for LTX-Fast AV; halo exchange uses Linear-fabric CCL.
_LTX_VAE_MESH_DEVICE_PARAMS = [
    ((1, 1), {}),
]

# Shape sanity for the small per-op tests below — they construct stand-alone
# LTXCausalConv3d / LTXResnetBlock3D / LTXDepthToSpaceUpsample with raw inputs
# (no sharding), so they only make sense on a single chip.

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
        (128, 128, 3, 1, 3, 16, 16),  # Standard residual block conv
        (128, 256, 3, 1, 3, 16, 16),  # Channel expansion
        (48, 128, 3, 1, 5, 32, 32),  # conv_in (after patchify)
    ],
    ids=["res_128_128", "expand_128_256", "conv_in_48_128"],
)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    _LTX_VAE_MESH_DEVICE_PARAMS,
    ids=["1x1"],
    indirect=["mesh_device", "device_params"],
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
        **_vae_parallel_kwargs(mesh_device),
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
    tt_out_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0])  # (B, T_out, H_out, W_out, C_out)
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
    "mesh_device, device_params",
    _LTX_VAE_MESH_DEVICE_PARAMS,
    ids=["1x1"],
    indirect=["mesh_device", "device_params"],
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
        **_vae_parallel_kwargs(mesh_device),
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
        (128, (2, 2, 2), 4, 8, 8),  # Full 3D upsample (compress_all)
        (128, (1, 2, 2), 3, 8, 8),  # Spatial only (compress_space)
        (128, (2, 1, 1), 4, 8, 8),  # Temporal only (compress_time)
    ],
    ids=["upsample_all", "upsample_space", "upsample_time"],
)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    _LTX_VAE_MESH_DEVICE_PARAMS,
    ids=["1x1"],
    indirect=["mesh_device", "device_params"],
)
def test_ltx_depth_to_space_upsample(mesh_device: ttnn.MeshDevice, in_c: int, stride: tuple, T: int, H: int, W: int):
    """Test LTXDepthToSpaceUpsample against PyTorch reference."""
    from ltx_core.model.video_vae.sampling import DepthToSpaceUpsample as TorchDTS

    B = 1
    torch.manual_seed(42)

    torch_model = TorchDTS(dims=3, in_channels=in_c, stride=stride)
    torch_model.eval()

    tt_model = LTXDepthToSpaceUpsample(
        in_channels=in_c,
        stride=stride,
        mesh_device=mesh_device,
        **_vae_parallel_kwargs(mesh_device),
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())

    x = torch.randn(B, in_c, T, H, W, dtype=torch.float32)
    with torch.no_grad():
        torch_out = torch_model(x)

    x_bthwc = x.permute(0, 2, 3, 4, 1)
    x_bthwc = conv_pad_in_channels(x_bthwc)
    x_tt = ttnn.from_torch(x_bthwc, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)

    tt_out = tt_model(x_tt)
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
    from ltx_core.model.video_vae.enums import NormLayerType, PaddingModeType
    from ltx_core.model.video_vae.video_vae import VideoDecoder as TorchVideoDecoder

    B = 1
    torch.manual_seed(42)
    spatial_compression = 32
    latent_frames = (num_frames - 1) // 8 + 1
    latent_h = height // spatial_compression
    latent_w = width // spatial_compression

    # PyTorch reference — full production decoder shape.
    torch_decoder = TorchVideoDecoder(
        convolution_dimensions=3,
        in_channels=128,
        out_channels=3,
        decoder_blocks=_LTX_PROD_DECODER_BLOCKS,
        patch_size=4,
        norm_layer=NormLayerType.PIXEL_NORM,
        causal=False,  # 22B distilled uses causal_decoder=False
        timestep_conditioning=False,
        base_channels=128,
        decoder_spatial_padding_mode=PaddingModeType.ZEROS,  # matches LTXCausalConv3d
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
    tt_decoder.load_torch_state_dict(torch_decoder.state_dict())

    # Latent shape matches what the pipeline would produce at this resolution.
    latent = torch.randn(B, 128, latent_frames, latent_h, latent_w, dtype=torch.float32) * std + mean

    with torch.no_grad():
        torch_out = torch_decoder(latent)

    tt_out = tt_decoder(latent)

    logger.info(f"Decoder: {latent.shape} -> torch {torch_out.shape}, tt {tt_out.shape}")
    assert_quality(torch_out, tt_out, pcc=pcc)
    logger.info("PASSED: LTXVideoDecoder matches PyTorch reference")


@pytest.mark.parametrize(
    ("num_frames, height, width"),
    [
        (17, 128, 256),  # latent (1, 128, 3, 4, 8) — fast for 2x4 (H=2 W=2 per device)
    ],
    ids=["17f_128x256"],
)
@pytest.mark.parametrize("mean, std", [(0, 1)])
@pytest.mark.parametrize(
    "mesh_device, h_axis, w_axis, num_links",
    [
        ((1, 1), 0, 1, 1),
        ((2, 4), 0, 1, 1),
        ((2, 4), 1, 0, 1),
    ],
    ids=[
        "1x1_h0_w1",
        "2x4_h0_w1",
        "2x4_h1_w0",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
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
    _run_ltx_decoder_parity(mesh_device, h_axis, w_axis, num_links, num_frames, height, width, mean=mean, std=std)


@pytest.mark.parametrize(
    "num_frames, height, width",
    # 2K = DCI 2048x1080. LTX requires H/W divisible by 64, so 1080 rounds up to
    # 1088; the VAE therefore sees 1088x2048 (latent 34x64). Frame count is kept
    # modest so the torch reference decode stays tractable — this validates 2K
    # *spatial* sharding/halo/blocking-lookup. Per-layer perf at the full 145-frame
    # production shape is covered by `test_bruteforce_sweep_ltx_h2w4_2k`.
    [
        (9, 1088, 2048),  # latent (1, 128, 2, 34, 64)
    ],
    ids=["9f_1088x2048"],
)
@pytest.mark.parametrize(
    "mesh_device, h_axis, w_axis, num_links",
    # Multi-device only: 2x4 is the production layout (latent 34/2=17, 64/4=16 per
    # device); 4x8 also exercised. 1x1 is omitted — a single device cannot hold a
    # full 1088x2048 decode.
    [
        ((2, 4), 0, 1, 1),
        ((2, 4), 1, 0, 1),
        ((4, 8), 0, 1, 2),
    ],
    ids=[
        "2x4_h0_w1",
        "2x4_h1_w0",
        "4x8_h0_w1",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_ltx_video_decoder_2k(
    mesh_device: ttnn.MeshDevice,
    num_frames: int,
    height: int,
    width: int,
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
    _run_ltx_decoder_parity(mesh_device, h_axis, w_axis, num_links, num_frames, height, width)


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
