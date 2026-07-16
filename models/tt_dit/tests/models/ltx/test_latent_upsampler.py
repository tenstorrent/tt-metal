# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""PCC tests for ``LTXLatentUpsampler`` vs the ``ltx_core`` reference.

All shapes match the production LTX-2 Fast pipeline (mid_channels=1024, num_frames=19,
2x4 BH mesh, half-res latent 17x30 / 18x32)."""

import sys

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.models.upsampler.latent_upsampler_ltx import LTXLatentUpsampler, LTXUpsamplerResBlock
from models.tt_dit.parallel.config import ParallelFactor, VaeHWParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.check import assert_quality
from models.tt_dit.utils.conv3d import ConvDims, conv_pad_height, conv_pad_width
from models.tt_dit.utils.tensor import fast_device_to_host, typed_tensor_2dshard

sys.path.insert(0, "LTX-2/packages/ltx-core/src")


_MESH_2x4 = ((2, 4), {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 23887872})


def _parallel_kwargs(mesh_device: ttnn.MeshDevice) -> dict:
    mesh_shape = tuple(mesh_device.shape)
    parallel_config = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=mesh_shape[0], mesh_axis=0),
        width_parallel=ParallelFactor(factor=mesh_shape[1], mesh_axis=1),
    )
    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    return {"parallel_config": parallel_config, "ccl_manager": ccl_manager}


@pytest.mark.parametrize(
    "H, W",
    [(17, 30), (18, 32)],
    ids=["nondiv_17x30", "div_18x32"],
)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [_MESH_2x4],
    ids=["2x4"],
    indirect=["mesh_device", "device_params"],
)
def test_ltx_upsampler_resblock(mesh_device: ttnn.MeshDevice, H: int, W: int):
    """Single LTXUpsamplerResBlock at the production shape (channels=1024, 2x4 sharded,
    T=19). ``nondiv_17x30`` exercises the sharded pad/mask/crop path; ``div_18x32`` is
    the no-mask control."""
    from ltx_core.model.upsampler.res_block import ResBlock as TorchResBlock

    channels, T, B = 1024, 19, 1
    torch.manual_seed(0xABBA)
    pk = _parallel_kwargs(mesh_device)
    pc = pk["parallel_config"]
    hf, wf = pc.height_parallel.factor, pc.width_parallel.factor
    padded_h = ((H + hf - 1) // hf) * hf
    padded_w = ((W + wf - 1) // wf) * wf

    torch_block = TorchResBlock(channels=channels, dims=3)
    torch_block.eval()

    tt_block = LTXUpsamplerResBlock(
        channels=channels,
        gn_input_nhw=T * H * W,
        mesh_device=mesh_device,
        conv_dims=ConvDims(T=T + 2, H=padded_h // hf, W=padded_w // wf),
        **pk,
    )
    tt_block.load_torch_state_dict(torch_block.state_dict())

    x = torch.randn(B, channels, T, H, W, dtype=torch.float32)
    with torch.no_grad():
        torch_out = torch_block(x)  # (B, C, T, H, W)

    x_bthwc = x.permute(0, 2, 3, 4, 1).contiguous()  # (B, T, H, W, C)
    x_bthwc, logical_h = conv_pad_height(x_bthwc, hf)
    x_bthwc, logical_w = conv_pad_width(x_bthwc, wf)
    x_tt = typed_tensor_2dshard(
        x_bthwc,
        mesh_device,
        shard_mapping={pc.height_parallel.mesh_axis: 2, pc.width_parallel.mesh_axis: 3},
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
    )

    out = tt_block(x_tt, logical_h, logical_w)

    concat_dims = [None, None]
    concat_dims[pc.height_parallel.mesh_axis] = 2
    concat_dims[pc.width_parallel.mesh_axis] = 3
    out_torch = fast_device_to_host(out, mesh_device, concat_dims, ccl_manager=pk["ccl_manager"])
    out_torch = out_torch[:, :, :logical_h, :logical_w, :channels].permute(0, 4, 1, 2, 3)

    assert out_torch.shape == torch_out.shape, f"shape mismatch: TT {out_torch.shape} vs torch {torch_out.shape}"
    assert_quality(torch_out, out_torch, pcc=0.99)
    logger.info(f"PASSED: LTXUpsamplerResBlock(1024 ch, {T}x{H}x{W}, 2x4) matches reference")


@pytest.mark.parametrize(
    "H, W",
    [(17, 30), (18, 32)],
    ids=["nondiv_17x30", "div_18x32"],
)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [_MESH_2x4],
    ids=["2x4"],
    indirect=["mesh_device", "device_params"],
)
def test_ltx_latent_upsampler(mesh_device: ttnn.MeshDevice, H: int, W: int):
    """Full ``LTXLatentUpsampler`` PCC vs reference, random weights at production shape
    (in=128, mid=1024, T=19, 2x4). No HF checkpoint required."""
    from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel

    in_c, mid_c, T, B = 128, 1024, 19, 1
    torch.manual_seed(0xC0FFEE)

    torch_model = LTX2LatentUpsamplerModel(
        in_channels=in_c,
        mid_channels=mid_c,
        num_blocks_per_stage=4,
        dims=3,
        spatial_upsample=True,
        temporal_upsample=False,
        rational_spatial_scale=2.0,
        use_rational_resampler=False,
    )
    torch_model.eval()

    tt_model = LTXLatentUpsampler(
        input_hw=(H, W),
        in_channels=in_c,
        mid_channels=mid_c,
        num_blocks_per_stage=4,
        spatial_upsample=True,
        temporal_upsample=False,
        spatial_scale=2.0,
        rational_resampler=False,
        mesh_device=mesh_device,
        num_frames=T,
        **_parallel_kwargs(mesh_device),
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())

    latent = torch.randn(B, in_c, T, H, W, dtype=torch.float32)

    with torch.no_grad():
        torch_out = torch_model(latent)
    tt_out = tt_model(latent)

    assert tt_out.shape == torch_out.shape, f"shape mismatch: TT {tt_out.shape} vs torch {torch_out.shape}"
    assert_quality(torch_out, tt_out, pcc=0.99)
    logger.info(f"PASSED: LTXLatentUpsampler({in_c}->{mid_c}, {T}x{H}x{W} → {T}x{H * 2}x{W * 2}) matches reference")


def _resolve_upsampler_checkpoint() -> str | None:
    """Resolve the real spatial-upscaler safetensors (local HF cache, no download)."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import LocalEntryNotFoundError

    repo_id, filename = "Lightricks/LTX-2.3", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
    try:
        return hf_hub_download(repo_id, filename, local_files_only=True)
    except (LocalEntryNotFoundError, Exception):  # noqa: BLE001
        return None


@pytest.mark.parametrize(
    "T, H, W",
    [(19, 17, 30), (19, 18, 32)],
    ids=["nondiv_17x30", "div_18x32"],
)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [_MESH_2x4],
    ids=["2x4"],
    indirect=["mesh_device", "device_params"],
)
def test_ltx_latent_upsampler_real_checkpoint(mesh_device: ttnn.MeshDevice, T: int, H: int, W: int):
    """PCC vs reference with the real ``ltx-2.3-spatial-upscaler-x2`` weights (mid_channels=1024)
    on the production 2x4 mesh. Input is a smooth, per-channel-scaled latent: white noise
    is unrepresentative and lower-bounds bf16 conv PCC through the 8-block residual chain."""
    import json

    from ltx_core.model.upsampler.model import LatentUpsampler as TorchLatentUpsampler
    from safetensors import safe_open
    from safetensors.torch import load_file

    ckpt = _resolve_upsampler_checkpoint()
    if ckpt is None:
        pytest.skip("real upsampler checkpoint not in local HF cache")

    with safe_open(ckpt, framework="pt") as f:
        cfg = json.loads(f.metadata()["config"])
    in_c, mid_c = cfg["in_channels"], cfg["mid_channels"]
    n_blocks = cfg["num_blocks_per_stage"]
    B = 1
    torch.manual_seed(0xC0FFEE)

    torch_model = TorchLatentUpsampler(
        in_channels=in_c,
        mid_channels=mid_c,
        num_blocks_per_stage=n_blocks,
        dims=3,
        spatial_upsample=True,
        temporal_upsample=False,
        spatial_scale=2.0,
        rational_resampler=False,
    )
    torch_model.load_state_dict(load_file(ckpt))
    torch_model.eval()

    tt_model = LTXLatentUpsampler(
        input_hw=(H, W),
        in_channels=in_c,
        mid_channels=mid_c,
        num_blocks_per_stage=n_blocks,
        spatial_upsample=True,
        temporal_upsample=False,
        spatial_scale=2.0,
        rational_resampler=False,
        mesh_device=mesh_device,
        num_frames=T,
        **_parallel_kwargs(mesh_device),
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())

    # Smooth, spatially-correlated latent with per-channel scale (mimics real VAE latents).
    noise = torch.randn(B, in_c, T, H, W, dtype=torch.float32)
    latent = torch.nn.functional.avg_pool3d(noise, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
    latent = latent * (torch.rand(in_c) * 4 + 0.5).view(1, in_c, 1, 1, 1)

    with torch.no_grad():
        torch_out = torch_model(latent)
    tt_out = tt_model(latent)

    assert tt_out.shape == torch_out.shape, f"shape mismatch: TT {tt_out.shape} vs torch {torch_out.shape}"
    assert_quality(torch_out, tt_out, pcc=0.997)
    logger.info(f"PASSED: real-checkpoint LTXLatentUpsampler({in_c}->{mid_c}, {n_blocks} blocks) matches reference")
