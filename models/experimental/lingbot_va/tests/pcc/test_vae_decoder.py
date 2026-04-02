# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PCC: torch VAE decode vs TTNN WanVAEDecoder (small latents; reference VAE in fp32 on CPU for speed).

PCC compares tensors at bfloat16 fidelity (reference round-tripped .bfloat16().float()) to match the
device path and avoid ~0.98x PCC from fp32-vs-bf16 mismatch. Torch output is clamped like ``decode_ttnn``.

Threshold ~0.985: TT path stacks conv3d, matmul/SDPA, and pure-ttnn upsample — small per-layer drift
compounds through the decoder.
"""

import os

import pytest
import torch
import ttnn
from diffusers import AutoencoderKLWan

from models.experimental.lingbot_va.tests.mesh_utils import (
    mesh_shape_request_param,
    vae_bcthw_to_torch,
    vae_hw_parallel_config_for_mesh,
)
from models.experimental.lingbot_va.tt.vae_decoder import WanVAEDecoder
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils import cache as tt_cache
from models.tt_dit.utils.check import assert_quality
from models.tt_dit.utils.conv3d import conv_pad_height, conv_pad_in_channels
from models.tt_dit.utils.test import line_params

os.environ.setdefault("TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT", "0")

CHECKPOINT_PATH = "models/experimental/lingbot_va/reference/checkpoints/vae"
MIN_PCC = 0.985
MAX_RELATIVE_RMSE = 0.25
LATENT_T = 1
LATENT_H = 8
LATENT_W = 4


@pytest.fixture
def vae_ccl_manager(mesh_device, num_links, topology):
    return CCLManager(
        mesh_device=mesh_device,
        num_links=num_links,
        topology=topology,
    )


def decode_torch(vae, latents):
    latents = latents.to(vae.dtype)
    latents_mean = (
        torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
        latents.device, latents.dtype
    )
    latents = latents / latents_std + latents_mean
    with torch.no_grad():
        return vae.decode(latents, return_dict=False)[0]


def decode_ttnn(vae, latents, mesh_device, ccl_manager):
    latents = latents.to(vae.dtype)
    latents_mean = (
        torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
        latents.device, latents.dtype
    )
    latents = latents / latents_std + latents_mean

    vae_parallel_config = vae_hw_parallel_config_for_mesh(mesh_device)

    tt_vae = WanVAEDecoder(
        base_dim=vae.config.base_dim,
        decoder_base_dim=getattr(vae.config, "decoder_base_dim", None),
        z_dim=vae.config.z_dim,
        dim_mult=list(vae.config.dim_mult),
        num_res_blocks=vae.config.num_res_blocks,
        attn_scales=list(vae.config.attn_scales),
        temperal_downsample=list(vae.config.temperal_downsample),
        out_channels=vae.config.out_channels,
        patch_size=getattr(vae.config, "patch_size", 1) or 1,
        latents_mean=list(vae.config.latents_mean),
        latents_std=list(vae.config.latents_std),
        mesh_device=mesh_device,
        parallel_config=vae_parallel_config,
        ccl_manager=ccl_manager,
    )

    tt_cache.load_model(
        tt_vae,
        model_name="lingbot-va",
        subfolder="vae",
        parallel_config=vae_parallel_config,
        mesh_shape=tuple(mesh_device.shape),
        get_torch_state_dict=lambda: vae.state_dict(),
    )

    tt_latents_BTHWC = latents.permute(0, 2, 3, 4, 1)
    tt_latents_BTHWC = conv_pad_in_channels(tt_latents_BTHWC)
    tt_latents_BTHWC, logical_h = conv_pad_height(tt_latents_BTHWC, vae_parallel_config.height_parallel.factor)

    tt_latents_BTHWC = ttnn.from_torch(
        tt_latents_BTHWC,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        device=mesh_device,
    )

    tt_video_BCTHW, new_logical_h = tt_vae(tt_latents_BTHWC, logical_h)
    ttnn.synchronize_device(mesh_device)

    video_torch = vae_bcthw_to_torch(tt_video_BCTHW, mesh_device, vae_parallel_config, ccl_manager)
    video_torch = video_torch[:, :, :, :new_logical_h, :]

    ps = getattr(vae.config, "patch_size", None)
    if ps and ps > 1:
        B, Cp, T_out, H_out, W_out = video_torch.shape
        channels = Cp // (ps * ps)
        video_torch = video_torch.view(B, channels, ps, ps, T_out, H_out, W_out)
        video_torch = video_torch.permute(0, 1, 4, 5, 3, 6, 2).contiguous()
        video_torch = video_torch.view(B, channels, T_out, H_out * ps, W_out * ps)

    return video_torch.clamp(-1.0, 1.0)


@pytest.mark.usefixtures("reset_seeds")
@pytest.mark.parametrize(
    ("mesh_device", "num_links", "device_params", "topology"),
    [
        pytest.param(
            mesh_shape_request_param(),
            1,
            line_params,
            ttnn.Topology.Linear,
            id="lingbot_vae_decoder_pcc",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(600)
def test_decode_one_video_pcc(mesh_device, num_links, topology, vae_ccl_manager):
    assert num_links >= 1
    assert topology == ttnn.Topology.Linear
    vae = AutoencoderKLWan.from_pretrained(CHECKPOINT_PATH, torch_dtype=torch.float32).to("cpu")
    vae.eval()

    torch.manual_seed(42)
    latents = torch.randn(1, vae.config.z_dim, LATENT_T, LATENT_H, LATENT_W, dtype=torch.float32)

    torch_out = decode_torch(vae, latents)
    torch_out = torch_out.float().clamp(-1.0, 1.0)

    ttnn_out = decode_ttnn(vae, latents, mesh_device, vae_ccl_manager)

    torch_cmp = torch_out.bfloat16().float()
    ttnn_cmp = ttnn_out.float()
    min_c = min(torch_cmp.shape[1], ttnn_cmp.shape[1])
    min_h = min(torch_cmp.shape[3], ttnn_cmp.shape[3])
    min_w = min(torch_cmp.shape[4], ttnn_cmp.shape[4])
    torch_cmp = torch_cmp[:, :min_c, :, :min_h, :min_w]
    ttnn_cmp = ttnn_cmp[:, :min_c, :, :min_h, :min_w]

    assert_quality(torch_cmp, ttnn_cmp, pcc=MIN_PCC, relative_rmse=MAX_RELATIVE_RMSE)
