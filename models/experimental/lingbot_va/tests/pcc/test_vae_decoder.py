# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PCC: torch VAE decode vs TTNN WanVAEDecoder (small latents; reference VAE in fp32 on CPU for speed)."""

import pytest
import torch
import ttnn
from diffusers import AutoencoderKLWan

from models.common.metrics import compute_pcc
from models.experimental.lingbot_va.tests.mesh_utils import mesh_shape_request_param
from models.experimental.lingbot_va.tt.vae_decoder import WanVAEDecoder
from models.tt_dit.parallel.config import ParallelFactor, VaeHWParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils import cache as tt_cache
from models.tt_dit.utils.conv3d import conv_pad_height, conv_pad_in_channels

CHECKPOINT_PATH = "models/experimental/lingbot_va/reference/checkpoints/vae"
PCC_THRESHOLD = 0.99
LATENT_T = 1
LATENT_H = 8
LATENT_W = 4


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


def decode_ttnn(vae, latents, mesh_device):
    latents = latents.to(vae.dtype)
    latents_mean = (
        torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
        latents.device, latents.dtype
    )
    latents = latents / latents_std + latents_mean

    vae_parallel_config = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=1, mesh_axis=0),
        width_parallel=ParallelFactor(factor=1, mesh_axis=1),
    )
    ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)

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

    video_torch = ttnn.to_torch(tt_video_BCTHW)
    video_torch = video_torch[:, :, :, :new_logical_h, :]

    ps = getattr(vae.config, "patch_size", None)
    if ps and ps > 1:
        B, Cp, T_out, H_out, W_out = video_torch.shape
        channels = Cp // (ps * ps)
        video_torch = video_torch.view(B, channels, ps, ps, T_out, H_out, W_out)
        video_torch = video_torch.permute(0, 1, 4, 5, 3, 6, 2).contiguous()
        video_torch = video_torch.view(B, channels, T_out, H_out * ps, W_out * ps)

    return video_torch.clamp(-1.0, 1.0)


@pytest.mark.parametrize(
    "mesh_device",
    [mesh_shape_request_param()],
    indirect=True,
)
@pytest.mark.timeout(0)
def test_decode_one_video_pcc(mesh_device):
    # bf16 conv3d on CPU is very slow; fp32 uses fast MKL paths. Timeout disabled (pytest-timeout: 0 = no limit).
    vae = AutoencoderKLWan.from_pretrained(CHECKPOINT_PATH, torch_dtype=torch.float32).to("cpu")
    vae.eval()

    torch.manual_seed(42)
    latents = torch.randn(1, vae.config.z_dim, LATENT_T, LATENT_H, LATENT_W, dtype=torch.float32)

    torch_out = decode_torch(vae, latents)

    ttnn_out = decode_ttnn(vae, latents, mesh_device)

    torch_cmp = torch_out.float()
    ttnn_cmp = ttnn_out.float()
    min_c = min(torch_cmp.shape[1], ttnn_cmp.shape[1])
    min_h = min(torch_cmp.shape[3], ttnn_cmp.shape[3])
    min_w = min(torch_cmp.shape[4], ttnn_cmp.shape[4])
    torch_cmp = torch_cmp[:, :min_c, :, :min_h, :min_w]
    ttnn_cmp = ttnn_cmp[:, :min_c, :, :min_h, :min_w]

    pcc = compute_pcc(ttnn_cmp, torch_cmp)
    assert pcc >= PCC_THRESHOLD, f"PCC {pcc:.6f} < {PCC_THRESHOLD}"
