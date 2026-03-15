# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PCC test: compare torch VAE decode vs TTNN WanVAEDecoder end-to-end.

Uses small latent dims (T=1, H=8, W=4) so the torch reference finishes in
a reasonable time on CPU.  H*W must be >= 32 for SDPA seq_len requirements.
"""

import time

import torch
import ttnn
from diffusers import AutoencoderKLWan

from models.common.metrics import compute_pcc
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
    """Reference torch decode (mirrors _decode_one_video)."""
    latents = latents.to(vae.dtype)
    latents_mean = (
        torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
        latents.device, latents.dtype
    )
    latents = latents / latents_std + latents_mean
    with torch.no_grad():
        video = vae.decode(latents, return_dict=False)[0]
    return video


def decode_ttnn(vae, latents, mesh_device):
    """TTNN decode path using WanVAEDecoder."""
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
        width_parallel=ParallelFactor(factor=1, mesh_axis=0),
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

    video_torch = video_torch.clamp(-1.0, 1.0)
    return video_torch


def test_decode_one_video_pcc():
    print("=" * 60)
    print("test_decode_one_video_pcc")
    print("=" * 60)

    print(f"Loading AutoencoderKLWan from {CHECKPOINT_PATH}...")
    vae = AutoencoderKLWan.from_pretrained(CHECKPOINT_PATH, torch_dtype=torch.bfloat16).to("cpu")
    vae.eval()
    print(
        f"VAE config: z_dim={vae.config.z_dim}, out_channels={vae.config.out_channels}, "
        f"patch_size={getattr(vae.config, 'patch_size', None)}, "
        f"is_residual={getattr(vae.config, 'is_residual', False)}"
    )

    torch.manual_seed(42)
    latents = torch.randn(1, vae.config.z_dim, LATENT_T, LATENT_H, LATENT_W, dtype=torch.float32)
    print(f"Input latents shape: {latents.shape}")

    # Torch reference
    print("\nRunning torch decode...")
    t0 = time.time()
    torch_out = decode_torch(vae, latents)
    torch_time = time.time() - t0
    print(f"Torch output shape: {torch_out.shape}, time: {torch_time:.1f}s")

    # TTNN
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    print("\nRunning TTNN decode...")
    t0 = time.time()
    ttnn_out = decode_ttnn(vae, latents, mesh_device)
    ttnn_time = time.time() - t0
    print(f"TTNN output shape: {ttnn_out.shape}, time: {ttnn_time:.1f}s")

    # Compare
    torch_cmp = torch_out.float()
    ttnn_cmp = ttnn_out.float()
    min_c = min(torch_cmp.shape[1], ttnn_cmp.shape[1])
    min_h = min(torch_cmp.shape[3], ttnn_cmp.shape[3])
    min_w = min(torch_cmp.shape[4], ttnn_cmp.shape[4])
    torch_cmp = torch_cmp[:, :min_c, :, :min_h, :min_w]
    ttnn_cmp = ttnn_cmp[:, :min_c, :, :min_h, :min_w]

    pcc = compute_pcc(ttnn_cmp, torch_cmp)
    max_err = (torch_cmp - ttnn_cmp).abs().max().item()
    mean_err = (torch_cmp - ttnn_cmp).abs().mean().item()

    print()
    print("=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"Torch  output shape    : {torch_out.shape}")
    print(f"TTNN   output shape    : {ttnn_out.shape}")
    print(f"Compared region        : {torch_cmp.shape}")
    print(f"PCC                    : {pcc:.6f}  (threshold={PCC_THRESHOLD})")
    print(f"Max  absolute error    : {max_err:.6f}")
    print(f"Mean absolute error    : {mean_err:.6f}")
    print(f"Torch decode time      : {torch_time:.1f}s")
    print(f"TTNN  decode time      : {ttnn_time:.1f}s")
    print("=" * 60)

    ttnn.close_mesh_device(mesh_device)

    assert pcc >= PCC_THRESHOLD, f"PCC {pcc:.6f} below threshold {PCC_THRESHOLD}"
    print("TEST PASSED")


if __name__ == "__main__":
    test_decode_one_video_pcc()
