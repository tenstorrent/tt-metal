# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PCC test for Lingbot-VA VAE decoder: compare torch _decode_one_video vs TTNN LingbotVAEDecoder.

Loads the real VAE checkpoint, generates random latents in the expected latent space,
runs both the diffusers (torch) decode path and the TTNN decode path, and computes
PCC between the two outputs.
"""

import sys
import time
from pathlib import Path

import torch

# Repo root so we can import reference.*
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import ttnn
from diffusers import AutoencoderKLWan
from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify

from models.common.metrics import compute_pcc
from models.tt_dit.parallel.config import ParallelFactor, VaeHWParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils import cache as tt_cache
from models.tt_dit.utils.conv3d import conv_pad_height, conv_pad_in_channels

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

CHECKPOINT_PATH = "models/experimental/lingbot_va/reference/checkpoints/vae"
PCC_THRESHOLD = 0.99
BATCH_SIZE = 1
# Latent dims: small for fast CPU decode, but H*W >= 32 for SDPA tile size.
# L1 overflow fixed by blocking configs (C_in_block=128) and SDPA k_chunk=32.
LATENT_T = 1
LATENT_H = 8
LATENT_W = 4


# ─────────────────────────────────────────────
# Torch reference decode (same as _decode_one_video)
# ─────────────────────────────────────────────


def decode_torch(vae, latents):
    """Run the diffusers AutoencoderKLWan.decode (torch, CPU)."""
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
    return video  # [B, C_rgb, T_out, H_out, W_out]


# ─────────────────────────────────────────────
# TTNN decode
# ─────────────────────────────────────────────


def decode_ttnn(vae, latents, mesh_device):
    """Run the TTNN LingbotVAEDecoder on device, return torch tensor."""
    from models.experimental.lingbot_va.tt.vae_decoder import LingbotVAEDecoder

    # De-normalize (same math as torch path)
    latents = latents.to(vae.dtype)
    latents_mean = (
        torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
        latents.device, latents.dtype
    )
    latents = latents / latents_std + latents_mean

    # Build and load TTNN decoder
    vae_parallel_config = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=1, mesh_axis=0),
        width_parallel=ParallelFactor(factor=1, mesh_axis=0),
    )
    ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)

    tt_vae = LingbotVAEDecoder(
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

    # BCTHW → BTHWC, pad for ttnn
    tt_latents_BTHWC = latents.permute(0, 2, 3, 4, 1)
    tt_latents_BTHWC = conv_pad_in_channels(tt_latents_BTHWC)
    tt_latents_BTHWC, logical_h = conv_pad_height(tt_latents_BTHWC, vae_parallel_config.height_parallel.factor)

    tt_latents_BTHWC = ttnn.from_torch(
        tt_latents_BTHWC,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        device=mesh_device,
    )

    # Decode on device
    tt_video_BCTHW, new_logical_h = tt_vae(tt_latents_BTHWC, logical_h)
    ttnn.synchronize_device(mesh_device)

    video_torch = ttnn.to_torch(tt_video_BCTHW)  # [B, C_out, T_out, H_out, W_out]
    video_torch = video_torch[:, :, :, :new_logical_h, :]

    # Unpatchify
    ps = getattr(vae.config, "patch_size", None)
    if ps and ps > 1:
        video_torch = unpatchify(video_torch, ps)

    video_torch = video_torch.clamp(-1.0, 1.0)
    return video_torch


# ─────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────


def test_decode_one_video_pcc():
    """Compare torch vs TTNN VAE decode outputs with PCC."""

    print("=" * 60)
    print("test_decode_one_video_pcc")
    print("=" * 60)

    # ── 1. Load torch VAE ──
    print(f"Loading AutoencoderKLWan from {CHECKPOINT_PATH}...")
    vae = AutoencoderKLWan.from_pretrained(
        CHECKPOINT_PATH,
        torch_dtype=torch.bfloat16,
    ).to(device="cpu")
    vae.eval()

    z_dim = vae.config.z_dim
    print(
        f"VAE config: z_dim={z_dim}, out_channels={vae.config.out_channels}, "
        f"patch_size={getattr(vae.config, 'patch_size', None)}, "
        f"is_residual={getattr(vae.config, 'is_residual', False)}"
    )

    # ── 2. Random latents (in normalized latent space) ──
    torch.manual_seed(42)
    latents = torch.randn(BATCH_SIZE, z_dim, LATENT_T, LATENT_H, LATENT_W, dtype=torch.float32)
    print(f"Input latents shape: {latents.shape}")

    # ── 3. Torch decode ──
    print("\nRunning torch decode...")
    t0 = time.time()
    torch_out = decode_torch(vae, latents.clone())
    t_torch = time.time() - t0
    print(f"Torch output shape: {torch_out.shape}, time: {t_torch:.1f}s")

    # ── 4. TTNN decode ──
    print("\nSetting up TT device...")
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    mesh_device.enable_program_cache()

    try:
        print("Running TTNN decode...")
        t0 = time.time()
        ttnn_out = decode_ttnn(vae, latents.clone(), mesh_device)
        t_ttnn = time.time() - t0
        print(f"TTNN output shape: {ttnn_out.shape}, time: {t_ttnn:.1f}s")

        # ── 5. Align shapes for comparison ──
        torch_out_f = torch_out.float()
        ttnn_out_f = ttnn_out.float()

        # Trim to matching shapes (TTNN may have slight padding differences)
        min_c = min(torch_out_f.shape[1], ttnn_out_f.shape[1])
        min_t = min(torch_out_f.shape[2], ttnn_out_f.shape[2])
        min_h = min(torch_out_f.shape[3], ttnn_out_f.shape[3])
        min_w = min(torch_out_f.shape[4], ttnn_out_f.shape[4])

        torch_trimmed = torch_out_f[:, :min_c, :min_t, :min_h, :min_w]
        ttnn_trimmed = ttnn_out_f[:, :min_c, :min_t, :min_h, :min_w]

        if torch_trimmed.shape != ttnn_trimmed.shape:
            print(f"Shape mismatch after trim: torch={torch_trimmed.shape}, ttnn={ttnn_trimmed.shape}")

        # ── 6. Compute PCC ──
        pcc = compute_pcc(ttnn_trimmed, torch_trimmed)
        max_err = (torch_trimmed - ttnn_trimmed).abs().max().item()
        mean_err = (torch_trimmed - ttnn_trimmed).abs().mean().item()

        # ── 7. Report ──
        print("\n" + "=" * 60)
        print("COMPARISON RESULTS")
        print("=" * 60)
        print(f"Torch  output shape    : {torch_out.shape}")
        print(f"TTNN   output shape    : {ttnn_out.shape}")
        print(f"Compared region        : {torch_trimmed.shape}")
        print(f"PCC                    : {pcc:.6f}  (threshold={PCC_THRESHOLD})")
        print(f"Max  absolute error    : {max_err:.6f}")
        print(f"Mean absolute error    : {mean_err:.6f}")
        print(f"Torch decode time      : {t_torch:.1f}s")
        print(f"TTNN  decode time      : {t_ttnn:.1f}s")
        print("=" * 60)

        passed = pcc >= PCC_THRESHOLD
        if passed:
            print("TEST PASSED")
        else:
            print(f"TEST FAILED - PCC {pcc:.6f} < {PCC_THRESHOLD}")

        assert passed, f"PCC {pcc:.6f} is below threshold {PCC_THRESHOLD}"

    finally:
        ttnn.close_mesh_device(mesh_device)

    return pcc, max_err, mean_err


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    try:
        pcc, max_err, mean_err = test_decode_one_video_pcc()
        sys.exit(0)
    except Exception as e:
        print(f"\nTest failed with exception: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
