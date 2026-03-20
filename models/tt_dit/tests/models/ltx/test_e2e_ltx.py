# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
End-to-end LTX-2 AudioVideo test.

Runs a 1-layer model with random weights to verify the full pipeline:
text → DiT denoise (video+audio) → TTNN VAE decode (video) → export

Uses PyTorch LTXModel for the transformer (AudioVideo mode) and
TTNN VAE decoder for video decoding.
"""

import sys

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.models.vae.ltx.vae_ltx import LTXVideoDecoder
from models.tt_dit.pipelines.ltx.pipeline_ltx import compute_sigmas, euler_step

sys.path.insert(0, "LTX-2/packages/ltx-core/src")


@pytest.mark.parametrize(
    "mesh_device, sp_axis, tp_axis",
    [
        [(1, 1), 0, 1],
        [(2, 4), 0, 1],
    ],
    ids=["1x1sp0tp1", "2x4sp0tp1"],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_e2e_audio_video(mesh_device: ttnn.MeshDevice, sp_axis: int, tp_axis: int):
    """
    End-to-end AudioVideo generation test.

    1. PyTorch LTXModel(AudioVideo) for DiT denoising (1 layer, random weights)
    2. TTNN VAE decoder for video
    3. Torch VAE decoder for audio (placeholder)
    4. Verify output shapes match expected video + audio dimensions
    """
    from ltx_core.guidance.perturbations import BatchedPerturbationConfig, PerturbationConfig
    from ltx_core.model.transformer.model import LTXModel, LTXModelType, Modality
    from ltx_core.model.video_vae.enums import NormLayerType
    from ltx_core.model.video_vae.video_vae import VideoDecoder

    torch.manual_seed(42)
    B = 1
    num_layers = 1
    num_inference_steps = 2

    # Video dimensions
    num_frames = 17
    px_height, px_width = 128, 128
    latent_frames = (num_frames - 1) // 8 + 1  # 3
    latent_h = px_height // 32  # 4
    latent_w = px_width // 32  # 4
    video_tokens = latent_frames * latent_h * latent_w  # 48

    # Audio dimensions (simplified: 64 tokens per chunk)
    audio_tokens = 64

    logger.info(f"Video: {num_frames}f @ {px_height}x{px_width} -> {video_tokens} tokens")
    logger.info(f"Audio: {audio_tokens} tokens")

    # === 1. Create PyTorch AudioVideo transformer ===
    torch_model = LTXModel(
        model_type=LTXModelType.AudioVideo,
        num_layers=num_layers,
        num_attention_heads=32,
        attention_head_dim=128,
        in_channels=128,
        out_channels=128,
        cross_attention_dim=4096,
        audio_num_attention_heads=32,
        audio_attention_head_dim=64,
        audio_in_channels=128,
        audio_out_channels=128,
        audio_cross_attention_dim=2048,
        use_middle_indices_grid=True,
    )
    torch_model.eval()

    # === 2. Create TTNN VAE decoder ===
    decoder_blocks = [
        ("compress_all", {"multiplier": 2}),
        ("compress_all", {"multiplier": 2}),
        ("compress_time", {"multiplier": 2}),
        ("compress_space", {"multiplier": 2}),
    ]
    torch_vae = VideoDecoder(
        convolution_dimensions=3,
        in_channels=128,
        out_channels=3,
        decoder_blocks=decoder_blocks,
        patch_size=4,
        norm_layer=NormLayerType.PIXEL_NORM,
        causal=True,
        timestep_conditioning=False,
        base_channels=128,
    )
    torch_vae.eval()
    vae_state = torch_vae.state_dict()
    vae_state["per_channel_statistics.mean-of-means"] = torch.zeros(128)
    vae_state["per_channel_statistics.std-of-means"] = torch.ones(128)
    torch_vae.load_state_dict(vae_state)

    tt_vae = LTXVideoDecoder(
        decoder_blocks=decoder_blocks,
        mesh_device=mesh_device,
    )
    tt_vae.load_torch_state_dict(torch_vae.state_dict())

    # === 3. Sigma schedule ===
    sigmas = compute_sigmas(steps=num_inference_steps, num_tokens=video_tokens)

    # === 4. Initial noise ===
    video_latent = torch.randn(B, video_tokens, 128) * sigmas[0]
    audio_latent = torch.randn(B, audio_tokens, 128) * sigmas[0]

    # Positions
    F, H, W = latent_frames, latent_h, latent_w
    t_ids = torch.arange(F)
    h_ids = torch.arange(H)
    w_ids = torch.arange(W)
    grid_t, grid_h, grid_w = torch.meshgrid(t_ids, h_ids, w_ids, indexing="ij")
    video_pos = (
        torch.stack([grid_t.flatten(), grid_h.flatten(), grid_w.flatten()], dim=0)
        .float()
        .unsqueeze(-1)
        .repeat(1, 1, 2)
        .unsqueeze(0)
    )  # (1, 3, T, 2)

    audio_pos = (
        torch.arange(audio_tokens).float().unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 2)
    )  # (1, 1, T, 2)

    # Context
    video_context = torch.randn(B, 16, 4096)
    audio_context = torch.randn(B, 16, 2048)

    # === 5. Denoising loop (PyTorch) ===
    for step_idx in range(num_inference_steps):
        sigma = sigmas[step_idx].item()
        sigma_next = sigmas[step_idx + 1].item()

        video_mod = Modality(
            latent=video_latent,
            sigma=torch.tensor([sigma]),
            timesteps=torch.ones(B, video_tokens) * sigma,
            positions=video_pos,
            context=video_context,
            enabled=True,
            context_mask=None,
            attention_mask=None,
        )
        audio_mod = Modality(
            latent=audio_latent,
            sigma=torch.tensor([sigma]),
            timesteps=torch.ones(B, audio_tokens) * sigma,
            positions=audio_pos,
            context=audio_context,
            enabled=True,
            context_mask=None,
            attention_mask=None,
        )

        perturbations = BatchedPerturbationConfig(perturbations=[PerturbationConfig(perturbations=None)])

        with torch.no_grad():
            video_denoised, audio_denoised = torch_model(video=video_mod, audio=audio_mod, perturbations=perturbations)

        # Euler step
        video_latent = euler_step(video_latent, video_denoised, sigma, sigma_next)
        audio_latent = euler_step(audio_latent, audio_denoised, sigma, sigma_next)

        logger.info(f"Step {step_idx + 1}/{num_inference_steps}: sigma {sigma:.4f} -> {sigma_next:.4f}")

    logger.info(f"Denoising complete: video={video_latent.shape}, audio={audio_latent.shape}")

    # === 6. TTNN VAE decode video ===
    video_spatial = video_latent.reshape(B, latent_frames, latent_h, latent_w, 128).permute(0, 4, 1, 2, 3)  # BCTHW
    video_pixels = tt_vae(video_spatial)
    logger.info(f"Video decoded: {video_pixels.shape}")

    # === 7. Verify shapes ===
    assert video_pixels.shape == (B, 3, num_frames, px_height, px_width), f"Video shape: {video_pixels.shape}"
    assert video_latent.shape == (B, video_tokens, 128), f"Video latent shape: {video_latent.shape}"
    assert audio_latent.shape == (B, audio_tokens, 128), f"Audio latent shape: {audio_latent.shape}"

    logger.info("PASSED: E2E AudioVideo pipeline (PyTorch DiT + TTNN VAE decode)")
