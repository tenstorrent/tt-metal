#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
LTX-2.3 End-to-End Video+Audio Generation Demo

Generates a video with synchronized audio from a text prompt.
Uses PyTorch LTXModel(AudioVideo) for DiT denoising and TTNN VAE for video decoding.

Requirements:
- HuggingFace access to Lightricks/LTX-2.3 and google/gemma-3-12b-it
- WH LB (8 chips) or equivalent
- ~88GB CPU RAM for model loading

Usage:
    export HF_TOKEN=your_token_here
    python models/tt_dit/demos/ltx/generate_video.py \
        --prompt "A cat playing piano in a cozy room" \
        --output output.mp4 \
        --num_frames 33 \
        --height 480 --width 832 \
        --steps 30

To download checkpoints first:
    huggingface-cli download Lightricks/LTX-2.3 ltx-2.3-22b-dev.safetensors --local-dir checkpoints/
    huggingface-cli download google/gemma-3-12b-it-qat-q4_0-unquantized --local-dir checkpoints/gemma/
"""

import argparse
import os
import sys
import time

import torch
from loguru import logger

# Add LTX-2 to path
sys.path.insert(0, "LTX-2/packages/ltx-core/src")
sys.path.insert(0, "LTX-2/packages/ltx-pipelines/src")

CHECKPOINT_DIR = os.environ.get("LTX_CHECKPOINT_DIR", "checkpoints")


def load_transformer(checkpoint_path: str, num_layers: int = 48):
    """Load the LTX-2 AudioVideo transformer."""
    from ltx_core.model.transformer.model import LTXModel, LTXModelType

    logger.info(f"Loading transformer from {checkpoint_path}...")
    model = LTXModel(
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

    if os.path.exists(checkpoint_path):
        from safetensors.torch import load_file

        state_dict = load_file(checkpoint_path)
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded {len(state_dict)} keys from checkpoint")
    else:
        logger.warning(f"Checkpoint not found: {checkpoint_path}, using random weights")

    model.eval()
    model = model.to(torch.bfloat16)
    return model


def encode_text(prompt: str, gemma_path: str = None):
    """Encode text prompt using Gemma or fallback to random embeddings."""
    try:
        from models.tt_dit.encoders.gemma.encoder_pair import GemmaTokenizerEncoderPair

        encoder = GemmaTokenizerEncoderPair(
            checkpoint=gemma_path or "google/gemma-3-12b-it",
            sequence_length=256,
            embedding_dim=4096,
        )
        video_context = encoder.encode([prompt])
        audio_context = video_context[:, :, :2048]  # Project to audio dim
        return video_context, audio_context
    except Exception as e:
        logger.warning(f"Text encoder failed ({e}), using random embeddings")
        return torch.randn(1, 128, 4096), torch.randn(1, 128, 2048)


def main():
    parser = argparse.ArgumentParser(description="LTX-2.3 Video+Audio Generation")
    parser.add_argument("--prompt", type=str, default="A cat playing piano in a cozy room with warm lighting")
    parser.add_argument("--output", type=str, default="ltx_output.mp4")
    parser.add_argument("--num_frames", type=int, default=33)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--gemma_path", type=str, default=None)
    parser.add_argument(
        "--num_layers", type=int, default=1, help="Number of transformer layers (1 for test, 48 for full)"
    )
    args = parser.parse_args()

    logger.info(f"Generating: '{args.prompt}'")
    logger.info(f"Output: {args.num_frames} frames @ {args.height}x{args.width}, {args.steps} steps")

    # Import ttnn and setup device
    import ttnn
    from models.tt_dit.models.vae.ltx.vae_ltx import LTXVideoDecoder
    from models.tt_dit.pipelines.ltx.pipeline_ltx import compute_sigmas, euler_step

    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(2, 4))
    logger.info(f"Opened 2x4 mesh ({mesh.get_num_devices()} devices)")

    # 1. Load transformer
    checkpoint = args.checkpoint or os.path.join(CHECKPOINT_DIR, "ltx-2.3-22b-dev.safetensors")
    transformer = load_transformer(checkpoint, num_layers=args.num_layers)

    # 2. Encode text
    video_context, audio_context = encode_text(args.prompt, args.gemma_path)

    # 3. Compute dimensions
    latent_frames = (args.num_frames - 1) // 8 + 1
    latent_h = args.height // 32
    latent_w = args.width // 32
    video_tokens = latent_frames * latent_h * latent_w
    audio_tokens = 64  # Simplified audio token count

    logger.info(
        f"Latent: {latent_frames}x{latent_h}x{latent_w} = {video_tokens} video tokens, {audio_tokens} audio tokens"
    )

    # 4. Sigma schedule
    sigmas = compute_sigmas(steps=args.steps, num_tokens=video_tokens)

    # 5. Initial noise
    torch.manual_seed(args.seed)
    video_latent = torch.randn(1, video_tokens, 128) * sigmas[0]
    audio_latent = torch.randn(1, audio_tokens, 128) * sigmas[0]

    # 6. Positions

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
    )
    audio_pos = torch.arange(audio_tokens).float().unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 2)

    # 7. Denoising loop
    from ltx_core.guidance.perturbations import BatchedPerturbationConfig, PerturbationConfig
    from ltx_core.model.transformer.model import Modality

    logger.info(f"Starting denoising: {args.steps} steps, guidance_scale={args.guidance_scale}")
    start_time = time.time()

    for step_idx in range(args.steps):
        sigma = sigmas[step_idx].item()
        sigma_next = sigmas[step_idx + 1].item()

        video_mod = Modality(
            latent=video_latent.bfloat16(),
            sigma=torch.tensor([sigma]),
            timesteps=torch.ones(1, video_tokens) * sigma,
            positions=video_pos,
            context=video_context.bfloat16(),
            enabled=True,
            context_mask=None,
            attention_mask=None,
        )
        audio_mod = Modality(
            latent=audio_latent.bfloat16(),
            sigma=torch.tensor([sigma]),
            timesteps=torch.ones(1, audio_tokens) * sigma,
            positions=audio_pos,
            context=audio_context.bfloat16(),
            enabled=True,
            context_mask=None,
            attention_mask=None,
        )

        perturbations = BatchedPerturbationConfig(perturbations=[PerturbationConfig(perturbations=None)])

        with torch.no_grad():
            video_denoised, audio_denoised = transformer(video=video_mod, audio=audio_mod, perturbations=perturbations)

        video_latent = euler_step(video_latent, video_denoised.float(), sigma, sigma_next)
        audio_latent = euler_step(audio_latent, audio_denoised.float(), sigma, sigma_next)

        if (step_idx + 1) % 5 == 0 or step_idx == 0 or step_idx == args.steps - 1:
            elapsed = time.time() - start_time
            logger.info(
                f"Step {step_idx + 1}/{args.steps}: sigma {sigma:.4f} -> {sigma_next:.4f}, " f"elapsed {elapsed:.1f}s"
            )

    denoise_time = time.time() - start_time
    logger.info(f"Denoising complete in {denoise_time:.1f}s")

    # 8. TTNN VAE decode video
    logger.info("Decoding video with TTNN VAE decoder...")
    from ltx_core.model.video_vae.enums import NormLayerType
    from ltx_core.model.video_vae.video_vae import VideoDecoder

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
    # TODO: Load real VAE weights from checkpoint
    vae_state = torch_vae.state_dict()
    vae_state["per_channel_statistics.mean-of-means"] = torch.zeros(128)
    vae_state["per_channel_statistics.std-of-means"] = torch.ones(128)
    torch_vae.load_state_dict(vae_state)

    tt_vae = LTXVideoDecoder(decoder_blocks=decoder_blocks, mesh_device=mesh)
    tt_vae.load_torch_state_dict(torch_vae.state_dict())

    video_spatial = video_latent.reshape(1, latent_frames, latent_h, latent_w, 128).permute(0, 4, 1, 2, 3)
    decode_start = time.time()
    video_pixels = tt_vae(video_spatial)
    decode_time = time.time() - decode_start
    logger.info(f"Video decoded: {video_pixels.shape} in {decode_time:.1f}s")

    # 9. Export video
    # Normalize to [0, 1] for video export
    video_pixels = video_pixels.float().clamp(-1, 1)
    video_pixels = (video_pixels + 1) / 2  # [-1,1] -> [0,1]
    video_np = (video_pixels[0].permute(1, 2, 3, 0).numpy() * 255).astype("uint8")  # (F, H, W, 3)

    try:
        import imageio

        imageio.mimwrite(args.output, video_np, fps=16, codec="libx264")
        logger.info(f"Video saved to {args.output}")
    except ImportError:
        # Fallback: save as numpy
        import numpy as np

        np.save(args.output.replace(".mp4", ".npy"), video_np)
        logger.info(f"Video saved as numpy array: {args.output.replace('.mp4', '.npy')}")

    # 10. Audio placeholder
    logger.info(f"Audio latent shape: {audio_latent.shape} (decoding requires AudioVAE + vocoder)")

    total_time = time.time() - start_time
    logger.info(f"Total generation time: {total_time:.1f}s")
    logger.info(f"  Denoising: {denoise_time:.1f}s")
    logger.info(f"  VAE decode: {decode_time:.1f}s")

    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
