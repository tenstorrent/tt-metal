#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
LTX-2.3 End-to-End Video Generation Demo

Denoising runs entirely on TT devices (2x4 WH LB mesh) using the TTNN
LTXTransformerModel. Text encoding and VAE decode use the reference LTX-2
pipeline (torch, on CPU).

Requirements:
- HuggingFace access to Lightricks/LTX-2.3 and google/gemma-3-12b-it
- WH LB (8 chips)
- ~88GB CPU RAM for model loading

Usage:
    python models/tt_dit/demos/ltx/generate_video.py \
        --prompt "A cat playing piano in a cozy room" \
        --output output.mp4 \
        --num_frames 33 \
        --height 512 --width 768 \
        --steps 30 \
        --checkpoint ~/.cache/ltx-checkpoints/ltx-2.3-22b-dev.safetensors \
        --gemma_path google/gemma-3-12b-it
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

# Monkey-patch CUDA for CPU-only text encoding (no NVIDIA GPU on TT machines)
torch.cuda.is_available = lambda: False
torch.cuda.synchronize = lambda *a, **kw: None
torch.cuda.empty_cache = lambda: None

CHECKPOINT_DIR = os.environ.get(
    "LTX_CHECKPOINT_DIR",
    os.path.expanduser("~/.cache/ltx-checkpoints"),
)

# Top-level 22B keys to skip (audio-only or not yet implemented in video-only TTNN model)
_SKIP_PREFIXES = (
    "audio_adaln_single",
    "audio_embeddings_connector",
    "audio_patchify_proj",
    "audio_proj_out",
    "audio_prompt_adaln_single",
    "audio_scale_shift_table",
    "av_ca_",
    "video_embeddings_connector",
)

# Per-block keys to skip (audio path, cross-modal)
_SKIP_BLOCK_SUBKEYS = (
    "audio_attn1",
    "audio_attn2",
    "audio_ff",
    "audio_norm",
    "audio_scale_shift_table",
    "audio_prompt_scale_shift_table",
    "audio_to_video_attn",
    "video_to_audio_attn",
    "scale_shift_table_a2v_ca_audio",
    "scale_shift_table_a2v_ca_video",
)


def load_22b_video_state_dict(checkpoint_path: str) -> dict[str, torch.Tensor]:
    """Load the LTX-2.3 22B safetensors checkpoint, extracting video-only DiT keys.

    The checkpoint uses ComfyUI naming with 'model.diffusion_model.' prefix.
    We strip the prefix and filter out audio/connector/prompt keys that our
    video-only TTNN model doesn't support yet.
    """
    from safetensors.torch import load_file

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    raw = load_file(checkpoint_path)
    logger.info(f"Loaded {len(raw)} total keys")

    prefix = "model.diffusion_model."
    state_dict = {}
    skipped = 0

    for key, tensor in raw.items():
        if not key.startswith(prefix):
            skipped += 1
            continue

        stripped = key[len(prefix) :]

        # Skip top-level audio/connector keys
        if any(stripped.startswith(p) for p in _SKIP_PREFIXES):
            skipped += 1
            continue

        # Skip per-block audio/cross-modal keys
        if "transformer_blocks." in stripped:
            parts = stripped.split(".", 2)
            if len(parts) >= 3:
                block_subkey = parts[2]
                if any(block_subkey.startswith(p) for p in _SKIP_BLOCK_SUBKEYS):
                    skipped += 1
                    continue

        state_dict[stripped] = tensor

    logger.info(f"Extracted {len(state_dict)} video DiT keys (skipped {skipped})")
    return state_dict


def _resolve_gemma_path(gemma_path: str) -> str:
    """Resolve Gemma model path to a local directory.

    If gemma_path is a HuggingFace model ID (e.g., 'google/gemma-3-12b-it'),
    resolves to the local HuggingFace cache snapshot directory.
    """
    if os.path.isdir(gemma_path):
        return gemma_path

    # Try HuggingFace cache
    try:
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(gemma_path, local_files_only=True)
        logger.info(f"Resolved Gemma path to: {local_dir}")
        return local_dir
    except Exception:
        pass

    # Manual cache lookup
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir = os.path.join(cache_dir, f"models--{gemma_path.replace('/', '--')}")
    if os.path.isdir(model_dir):
        snapshots = os.path.join(model_dir, "snapshots")
        if os.path.isdir(snapshots):
            versions = sorted(os.listdir(snapshots))
            if versions:
                resolved = os.path.join(snapshots, versions[-1])
                logger.info(f"Resolved Gemma path to: {resolved}")
                return resolved

    return gemma_path


def encode_prompt_reference(checkpoint_path: str, gemma_path: str, prompt: str):
    """Encode prompt and negative prompt using the reference pipeline.

    Returns (video_embeds, negative_video_embeds) both as float tensors.
    Uses the official encode_prompts helper which encodes both prompt and
    negative prompt through Gemma → EmbeddingsProcessor.
    """
    from ltx_pipelines.utils.helpers import encode_prompts
    from ltx_pipelines.utils.model_ledger import ModelLedger

    gemma_local = _resolve_gemma_path(gemma_path)
    logger.info("Loading text encoder and embeddings processor...")
    ledger = ModelLedger(
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        checkpoint_path=checkpoint_path,
        gemma_root_path=gemma_local,
    )

    # Encode both prompt and negative prompt ("") through the full pipeline
    ctx_p, ctx_n = encode_prompts([prompt, ""], ledger)
    video_embeds = ctx_p.video_encoding  # (1, 1024, 4096)
    neg_video_embeds = ctx_n.video_encoding  # (1, 1024, 4096) — NOT zeros!
    logger.info(f"Video embeddings: {video_embeds.shape}, neg: {neg_video_embeds.shape}")

    return video_embeds.float(), neg_video_embeds.float()


def main():
    parser = argparse.ArgumentParser(description="LTX-2.3 Video Generation (TTNN)")
    parser.add_argument("--prompt", type=str, default="A cat playing piano in a cozy room with warm lighting")
    parser.add_argument("--output", type=str, default="ltx_output.mp4")
    parser.add_argument("--num_frames", type=int, default=121, help="Reference default: 121 (5s at 24fps)")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=10, help="Reference default seed")
    parser.add_argument("--guidance_scale", type=float, default=3.0, help="Reference default: 3.0")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--gemma_path", type=str, default="google/gemma-3-12b-it")
    parser.add_argument(
        "--num_layers", type=int, default=48, help="Number of transformer layers (1 for test, 48 for full)"
    )
    args = parser.parse_args()

    logger.info(f"Generating: '{args.prompt}'")
    logger.info(f"Output: {args.num_frames} frames @ {args.height}x{args.width}, {args.steps} steps")

    checkpoint = args.checkpoint or os.path.join(CHECKPOINT_DIR, "ltx-2.3-22b-dev.safetensors")

    # 1. Encode prompt AND negative prompt using reference pipeline
    t0 = time.time()
    prompt_embeds, neg_prompt_embeds = encode_prompt_reference(checkpoint, args.gemma_path, args.prompt)
    logger.info(f"Text encoding: {time.time()-t0:.1f}s, shape {prompt_embeds.shape}")

    # 2. Open TT mesh and load TTNN transformer
    import ttnn
    from models.tt_dit.models.transformers.ltx.transformer_ltx import LTXTransformerModel
    from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
    from models.tt_dit.parallel.manager import CCLManager
    from models.tt_dit.pipelines.ltx.pipeline_ltx import LTXPipeline, compute_sigmas, euler_step
    from models.tt_dit.utils.tensor import bf16_tensor

    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
    )
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(2, 4))
    logger.info(f"Opened 2x4 mesh ({mesh.get_num_devices()} devices)")

    sp_axis, tp_axis = 0, 1
    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        sequence_parallel=ParallelFactor(factor=tuple(mesh.shape)[sp_axis], mesh_axis=sp_axis),
        tensor_parallel=ParallelFactor(factor=tuple(mesh.shape)[tp_axis], mesh_axis=tp_axis),
    )
    ccl_manager = CCLManager(mesh, topology=ttnn.Topology.Linear)

    pipeline = LTXPipeline(
        mesh_device=mesh,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
        num_layers=args.num_layers,
        cross_attention_dim=4096,
    )

    t0 = time.time()
    video_state_dict = load_22b_video_state_dict(checkpoint)
    pipeline.load_transformer(video_state_dict)
    logger.info(f"Transformer loaded to device in {time.time()-t0:.1f}s")
    del video_state_dict  # Free CPU memory

    # 3. Prepare inputs
    latent_frames = (args.num_frames - 1) // 8 + 1
    latent_h = args.height // 32
    latent_w = args.width // 32
    num_tokens = latent_frames * latent_h * latent_w

    logger.info(f"Latent: {latent_frames}x{latent_h}x{latent_w} = {num_tokens} tokens")

    # Check tile alignment
    sp_factor = tuple(mesh.shape)[sp_axis]
    n_local = num_tokens // sp_factor
    if n_local % 32 != 0:
        logger.error(
            f"N_local={n_local} not divisible by tile height 32. "
            f"Choose dimensions where (latent_frames*latent_h*latent_w/{sp_factor}) % 32 == 0"
        )
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        return

    # Prepare RoPE and prompt on device
    from models.tt_dit.models.transformers.ltx.rope_ltx import precompute_freqs_cis
    from models.tt_dit.utils.mochi import get_rot_transformation_mat
    from models.tt_dit.utils.tensor import bf16_tensor_2dshard

    # RoPE positions: use center of each latent cell in pixel space
    # Reference uses (start, end) pairs with use_middle_indices_grid=True → takes average
    # Each latent cell spans: T_scale=8, H_scale=32, W_scale=32 pixels
    t_ids = torch.arange(latent_frames)
    h_ids = torch.arange(latent_h)
    w_ids = torch.arange(latent_w)
    grid_t, grid_h, grid_w = torch.meshgrid(t_ids, h_ids, w_ids, indexing="ij")

    fps = 24.0
    # Pixel-space start/end for each cell, then take midpoint
    t_start = (grid_t.float() * 8 + 1 - 8).clamp(min=0) / fps
    t_end = ((grid_t.float() + 1) * 8 + 1 - 8).clamp(min=0) / fps
    h_start = grid_h.float() * 32
    h_end = (grid_h.float() + 1) * 32
    w_start = grid_w.float() * 32
    w_end = (grid_w.float() + 1) * 32

    t_mid = (t_start + t_end) / 2
    h_mid = (h_start + h_end) / 2
    w_mid = (w_start + w_end) / 2

    indices_grid = torch.stack([t_mid.flatten(), h_mid.flatten(), w_mid.flatten()], dim=-1).float().unsqueeze(0)

    cos_freq, sin_freq = precompute_freqs_cis(
        indices_grid,
        dim=4096,
        out_dtype=torch.float32,
        theta=10000.0,
        max_pos=[20, 2048, 2048],
        num_attention_heads=32,
    )

    head_dim = 128
    cos_heads = cos_freq.reshape(1, num_tokens, 32, head_dim).permute(0, 2, 1, 3)
    sin_heads = sin_freq.reshape(1, num_tokens, 32, head_dim).permute(0, 2, 1, 3)

    tt_cos = bf16_tensor_2dshard(cos_heads, device=mesh, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_sin = bf16_tensor_2dshard(sin_heads, device=mesh, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_trans_mat = bf16_tensor(get_rot_transformation_mat(), device=mesh)

    # Push prompt to device (1, B, L, D)
    tt_prompt = bf16_tensor(prompt_embeds.unsqueeze(0), device=mesh)

    # Sigma schedule
    sigmas = compute_sigmas(steps=args.steps, num_tokens=num_tokens)
    logger.info(f"Sigmas: {sigmas[0]:.4f} -> {sigmas[-1]:.4f}")

    # Initial noise
    # Generate noise in bf16 to match reference (bf16 randn produces different values than fp32)
    torch.manual_seed(args.seed)
    latent = torch.randn(1, num_tokens, 128, dtype=torch.bfloat16).float() * sigmas[0]

    # Negative prompt for CFG
    do_cfg = args.guidance_scale > 1.0
    if do_cfg:
        tt_neg_prompt = bf16_tensor(neg_prompt_embeds.unsqueeze(0), device=mesh)

    # 4. Denoising loop (forward pass on TT devices, euler step on host)
    logger.info(f"Starting denoising: {args.steps} steps, guidance_scale={args.guidance_scale}")
    denoise_start = time.time()

    for step_idx in range(args.steps):
        sigma = sigmas[step_idx].item()
        sigma_next = sigmas[step_idx + 1].item()

        spatial_torch = latent.unsqueeze(0)  # (1, B, N, C)
        timestep_torch = torch.tensor([sigma])

        # Conditioned forward pass
        tt_denoised = pipeline.transformer.inner_step(
            spatial_1BNI_torch=spatial_torch,
            prompt_1BLP=tt_prompt,
            rope_cos=tt_cos,
            rope_sin=tt_sin,
            trans_mat=tt_trans_mat,
            N=num_tokens,
            timestep_torch=timestep_torch,
        )
        # Model output is velocity; convert to x0: denoised = sample - velocity * sigma
        velocity = LTXTransformerModel.device_to_host(tt_denoised).squeeze(0)
        denoised = latent.float() - velocity.float() * sigma

        # CFG with variance rescaling (matching reference MultiModalGuider.calculate)
        if do_cfg:
            tt_uncond = pipeline.transformer.inner_step(
                spatial_1BNI_torch=spatial_torch,
                prompt_1BLP=tt_neg_prompt,
                rope_cos=tt_cos,
                rope_sin=tt_sin,
                trans_mat=tt_trans_mat,
                N=num_tokens,
                timestep_torch=timestep_torch,
            )
            uncond_velocity = LTXTransformerModel.device_to_host(tt_uncond).squeeze(0)
            uncond = latent.float() - uncond_velocity.float() * sigma
            # Reference formula: pred = cond + (scale - 1) * (cond - uncond)
            pred = denoised + (args.guidance_scale - 1) * (denoised - uncond)
            # Variance rescaling to prevent oversaturation
            rescale = 0.7
            factor = rescale * (denoised.std() / pred.std()) + (1 - rescale)
            denoised = pred * factor

        latent = euler_step(latent, denoised, sigma, sigma_next)

        if (step_idx + 1) % 5 == 0 or step_idx == 0 or step_idx == args.steps - 1:
            elapsed = time.time() - denoise_start
            logger.info(
                f"Step {step_idx+1}/{args.steps}: sigma {sigma:.4f} -> {sigma_next:.4f}, "
                f"range [{latent.min():.3f}, {latent.max():.3f}], {elapsed:.1f}s"
            )

    denoise_time = time.time() - denoise_start
    logger.info(f"Denoising complete in {denoise_time:.1f}s ({denoise_time/args.steps:.2f}s/step)")

    # Close TT devices
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    # 5. VAE decode (torch, on CPU)
    logger.info("Decoding latent to video with reference VAE...")
    from ltx_pipelines.utils.model_ledger import ModelLedger

    ledger = ModelLedger(dtype=torch.bfloat16, device=torch.device("cpu"), checkpoint_path=checkpoint)
    vae_decoder = ledger.video_decoder()

    latent_spatial = latent.reshape(1, latent_frames, latent_h, latent_w, 128).permute(0, 4, 1, 2, 3)  # BCTHW

    decode_start = time.time()
    with torch.no_grad():
        video_pixels = vae_decoder(latent_spatial.bfloat16())
    decode_time = time.time() - decode_start
    logger.info(f"VAE decode: {video_pixels.shape} in {decode_time:.1f}s")
    del vae_decoder

    # 6. Export video
    video_pixels = video_pixels.float().clamp(-1, 1)
    video_pixels = (video_pixels + 1) / 2  # [-1,1] -> [0,1]
    video_np = (video_pixels[0].permute(1, 2, 3, 0).numpy() * 255).astype("uint8")

    try:
        import imageio

        imageio.mimwrite(args.output, video_np, fps=24, codec="libx264")
        logger.info(f"Video saved to {args.output}")
    except ImportError:
        import numpy as np

        np.save(args.output.replace(".mp4", ".npy"), video_np)
        logger.info(f"Video saved as numpy: {args.output.replace('.mp4', '.npy')}")

    total_time = denoise_time + decode_time
    logger.info(f"Total: {total_time:.1f}s (denoise: {denoise_time:.1f}s, VAE: {decode_time:.1f}s)")


if __name__ == "__main__":
    main()
