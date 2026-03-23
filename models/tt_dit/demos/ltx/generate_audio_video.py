#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
LTX-2.3 Audio+Video Generation Demo

Both video and audio denoising run on TT devices (2x4 WH LB mesh) using
the TTNN LTXAudioVideoTransformerModel. Text encoding and VAE/vocoder
decode use the reference LTX-2 pipeline (torch, CPU).

Usage:
    HF_TOKEN=... python models/tt_dit/demos/ltx/generate_audio_video.py \
        --prompt "A cat playing piano in a cozy room" \
        --output output.mp4 \
        --num_frames 33 --height 512 --width 768 --steps 30
"""

import argparse
import os
import sys
import time

import torch
from loguru import logger

sys.path.insert(0, "LTX-2/packages/ltx-core/src")
sys.path.insert(0, "LTX-2/packages/ltx-pipelines/src")

CHECKPOINT_DIR = os.environ.get("LTX_CHECKPOINT_DIR", os.path.expanduser("~/.cache/ltx-checkpoints"))


def resolve_gemma_path(gemma_path: str) -> str:
    """Resolve HF model ID to local cache directory."""
    if os.path.isdir(gemma_path):
        return gemma_path
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir = os.path.join(cache_dir, f"models--{gemma_path.replace('/', '--')}")
    if os.path.isdir(model_dir):
        snapshots = os.path.join(model_dir, "snapshots")
        if os.path.isdir(snapshots):
            versions = sorted(os.listdir(snapshots))
            if versions:
                return os.path.join(snapshots, versions[-1])
    return gemma_path


def encode_prompt(checkpoint_path: str, gemma_path: str, prompt: str):
    """Encode prompt using reference pipeline → (video_embeds, audio_embeds)."""
    from ltx_pipelines.utils.model_ledger import ModelLedger

    gemma_local = resolve_gemma_path(gemma_path)
    logger.info(f"Loading text encoder from {gemma_local}")
    ledger = ModelLedger(
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        checkpoint_path=checkpoint_path,
        gemma_root_path=gemma_local,
    )

    text_encoder = ledger.text_encoder()
    hidden_states_tuple, attention_mask = text_encoder.encode(prompt)
    logger.info(f"Gemma: {len(hidden_states_tuple)} hidden layers")
    del text_encoder

    embeddings_processor = ledger.gemma_embeddings_processor()
    output = embeddings_processor.process_hidden_states(hidden_states_tuple, attention_mask)
    video_embeds = output.video_encoding.float()  # (1, L, 4096)
    audio_embeds = output.audio_encoding.float() if output.audio_encoding is not None else None  # (1, L, 2048)
    logger.info(f"Video embeds: {video_embeds.shape}")
    if audio_embeds is not None:
        logger.info(f"Audio embeds: {audio_embeds.shape}")
    del embeddings_processor

    return video_embeds, audio_embeds


def main():
    parser = argparse.ArgumentParser(description="LTX-2.3 Audio+Video Generation (TTNN)")
    parser.add_argument("--prompt", type=str, default="A cat playing piano in a cozy room with warm lighting")
    parser.add_argument("--output", type=str, default="ltx_av_output.mp4")
    parser.add_argument("--num_frames", type=int, default=33)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--gemma_path", type=str, default="google/gemma-3-12b-it")
    parser.add_argument("--num_layers", type=int, default=48)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--audio_tokens", type=int, default=256, help="Number of audio latent tokens")
    args = parser.parse_args()

    checkpoint = args.checkpoint or os.path.join(CHECKPOINT_DIR, "ltx-2.3-22b-dev.safetensors")

    logger.info(f"Generating: '{args.prompt}'")
    logger.info(f"Output: {args.num_frames} frames @ {args.height}x{args.width}, {args.steps} steps")

    # 1. Encode text
    t0 = time.time()
    video_embeds, audio_embeds = encode_prompt(checkpoint, args.gemma_path, args.prompt)
    logger.info(f"Text encoding: {time.time()-t0:.1f}s")

    if audio_embeds is None:
        audio_embeds = torch.zeros(1, video_embeds.shape[1], 2048)
        logger.warning("No audio embeddings from encoder, using zeros")

    # 2. Load AV model on TT mesh
    import ttnn
    from models.tt_dit.models.transformers.ltx.audio_ltx import LTXAudioVideoTransformerModel
    from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
    from models.tt_dit.parallel.manager import CCLManager
    from models.tt_dit.pipelines.ltx.pipeline_ltx import compute_sigmas, euler_step
    from models.tt_dit.utils.tensor import bf16_tensor, bf16_tensor_2dshard

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

    # Load checkpoint (all keys, no filtering)
    from safetensors.torch import load_file

    t0 = time.time()
    raw = load_file(checkpoint)
    prefix = "model.diffusion_model."
    state_dict = {k[len(prefix) :]: v for k, v in raw.items() if k.startswith(prefix)}
    logger.info(f"Checkpoint: {len(state_dict)} DiT keys in {time.time()-t0:.1f}s")
    del raw

    t0 = time.time()
    model = LTXAudioVideoTransformerModel(
        num_layers=args.num_layers,
        mesh_device=mesh,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
    )
    model.load_torch_state_dict(state_dict)
    logger.info(f"AV transformer loaded in {time.time()-t0:.1f}s")
    del state_dict

    # 3. Prepare inputs
    latent_frames = (args.num_frames - 1) // 8 + 1
    latent_h = args.height // 32
    latent_w = args.width // 32
    video_N = latent_frames * latent_h * latent_w
    audio_N = args.audio_tokens

    # Check tile alignment
    sp_factor = tuple(mesh.shape)[sp_axis]
    for name, N in [("video", video_N), ("audio", audio_N)]:
        n_local = N // sp_factor
        if n_local % 32 != 0:
            logger.error(f"{name} N_local={n_local} not tile-aligned (N={N}, SP={sp_factor})")
            ttnn.close_mesh_device(mesh)
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            return

    logger.info(f"Video: {latent_frames}x{latent_h}x{latent_w} = {video_N} tokens")
    logger.info(f"Audio: {audio_N} tokens")

    # Video RoPE (SPLIT format with double-precision freq grid — matching official pipeline)
    from ltx_core.components.patchifiers import VideoLatentPatchifier, get_pixel_coords
    from ltx_core.model.transformer.rope import LTXRopeType as RefLTXRopeType
    from ltx_core.model.transformer.rope import generate_freq_grid_np
    from ltx_core.model.transformer.rope import precompute_freqs_cis as ref_precompute
    from ltx_core.types import VideoLatentShape

    fps = args.fps
    v_patchifier = VideoLatentPatchifier(patch_size=1)
    v_latent_shape = VideoLatentShape(batch=1, channels=128, frames=latent_frames, height=latent_h, width=latent_w)
    v_latent_coords = v_patchifier.get_patch_grid_bounds(output_shape=v_latent_shape, device="cpu")
    v_positions = get_pixel_coords(v_latent_coords, scale_factors=(8, 32, 32), causal_fix=True).float()
    v_positions[:, 0, ...] = v_positions[:, 0, ...] / fps

    v_cos, v_sin = ref_precompute(
        v_positions.bfloat16(),
        dim=4096,
        out_dtype=torch.float32,
        theta=10000.0,
        max_pos=[20, 2048, 2048],
        use_middle_indices_grid=True,
        num_attention_heads=32,
        rope_type=RefLTXRopeType.SPLIT,
        freq_grid_generator=generate_freq_grid_np,
    )
    tt_v_cos = bf16_tensor_2dshard(v_cos, device=mesh, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_v_sin = bf16_tensor_2dshard(v_sin, device=mesh, shard_mapping={sp_axis: 2, tp_axis: 1})

    # Audio RoPE (1D temporal, SPLIT format with double-precision)
    # Audio positions: (1, 1, N, 2) for use_middle_indices_grid
    a_pos = torch.arange(audio_N).float()
    a_positions = torch.stack([a_pos, a_pos], dim=-1).unsqueeze(0).unsqueeze(1)  # (1, 1, N, 2)
    a_cos, a_sin = ref_precompute(
        a_positions.bfloat16(),
        dim=2048,
        out_dtype=torch.float32,
        theta=10000.0,
        max_pos=[20],
        use_middle_indices_grid=True,
        num_attention_heads=32,
        rope_type=RefLTXRopeType.SPLIT,
        freq_grid_generator=generate_freq_grid_np,
    )
    tt_a_cos = bf16_tensor_2dshard(a_cos, device=mesh, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_a_sin = bf16_tensor_2dshard(a_sin, device=mesh, shard_mapping={sp_axis: 2, tp_axis: 1})

    # Push prompts to device
    tt_v_prompt = bf16_tensor(video_embeds.unsqueeze(0), device=mesh)
    tt_a_prompt = bf16_tensor(audio_embeds.unsqueeze(0), device=mesh)

    # Sigma schedule and initial noise
    sigmas = compute_sigmas(steps=args.steps, num_tokens=video_N)
    logger.info(f"Sigmas: {sigmas[0]:.4f} -> {sigmas[-1]:.4f}")

    torch.manual_seed(args.seed)
    video_latent = torch.randn(1, video_N, 128, dtype=torch.float32) * sigmas[0]
    audio_latent = torch.randn(1, audio_N, 128, dtype=torch.float32) * sigmas[0]

    # 4. Joint denoising loop
    logger.info(f"Starting AV denoising: {args.steps} steps")
    denoise_start = time.time()

    for step_idx in range(args.steps):
        sigma = sigmas[step_idx].item()
        sigma_next = sigmas[step_idx + 1].item()
        timestep = torch.tensor([sigma])

        v_out, a_out = model.inner_step(
            video_1BNI_torch=video_latent.unsqueeze(0),
            video_prompt_1BLP=tt_v_prompt,
            video_rope_cos=tt_v_cos,
            video_rope_sin=tt_v_sin,
            video_N=video_N,
            audio_1BNI_torch=audio_latent.unsqueeze(0),
            audio_prompt_1BLP=tt_a_prompt,
            audio_rope_cos=tt_a_cos,
            audio_rope_sin=tt_a_sin,
            audio_N=audio_N,
            trans_mat=None,  # Split RoPE uses elementwise rotation, no trans_mat
            timestep_torch=timestep,
        )

        # Model output is velocity; convert to x0 in bf16 (matching reference X0Model)
        v_velocity = LTXAudioVideoTransformerModel.device_to_host(v_out).squeeze(0)
        a_velocity = LTXAudioVideoTransformerModel.device_to_host(a_out).squeeze(0)
        v_denoised = (video_latent.bfloat16().float() - v_velocity.float() * sigma).bfloat16()
        a_denoised = (audio_latent.bfloat16().float() - a_velocity.float() * sigma).bfloat16()

        video_latent = euler_step(video_latent, v_denoised.float(), sigma, sigma_next).bfloat16().float()
        audio_latent = euler_step(audio_latent, a_denoised.float(), sigma, sigma_next).bfloat16().float()

        if (step_idx + 1) % 5 == 0 or step_idx == 0 or step_idx == args.steps - 1:
            elapsed = time.time() - denoise_start
            logger.info(
                f"Step {step_idx+1}/{args.steps}: sigma {sigma:.4f}->{sigma_next:.4f}, "
                f"v=[{video_latent.min():.2f},{video_latent.max():.2f}], "
                f"a=[{audio_latent.min():.2f},{audio_latent.max():.2f}], {elapsed:.1f}s"
            )

    denoise_time = time.time() - denoise_start
    logger.info(f"AV denoising: {denoise_time:.1f}s ({denoise_time/args.steps:.2f}s/step)")

    # Close TT
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    # 5. Video VAE decode
    logger.info("Decoding video...")
    from ltx_pipelines.utils.model_ledger import ModelLedger

    ledger = ModelLedger(dtype=torch.bfloat16, device=torch.device("cpu"), checkpoint_path=checkpoint)
    vae_decoder = ledger.video_decoder()

    latent_spatial = video_latent.reshape(1, latent_frames, latent_h, latent_w, 128).permute(0, 4, 1, 2, 3)
    with torch.no_grad():
        video_pixels = vae_decoder(latent_spatial.bfloat16())
    logger.info(f"Video decoded: {video_pixels.shape}")
    del vae_decoder

    # 6. Audio VAE decode + vocoder
    logger.info("Decoding audio...")
    try:
        audio_decoder = ledger.audio_decoder()
        vocoder = ledger.vocoder()

        # Reshape audio latent: (1, N, 128) -> (1, C, F, mel_bins)
        # Audio latent channels=128 for the patchified form, but AudioVAE expects (1, 8, F, 16)
        # This is a simplification — proper unpatchify needed
        # For now, reshape to closest valid shape
        audio_C = 8
        audio_mel = 16
        audio_F = audio_N  # Each token maps to one time step in audio space
        audio_spatial = audio_latent.reshape(1, audio_F, audio_C, audio_mel).permute(0, 2, 1, 3)  # (1, 8, F, 16)

        with torch.no_grad():
            audio_decoded = audio_decoder(audio_spatial.bfloat16())
            audio_waveform = vocoder(audio_decoded)
        logger.info(f"Audio decoded: {audio_waveform.shape}")
        has_audio = True
    except Exception as e:
        logger.warning(f"Audio decode failed ({e}), video only")
        audio_waveform = None
        has_audio = False

    # 7. Export video + audio combined using official pipeline's encode_video
    video_pixels = video_pixels.float().clamp(-1, 1)
    video_pixels = (video_pixels + 1) / 2
    video_uint8 = (video_pixels[0].permute(1, 2, 3, 0) * 255).to(torch.uint8)  # (F, H, W, 3)

    if has_audio and audio_waveform is not None:
        try:
            from ltx_core.types import Audio
            from ltx_pipelines.utils.media_io import encode_video

            audio_obj = Audio(waveform=audio_waveform.float().cpu(), sampling_rate=16000)
            encode_video(
                video=video_uint8,
                fps=fps,
                audio=audio_obj,
                output_path=args.output,
                video_chunks_number=1,
            )
            logger.info(f"Video+audio saved to {os.path.abspath(args.output)}")
        except Exception as e:
            logger.warning(f"Combined export failed ({e}), saving separately")
            has_audio = False

    if not has_audio or audio_waveform is None:
        # Fallback: save video only
        import imageio

        video_np = video_uint8.numpy()
        imageio.mimwrite(args.output, video_np, fps=fps, codec="libx264")
        logger.info(f"Video saved to {os.path.abspath(args.output)}")

    logger.info(f"Done! Denoise: {denoise_time:.1f}s")


if __name__ == "__main__":
    main()
