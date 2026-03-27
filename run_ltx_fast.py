#!/usr/bin/env python3
"""
LTX-2.3 Fast (distilled, 2-stage) pipeline runner.

Matches the reference distilled.py flow:
- Stage 1: Half-resolution generation (8 denoising steps, no guidance)
- Spatial upsampling via learned upsampler (2x)
- Stage 2: Full-resolution refinement (3 denoising steps, no guidance)
- Total: ~11 steps (vs 30 for Pro)

Usage:
    python run_ltx_fast.py --prompt "A cat playing piano" --output output.mp4
    python run_ltx_fast.py --prompt "..." --output out.mp4 --seed 10
"""

import argparse
import gc
import json
import os
import sys
import time
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "LTX-2/packages/ltx-core/src")
sys.path.insert(0, "LTX-2/packages/ltx-pipelines/src")

import torch
from loguru import logger
from safetensors.torch import load_file

torch.cuda.synchronize = lambda *a, **kw: None

import ttnn
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.ltx.pipeline_ltx import LTXPipeline, euler_step

# Reference sigma schedules from constants.py
DISTILLED_SIGMA_VALUES = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
STAGE_2_DISTILLED_SIGMA_VALUES = [0.909375, 0.725, 0.421875, 0.0]

DEFAULT_CKPT = "/localdev/kevinmi/.cache/ltx-checkpoints/ltx-2-19b-distilled.safetensors"
DEFAULT_UPSAMPLER = "/localdev/kevinmi/.cache/ltx-checkpoints/ltx-2-spatial-upscaler-x2-1.0.safetensors"
DEFAULT_GEMMA = "/localdev/kevinmi/.cache/gemma-3-12b-it-qat-q4_0-unquantized"


def decode_audio(audio_latent, checkpoint_path, num_frames, fps=24.0):
    """Decode audio using reference audio VAE + vocoder."""
    try:
        from ltx_core.model.audio_vae.audio_vae import decode_audio as vae_decode_audio
        from ltx_core.types import Audio
        from ltx_pipelines.utils.model_ledger import ModelLedger

        ledger = ModelLedger(dtype=torch.bfloat16, device=torch.device("cpu"), checkpoint_path=checkpoint_path)
        audio_decoder = ledger.audio_decoder()
        vocoder = ledger.vocoder()

        audio_N = audio_latent.shape[1]
        audio_spatial = audio_latent.reshape(1, audio_N, 8, 16).permute(0, 2, 1, 3).bfloat16()

        with torch.no_grad():
            audio_obj = vae_decode_audio(audio_spatial, audio_decoder, vocoder)

        video_duration = num_frames / fps
        target_samples = int(video_duration * audio_obj.sampling_rate)
        if audio_obj.waveform.shape[-1] > target_samples:
            audio_obj = Audio(waveform=audio_obj.waveform[..., :target_samples], sampling_rate=audio_obj.sampling_rate)

        logger.info(
            f"Audio: {audio_obj.waveform.shape[-1] / audio_obj.sampling_rate:.2f}s @ {audio_obj.sampling_rate}Hz"
        )
        return audio_obj
    except Exception as e:
        logger.error(f"Audio decode failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def export_video(video_pixels, output_path, fps=24, audio=None):
    """Export video+audio to MP4."""
    import av
    import numpy as np

    frames = (((video_pixels[0] + 1.0) / 2.0).clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    frames = frames.permute(1, 2, 3, 0).cpu().numpy()

    container = av.open(output_path, mode="w")
    stream = container.add_stream("libx264", rate=int(fps))
    stream.width = frames.shape[2]
    stream.height = frames.shape[1]
    stream.pix_fmt = "yuv420p"

    audio_stream = None
    if audio is not None:
        audio_stream = container.add_stream("aac", rate=audio.sampling_rate)
        audio_stream.codec_context.sample_rate = audio.sampling_rate
        audio_stream.codec_context.layout = "stereo"
        audio_stream.codec_context.time_base = Fraction(1, audio.sampling_rate)

    for frame_array in frames:
        frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)

    if audio is not None and audio_stream is not None:
        samples = audio.waveform
        if samples.ndim == 1:
            samples = samples[:, None]
        if samples.shape[1] != 2 and samples.shape[0] == 2:
            samples = samples.T
        if samples.shape[1] != 2:
            samples = samples[:, :1].repeat(1, 2)
        if samples.dtype != torch.int16:
            samples = torch.clip(samples, -1.0, 1.0)
            samples = (samples * 32767.0).to(torch.int16)
        frame_in = av.AudioFrame.from_ndarray(
            samples.contiguous().reshape(1, -1).cpu().numpy(),
            format="s16",
            layout="stereo",
        )
        frame_in.sample_rate = audio.sampling_rate
        cc = audio_stream.codec_context
        resampler = av.audio.resampler.AudioResampler(
            format=cc.format or "fltp",
            layout=cc.layout or "stereo",
            rate=cc.sample_rate or audio.sampling_rate,
        )
        for resampled in resampler.resample(frame_in):
            for packet in audio_stream.encode(resampled):
                container.mux(packet)
        for packet in audio_stream.encode():
            container.mux(packet)

    container.close()
    logger.info(f"Saved: {output_path} ({frames.shape[0]}f @ {fps}fps)")


def upsample_latent_reference(video_latent, upsampler_path, checkpoint_path):
    """Upsample video latent 2x spatially using reference upsampler on CPU."""
    from ltx_pipelines.utils.model_ledger import ModelLedger

    ledger = ModelLedger(
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        checkpoint_path=checkpoint_path,
        spatial_upsampler_path=upsampler_path,
    )
    upsampler = ledger.spatial_latent_upsampler()

    with torch.no_grad():
        upsampled = upsampler(video_latent.bfloat16())

    logger.info(f"Upsampled: {video_latent.shape} → {upsampled.shape}")
    del upsampler, ledger
    gc.collect()
    return upsampled.float()


def run_denoise_no_guidance(
    pipeline,
    v_embeds,
    a_embeds,
    num_frames,
    height,
    width,
    sigma_values,
    seed,
    initial_video_latent=None,
    initial_audio_latent=None,
):
    """Simple denoising loop with no guidance (matching distilled pipeline).

    The distilled model is trained to denoise without CFG/STG, so we just run
    a single forward pass per step.
    """
    from models.tt_dit.models.transformers.ltx.ltx_transformer import LTXTransformerModel
    from models.tt_dit.utils.ltx import AudioLatentShape, VideoPixelShape
    from models.tt_dit.utils.tensor import bf16_tensor

    B = 1
    latent_frames = (num_frames - 1) // 8 + 1
    latent_h, latent_w = height // 32, width // 32
    video_N = latent_frames * latent_h * latent_w

    vps = VideoPixelShape(batch=B, frames=num_frames, height=height, width=width, fps=24)
    als = AudioLatentShape.from_video_pixel_shape(vps)
    audio_N_real = als.frames
    sp_factor = pipeline.parallel_config.sequence_parallel.factor
    audio_N = ((audio_N_real + 32 * sp_factor - 1) // (32 * sp_factor)) * (32 * sp_factor)

    v_cos, v_sin = pipeline._prepare_rope(latent_frames, latent_h, latent_w)
    a_cos, a_sin = pipeline._prepare_audio_rope(audio_N, audio_N_real)
    tt_attn_mask, tt_pad_mask = pipeline._prepare_audio_masks(audio_N, audio_N_real)

    tt_vp = pipeline._prepare_prompt(v_embeds)
    tt_ap = bf16_tensor(a_embeds.unsqueeze(0), device=pipeline.mesh_device)

    sigmas = torch.tensor(sigma_values, dtype=torch.float32)

    # Initial noise or continue from provided latent
    if initial_video_latent is not None:
        video_lat = initial_video_latent.float()
        # Add noise at first sigma level
        torch.manual_seed(seed)
        noise_v = torch.randn_like(video_lat)
        video_lat = video_lat * (1 - sigmas[0]) + noise_v * sigmas[0]
    else:
        torch.manual_seed(seed)
        video_lat = torch.randn(B, video_N, pipeline.in_channels, dtype=torch.bfloat16).float() * sigmas[0]

    if initial_audio_latent is not None:
        audio_lat = torch.zeros(B, audio_N, pipeline.in_channels)
        audio_lat[:, :audio_N_real, :] = initial_audio_latent[:, :audio_N_real, :].float()
        torch.manual_seed(seed + 1)
        noise_a = torch.randn_like(audio_lat)
        audio_lat = audio_lat * (1 - sigmas[0]) + noise_a * sigmas[0]
    else:
        torch.manual_seed(seed)
        # Skip video noise draw to keep seeds aligned
        _ = torch.randn(B, video_N, pipeline.in_channels, dtype=torch.bfloat16)
        audio_lat_real = torch.randn(B, audio_N_real, pipeline.in_channels, dtype=torch.bfloat16).float() * sigmas[0]
        audio_lat = torch.zeros(B, audio_N, pipeline.in_channels)
        audio_lat[:, :audio_N_real, :] = audio_lat_real

    num_steps = len(sigma_values) - 1
    for step_idx in range(num_steps):
        sigma = sigmas[step_idx].item()
        sigma_next = sigmas[step_idx + 1].item()

        # Single forward pass — no guidance for distilled model
        v_out, a_out = pipeline.transformer.inner_step(
            video_1BNI_torch=video_lat.unsqueeze(0),
            video_prompt_1BLP=tt_vp,
            video_rope_cos=v_cos,
            video_rope_sin=v_sin,
            video_N=video_N,
            audio_1BNI_torch=audio_lat.unsqueeze(0),
            audio_prompt_1BLP=tt_ap,
            audio_rope_cos=a_cos,
            audio_rope_sin=a_sin,
            audio_N=audio_N,
            trans_mat=None,
            timestep_torch=torch.tensor([sigma]),
            skip_cross_attn=False,
            skip_self_attn_blocks=None,
            audio_attn_mask=tt_attn_mask,
            audio_padding_mask=tt_pad_mask,
        )
        v_vel = LTXTransformerModel.device_to_host(v_out).squeeze(0)
        a_vel = LTXTransformerModel.device_to_host(a_out).squeeze(0)

        # Convert velocity to denoised: denoised = x - velocity * sigma
        v_den = (video_lat.bfloat16().float() - v_vel.float() * sigma).bfloat16()
        a_den = (audio_lat.bfloat16().float() - a_vel.float() * sigma).bfloat16()

        # Euler step
        if sigma_next == 0.0:
            video_lat = v_den.float()
            a_new = a_den.float()
        else:
            video_lat = euler_step(video_lat, v_den.float(), sigma, sigma_next).bfloat16().float()
            a_new = euler_step(audio_lat, a_den.float(), sigma, sigma_next).bfloat16().float()

        audio_lat = torch.zeros_like(audio_lat)
        audio_lat[:, :audio_N_real, :] = a_new[:, :audio_N_real, :]

        logger.info(f"  Step {step_idx + 1}/{num_steps}: σ {sigma:.4f} → {sigma_next:.4f}")

    return video_lat, audio_lat[:, :audio_N_real, :]


def main():
    parser = argparse.ArgumentParser(description="LTX-2.3 Fast (distilled, 2-stage) pipeline")
    parser.add_argument("--prompt", required=True, help="Text prompt")
    parser.add_argument("--output", required=True, help="Output MP4 path")
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT, help="Distilled checkpoint")
    parser.add_argument("--upsampler", default=DEFAULT_UPSAMPLER, help="Spatial upsampler checkpoint")
    parser.add_argument("--gemma", default=DEFAULT_GEMMA, help="Gemma model path")
    parser.add_argument("--seed", type=int, default=10, help="Random seed")
    parser.add_argument("--frames", type=int, default=121, help="Number of frames")
    parser.add_argument("--height", type=int, default=512, help="Final video height (must be divisible by 64)")
    parser.add_argument("--width", type=int, default=768, help="Final video width (must be divisible by 64)")
    parser.add_argument("--fps", type=int, default=24, help="Frame rate")
    args = parser.parse_args()

    assert os.path.exists(args.checkpoint), f"Distilled checkpoint not found: {args.checkpoint}"
    assert os.path.exists(args.upsampler), f"Upsampler not found: {args.upsampler}"
    assert os.path.isdir(args.gemma), f"Gemma not found: {args.gemma}"
    assert args.height % 64 == 0, f"Height must be divisible by 64 (got {args.height})"
    assert args.width % 64 == 0, f"Width must be divisible by 64 (got {args.width})"

    from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT

    s1_height = args.height // 2
    s1_width = args.width // 2

    logger.info("=" * 60)
    logger.info("LTX-2.3 Fast (distilled, 2-stage)")
    logger.info("=" * 60)
    logger.info(f"Prompt: {args.prompt[:80]}...")
    logger.info(f"Stage 1: {args.frames}f @ {s1_height}x{s1_width}, {len(DISTILLED_SIGMA_VALUES)-1} steps")
    logger.info(f"Stage 2: {args.frames}f @ {args.height}x{args.width}, {len(STAGE_2_DISTILLED_SIGMA_VALUES)-1} steps")
    logger.info(f"Output: {args.output}")

    total_t0 = time.time()

    # === Text encoding ===
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(2, 4), l1_small_size=16384)

    sp_axis, tp_axis = 0, 1
    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        sequence_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[sp_axis], mesh_axis=sp_axis),
        tensor_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[tp_axis], mesh_axis=tp_axis),
    )
    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    pipeline = LTXPipeline(
        mesh_device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
        mode="av",
    )

    t0 = time.time()
    results = pipeline.encode_prompts_reference([args.prompt], args.checkpoint, args.gemma)
    logger.info(f"Encoding: {time.time() - t0:.1f}s")

    v_embeds = results[0].video_encoding.float()
    a_embeds = results[0].audio_encoding.float()

    # === Load transformer (distilled) ===
    raw = load_file(args.checkpoint)
    prefix = "model.diffusion_model."
    transformer_sd = {k[len(prefix) :]: v for k, v in raw.items() if k.startswith(prefix)}

    with open(args.checkpoint, "rb") as f:
        header_size = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_size))
    vae_cfg = json.loads(header.get("__metadata__", {}).get("config", "{}")).get("vae", {})
    pipeline._vae_checkpoint_path = args.checkpoint
    pipeline._vae_decoder_blocks = vae_cfg.get("decoder_blocks", [])
    pipeline._vae_causal = vae_cfg.get("causal_decoder", False)
    pipeline._vae_base_channels = vae_cfg.get("decoder_base_channels", 128)

    del raw
    pipeline.load_transformer(transformer_sd)
    del transformer_sd
    gc.collect()

    # === Stage 1: Half-resolution denoising ===
    logger.info(f"=== Stage 1: {s1_height}x{s1_width}, {len(DISTILLED_SIGMA_VALUES)-1} steps ===")
    t0 = time.time()
    s1_video, s1_audio = run_denoise_no_guidance(
        pipeline,
        v_embeds,
        a_embeds,
        num_frames=args.frames,
        height=s1_height,
        width=s1_width,
        sigma_values=DISTILLED_SIGMA_VALUES,
        seed=args.seed,
    )
    logger.info(f"Stage 1: {time.time() - t0:.1f}s")
    logger.info(f"  Video: {s1_video.shape}, range [{s1_video.min():.3f}, {s1_video.max():.3f}]")

    # Free transformer for upsampler
    pipeline.transformer = None
    gc.collect()

    # === Spatial upsampling (reference CPU) ===
    logger.info("=== Upsampling 2x ===")

    # Reshape latent from (B, N, C) to spatial (B, C, T, H, W) for upsampler
    latent_frames = (args.frames - 1) // 8 + 1
    s1_h, s1_w = s1_height // 32, s1_width // 32
    s1_spatial = s1_video.reshape(1, latent_frames, s1_h, s1_w, 128).permute(0, 4, 1, 2, 3)
    # (B, C, T, H, W)

    t0 = time.time()
    upsampled = upsample_latent_reference(s1_spatial, args.upsampler, args.checkpoint)
    logger.info(f"Upsampling: {time.time() - t0:.1f}s")

    # Reshape back to (B, N, C)
    B, C, T, H, W = upsampled.shape
    upsampled_flat = upsampled.permute(0, 2, 3, 4, 1).reshape(1, T * H * W, C)
    logger.info(f"Upsampled flat: {upsampled_flat.shape}")

    # === Reload transformer for Stage 2 ===
    raw = load_file(args.checkpoint)
    transformer_sd = {k[len(prefix) :]: v for k, v in raw.items() if k.startswith(prefix)}
    del raw
    pipeline.load_transformer(transformer_sd)
    del transformer_sd
    gc.collect()

    # === Stage 2: Full-resolution refinement ===
    logger.info(f"=== Stage 2: {args.height}x{args.width}, {len(STAGE_2_DISTILLED_SIGMA_VALUES)-1} steps ===")
    t0 = time.time()
    s2_video, s2_audio = run_denoise_no_guidance(
        pipeline,
        v_embeds,
        a_embeds,
        num_frames=args.frames,
        height=args.height,
        width=args.width,
        sigma_values=STAGE_2_DISTILLED_SIGMA_VALUES,
        seed=args.seed,
        initial_video_latent=upsampled_flat,
        initial_audio_latent=s1_audio.unsqueeze(0) if s1_audio.dim() == 2 else s1_audio,
    )
    logger.info(f"Stage 2: {time.time() - t0:.1f}s")

    # === VAE decode ===
    pipeline.transformer = None
    gc.collect()

    t0 = time.time()
    pipeline.load_vae_from_checkpoint()
    logger.info(f"VAE loaded in {time.time() - t0:.0f}s")

    latent_h, latent_w = args.height // 32, args.width // 32
    t0 = time.time()
    video_pixels = pipeline.decode_latents(s2_video, latent_frames, latent_h, latent_w)
    logger.info(f"VAE decode: {time.time() - t0:.1f}s — {video_pixels.shape}")

    # === Audio decode + export ===
    audio_obj = decode_audio(s2_audio, args.checkpoint, args.frames, fps=args.fps)
    export_video(video_pixels, args.output, fps=args.fps, audio=audio_obj)

    total_time = time.time() - total_t0
    logger.info("=" * 60)
    logger.info(f"Total: {total_time:.1f}s | Output: {args.output}")
    logger.info("=" * 60)

    ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
