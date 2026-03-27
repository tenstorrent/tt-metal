#!/usr/bin/env python3
"""
LTX-2.3 Pro (one-stage) pipeline runner.

Matches the reference ti2vid_one_stage.py flow exactly:
- Gemma QAT encoding via reference FeatureExtractorV2
- Full MultiModalGuider guidance (CFG + STG + modality)
- 30 denoising steps
- TTNN DiT transformer on device
- TTNN VAE decoder on device
- Reference audio decoder + MP4 export

Usage:
    python run_ltx_pro.py --prompt "A cat playing piano" --output output.mp4
    python run_ltx_pro.py --prompt "..." --output out.mp4 --steps 30 --seed 10
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
from models.tt_dit.pipelines.ltx.pipeline_ltx import LTXPipeline

# Reference defaults from LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py
DEFAULT_CKPT = os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2.3-22b-dev.safetensors")
DEFAULT_GEMMA = "/localdev/kevinmi/.cache/gemma-3-12b-it-qat-q4_0-unquantized"


def decode_audio(audio_latent, checkpoint_path, num_frames, fps=24.0):
    """Decode audio latent using reference audio VAE + vocoder."""
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


def main():
    parser = argparse.ArgumentParser(description="LTX-2.3 Pro (one-stage) pipeline")
    parser.add_argument("--prompt", required=True, help="Text prompt for video generation")
    parser.add_argument("--output", required=True, help="Output MP4 path")
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT, help="LTX-2 checkpoint path")
    parser.add_argument("--gemma", default=DEFAULT_GEMMA, help="Gemma model path")
    parser.add_argument("--seed", type=int, default=10, help="Random seed (reference default: 10)")
    parser.add_argument("--steps", type=int, default=30, help="Denoising steps (reference default: 30)")
    parser.add_argument("--frames", type=int, default=121, help="Number of frames (must be 8k+1)")
    parser.add_argument("--height", type=int, default=512, help="Video height (divisible by 32)")
    parser.add_argument("--width", type=int, default=768, help="Video width (divisible by 32)")
    parser.add_argument("--fps", type=int, default=24, help="Frame rate")
    # Guidance scales matching reference ti2vid_one_stage.py defaults for LTX-2.3
    parser.add_argument("--video-cfg", type=float, default=3.0)
    parser.add_argument("--audio-cfg", type=float, default=7.0)
    parser.add_argument("--video-stg", type=float, default=1.0)
    parser.add_argument("--audio-stg", type=float, default=1.0)
    parser.add_argument("--video-modality", type=float, default=3.0)
    parser.add_argument("--audio-modality", type=float, default=3.0)
    parser.add_argument("--rescale", type=float, default=0.7)
    parser.add_argument("--stg-block", type=int, default=28)
    parser.add_argument("--ge-gamma", type=float, default=0.0, help="Gradient estimation (0=disabled)")
    args = parser.parse_args()

    assert os.path.exists(args.checkpoint), f"Checkpoint not found: {args.checkpoint}"
    assert os.path.isdir(args.gemma), f"Gemma not found: {args.gemma}"

    from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT

    logger.info("=" * 60)
    logger.info("LTX-2.3 Pro (one-stage)")
    logger.info("=" * 60)
    logger.info(f"Prompt: {args.prompt[:80]}...")
    logger.info(f"Config: {args.frames}f @ {args.height}x{args.width}, {args.steps} steps, seed={args.seed}")
    logger.info(f"Guidance: CFG v={args.video_cfg}/a={args.audio_cfg}, STG v={args.video_stg}/a={args.audio_stg}")
    logger.info(f"Output: {args.output}")

    total_t0 = time.time()

    # === Stage 1: Text encoding ===
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
    results = pipeline.encode_prompts_reference([args.prompt, DEFAULT_NEGATIVE_PROMPT], args.checkpoint, args.gemma)
    logger.info(f"Encoding: {time.time() - t0:.1f}s")

    v_embeds = results[0].video_encoding.float()
    a_embeds = results[0].audio_encoding.float()
    neg_v = results[1].video_encoding.float()
    neg_a = results[1].audio_encoding.float()

    # === Stage 2: Load DiT + denoise ===
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

    t0 = time.time()
    video_latent, audio_latent = pipeline.call_av(
        video_prompt_embeds=v_embeds,
        audio_prompt_embeds=a_embeds,
        neg_video_prompt_embeds=neg_v,
        neg_audio_prompt_embeds=neg_a,
        num_frames=args.frames,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        video_cfg_scale=args.video_cfg,
        audio_cfg_scale=args.audio_cfg,
        video_stg_scale=args.video_stg,
        audio_stg_scale=args.audio_stg,
        video_modality_scale=args.video_modality,
        audio_modality_scale=args.audio_modality,
        rescale_scale=args.rescale,
        stg_block=args.stg_block,
        seed=args.seed,
        ge_gamma=args.ge_gamma,
    )
    denoise_time = time.time() - t0
    logger.info(f"Denoising: {denoise_time:.1f}s ({denoise_time / args.steps:.1f}s/step)")

    # === Stage 3: VAE decode ===
    pipeline.transformer = None
    gc.collect()

    t0 = time.time()
    pipeline.load_vae_from_checkpoint()
    logger.info(f"VAE loaded in {time.time() - t0:.0f}s")

    latent_frames = (args.frames - 1) // 8 + 1
    latent_h, latent_w = args.height // 32, args.width // 32

    t0 = time.time()
    video_pixels = pipeline.decode_latents(video_latent, latent_frames, latent_h, latent_w)
    logger.info(f"VAE decode: {time.time() - t0:.1f}s — {video_pixels.shape}")

    # === Stage 4: Audio decode + export ===
    audio_obj = decode_audio(audio_latent, args.checkpoint, args.frames, fps=args.fps)
    export_video(video_pixels, args.output, fps=args.fps, audio=audio_obj)

    total_time = time.time() - total_t0
    logger.info("=" * 60)
    logger.info(f"Total: {total_time:.1f}s | Output: {args.output}")
    logger.info("=" * 60)

    ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
