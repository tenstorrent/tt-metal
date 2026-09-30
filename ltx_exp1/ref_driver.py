"""Experiment 1 reference driver: run Lightricks' DistilledPipeline (ltx-pipelines, ~/LTX-2) on CPU
and dump everything needed for an injected-noise parity run against the ttnn pipeline.

Run inside ~/ltx2-ref-venv.  Dumps into --out:
  embeds.pt        {"video": (1,L,D), "audio": (1,L,D)}  connector outputs (prompt embeddings)
  noise_s1_video.pt / noise_s1_audio.pt / noise_s2_video.pt / noise_s2_audio.pt
                   raw N(0,1) draws from GaussianNoiser, already in the ttnn token layout
                   (video: b c f h w -> b (f h w) c, audio: b c t f -> b t (c f)), fp32
  s1_video.pt, s1_audio.pt, upsampled.pt, s2_video.pt, s2_audio.pt   stage outputs, same token layout
  audio.wav        vocoded 48 kHz audio (torchaudio)
  out.mp4          (only with --decode-video)
"""
import argparse, json, os, sys, time, logging
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("ref_driver")

SNAP = "/home/rsalman/.cache/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/5948be4ced3a4493d1f836df64378ff136ddb770"
GEMMA = "/home/rsalman/.cache/huggingface/hub/models--google--gemma-3-12b-it-qat-q4_0-unquantized/snapshots"


def _first_snapshot(root):
    subs = sorted(os.listdir(root))
    return os.path.join(root, subs[0])


def tokens_video(x):  # b c f h w -> b (f h w) c   (noise draws arrive already patchified: b n c)
    if x.dim() == 3:
        return x.float().contiguous()
    return x.permute(0, 2, 3, 4, 1).reshape(x.shape[0], -1, x.shape[1]).float().contiguous()


def tokens_audio(x):  # b c t f -> b t (c f)   (noise draws arrive already patchified: b t (c f))
    if x.dim() == 3:
        return x.float().contiguous()
    return x.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[2], -1).float().contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--frames", type=int, default=153)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--height", type=int, default=1088)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=56)
    ap.add_argument("--decode-video", action="store_true")
    ap.add_argument("--checkpoint", default=os.path.join(SNAP, "ltx-2.3-22b-distilled-1.1.safetensors"))
    ap.add_argument("--upsampler", default=os.path.join(SNAP, "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"))
    ap.add_argument("--gemma-root", default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    gemma_root = args.gemma_root or _first_snapshot(GEMMA)

    from ltx_core.components.noisers import GaussianNoiser
    from ltx_core.types import VideoPixelShape
    from ltx_pipelines.distilled import DistilledPipeline
    from ltx_pipelines.chunks import (
        ChunkConfig,
        generate_uniform_chunks,
        denoise_chunks,
        spatially_upsample_chunks,
        decode_chunks,
        pipeline_output_from_chunks,
    )
    from ltx_pipelines.utils.constants import DISTILLED_SIGMAS, STAGE_2_DISTILLED_SIGMAS
    from ltx_pipelines.utils.model_paths import ModelPaths
    from ltx_pipelines.utils.types import VideoAudio
    from ltx_pipelines.distilled import ANCESTRAL_NOISE_SEED_OFFSET, ANCESTRAL_STAGE_2_NOISE_SEED_OFFSET

    # --- record every noise draw (order: s1 video, s1 audio, s2 video, s2 audio) ---
    draws = []
    orig_sample = GaussianNoiser._sample_noise

    def rec_sample(self, latent_state):
        n = orig_sample(self, latent_state)
        draws.append(n.detach().clone())
        log.info("noise draw %d: shape=%s dtype=%s", len(draws), tuple(n.shape), n.dtype)
        return n

    GaussianNoiser._sample_noise = rec_sample
    # CPU: the vocoder is stored bf16 and relies on CUDA autocast to run fp32; on CPU no autocast is applied,
    # so upcast the vocoder module itself (this is what the CUDA path computes in effect: fp32 vocoder).
    import ltx_pipelines.utils.blocks as _blocks

    _orig_vda = _blocks.vae_decode_audio
    _blocks.vae_decode_audio = lambda latent, decoder, vocoder: _orig_vda(latent, decoder, vocoder.float())

    t0 = time.time()
    dev = torch.device("cpu")
    pipe = DistilledPipeline(
        model_paths=ModelPaths.from_monolith(args.checkpoint, gemma_root=gemma_root),
        spatial_upsampler_path=args.upsampler,
        loras=[],
        device=dev,
    )
    log.info("pipeline built in %.0fs; ancestral=%s", time.time() - t0, pipe.use_ancestral_sampler)

    ctx = pipe._prepare_run(
        prompt=args.prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        frame_rate=args.fps,
        images=[],
        num_frames=args.frames,
        vae_dtype=None,
        tiling_config=None if False else __import__("ltx_core.model.video_vae", fromlist=["AUTO_TILING"]).AUTO_TILING,
        enhance_prompt=False,
        enhance_static_cache=False,
        generated_keyframes=0,
        decode_with_keyframes=False,
    )
    log.info(
        "prompt encoded in %.0fs: video ctx %s audio ctx %s; num_frames=%d",
        time.time() - t0,
        tuple(ctx.video_context.shape),
        tuple(ctx.audio_context.shape),
        ctx.num_frames,
    )
    torch.save(
        {"video": ctx.video_context.float().cpu(), "audio": ctx.audio_context.float().cpu(), "prompt": args.prompt},
        os.path.join(args.out, "embeds.pt"),
    )
    meta = dict(vars(args))
    meta.update(
        num_frames=ctx.num_frames,
        stage1=[ctx.stage_1_height, ctx.stage_1_width],
        sigmas1=DISTILLED_SIGMAS.tolist(),
        sigmas2=STAGE_2_DISTILLED_SIGMAS.tolist(),
    )
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=1)

    s1 = DISTILLED_SIGMAS.to(dtype=torch.float32, device=dev)
    s2 = STAGE_2_DISTILLED_SIGMAS.to(dtype=torch.float32, device=dev)
    target = VideoPixelShape(
        batch=1, frames=ctx.num_frames, height=ctx.stage_1_height, width=ctx.stage_1_width, fps=args.fps
    )
    chunk_config = ChunkConfig(chunk_pixel_frames=ctx.num_frames, next_video_carry_frames=0)
    chunks = generate_uniform_chunks(
        target=target,
        context=VideoAudio(video=ctx.video_context, audio=ctx.audio_context),
        device=dev,
        dtype=ctx.dtype,
        config=chunk_config,
        video_scale_factors=pipe.stage.video_scale_factors,
        make_video_conditionings=pipe._video_conditionings(ctx.images, ctx.num_frames, None),
        generated_keyframes=0,
    )
    t1 = time.time()
    base = list(
        denoise_chunks(
            chunks,
            pipe.stage,
            sigmas=s1,
            noiser=ctx.noiser,
            fps=args.fps,
            noise_scale=1.0,
            **pipe._sampler_kwargs(args.seed, ANCESTRAL_NOISE_SEED_OFFSET),
        )
    )
    log.info("stage 1 done in %.0fs", time.time() - t1)
    assert len(base) == 1, len(base)
    torch.save(tokens_video(base[0].video), os.path.join(args.out, "s1_video.pt"))
    torch.save(tokens_audio(base[0].audio), os.path.join(args.out, "s1_audio.pt"))
    torch.save(tokens_video(draws[0]), os.path.join(args.out, "noise_s1_video.pt"))
    torch.save(tokens_audio(draws[1]), os.path.join(args.out, "noise_s1_audio.pt"))

    t1 = time.time()
    up = list(spatially_upsample_chunks(iter(base), pipe.upsampler))
    log.info("upsample done in %.0fs: %s", time.time() - t1, tuple(up[0].video.shape))
    torch.save(tokens_video(up[0].video), os.path.join(args.out, "upsampled.pt"))

    t1 = time.time()
    refined = list(
        denoise_chunks(
            iter(up),
            pipe.stage,
            sigmas=s2,
            noiser=ctx.noiser,
            fps=args.fps,
            **pipe._sampler_kwargs(args.seed, ANCESTRAL_STAGE_2_NOISE_SEED_OFFSET),
        )
    )
    log.info("stage 2 done in %.0fs", time.time() - t1)
    torch.save(tokens_video(refined[0].video), os.path.join(args.out, "s2_video.pt"))
    torch.save(tokens_audio(refined[0].audio), os.path.join(args.out, "s2_audio.pt"))
    torch.save(tokens_video(draws[2]), os.path.join(args.out, "noise_s2_video.pt"))
    torch.save(tokens_audio(draws[3]), os.path.join(args.out, "noise_s2_audio.pt"))
    log.info("total noise draws: %d", len(draws))

    t1 = time.time()
    audio = pipe.audio_decoder(refined[0].audio)
    wav = audio.waveform.float().cpu()
    log.info("audio decoded in %.0fs: %s @ %d Hz", time.time() - t1, tuple(wav.shape), audio.sampling_rate)
    torch.save({"waveform": wav, "sampling_rate": audio.sampling_rate}, os.path.join(args.out, "audio.pt"))
    try:
        import torchaudio

        w = wav if wav.dim() == 2 else wav.reshape(-1, wav.shape[-1])
        torchaudio.save(os.path.join(args.out, "audio.wav"), w.clamp(-1, 1), audio.sampling_rate)
    except Exception as e:  # noqa: BLE001
        log.warning("wav export failed: %s", e)

    if args.decode_video:
        from ltx_pipelines.utils.media_io import encode_video
        from ltx_core.model.video_vae import get_video_chunks_number

        t1 = time.time()
        result = pipeline_output_from_chunks(
            decode_chunks(
                iter(refined),
                pipe.video_decoder,
                pipe.audio_decoder,
                fps=args.fps,
                tiling_config=ctx.tiling_config,
                generator=ctx.generator,
                dtype=ctx.vae_dtype,
                keyframes=False,
            ),
            num_frames=ctx.num_frames,
            tiling_config=ctx.tiling_config,
        )
        encode_video(
            video=result.video,
            fps=args.fps,
            audio=result.audio,
            output_path=os.path.join(args.out, "out.mp4"),
            video_chunks_number=get_video_chunks_number(result.num_frames, result.tiling_config),
        )
        log.info("video decoded+encoded in %.0fs", time.time() - t1)
    log.info("ALL DONE in %.0fs -> %s", time.time() - t0, args.out)


if __name__ == "__main__":
    # Same as ltx_pipelines.distilled.main(): without inference_mode the forward keeps every activation alive
    # for autograd (OOM-killed at 540 GB on the served shape).
    with torch.inference_mode():
        main()
