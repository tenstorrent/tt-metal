# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Performance benchmark for the S2V pipeline.

Measures the time breakdown across the four base-pipeline stages already
hooked in ``pipeline_wan.py``:

  * ``encoder``         — UMT5 text encoder forward.
  * ``prepare_latents`` — for S2V this is wav2vec2 + audio prep + VAE encode
    (ref image, motion latents) + ``prepare_cond_emb``.
  * ``denoising``       — the full diffusion loop (``num_inference_steps``
    transformer forward passes + scheduler steps).
  * ``vae``             — VAE decoder.

Reports mean/min/max wall-clock for each stage so we can see where
optimization effort should focus on BH-LB (2x4).
"""

import statistics

import numpy as np
import PIL
import pytest
import torch
from loguru import logger

import ttnn
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_dit.pipelines.wan.pipeline_wan_s2v import WanPipelineS2V

from ....utils.test import line_params

_REF_IMAGE_PATH = "./ref_image.png"
_AUDIO_PATH = "./prompt_audio.wav"
_PROMPT = "a person is talking"
_NEGATIVE_PROMPT = (
    "画面模糊，最差质量，画面模糊，细节模糊不清，情绪激动剧烈，手快速抖动，字幕，丑陋的，残缺的，"
    "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, device_params, topology, is_fsdp",
    [
        # BH-LB (2x4) — the only S2V-supported BH mesh in device_configs today.
        [(2, 4), (2, 4), 1, 0, 2, False, line_params, ttnn.Topology.Linear, False],
    ],
    ids=["bh_2x4sp1tp0"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "width, height",
    [(832, 480)],
    ids=["resolution_480p"],
)
@pytest.mark.parametrize(
    "num_inference_steps",
    [5, 40],
    ids=["steps5", "steps40"],
)
def test_s2v_pipeline_performance(
    *,
    mesh_device: ttnn.MeshDevice,
    mesh_shape: tuple,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    dynamic_load: bool,
    topology: ttnn.Topology,
    width: int,
    height: int,
    is_fsdp: bool,
    num_inference_steps: int,
) -> None:
    """Performance breakdown for WanPipelineS2V."""
    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]

    num_frames = 81
    # Reference s2v_14B (wan_s2v_14B.py:59) uses sample_guide_scale=4.5.
    guidance_scale = 4.5
    guidance_scale_2 = 4.5

    print(f"Parameters: {height}x{width}, {num_frames} frames, {num_inference_steps} steps")

    pipeline = WanPipelineS2V.create_pipeline(
        mesh_device=mesh_device,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_links=num_links,
        dynamic_load=dynamic_load,
        topology=topology,
        is_fsdp=is_fsdp,
        sdpa_t_fracture_w_only=False,
        height=height,
        width=width,
        num_frames=num_frames,
    )

    ref_image = PIL.Image.open(_REF_IMAGE_PATH)

    benchmark_profiler = BenchmarkProfiler()

    # No warmup — S2V's weight load + audio encoder warmup is part of what we
    # want to measure here. Single perf run.
    num_perf_runs = 1

    ttnn.synchronize_device(mesh_device)

    for i in range(num_perf_runs):
        logger.info(f"S2V performance run {i+1}/{num_perf_runs} ({num_inference_steps} steps)...")
        with benchmark_profiler("run", iteration=i):
            with torch.no_grad():
                result = pipeline(
                    prompt=_PROMPT,
                    image_prompt=ref_image,
                    audio_prompt=_AUDIO_PATH,
                    negative_prompt=_NEGATIVE_PROMPT,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    seed=0,
                    guidance_scale=guidance_scale,
                    guidance_scale_2=guidance_scale_2,
                    output_type="uint8",
                    profiler=benchmark_profiler,
                    profiler_iteration=i,
                )
                ttnn.synchronize_device(mesh_device)
        logger.info(f"  Run {i+1} completed in {benchmark_profiler.get_duration('run', i):.2f}s")

    frames = result.frames if hasattr(result, "frames") else (result[0] if isinstance(result, tuple) else result)
    if isinstance(frames, np.ndarray):
        print(f"Output shape: {frames.shape}, range: [{frames.min()}, {frames.max()}]")
    elif isinstance(frames, torch.Tensor):
        print(f"Output shape: {tuple(frames.shape)}, range: [{frames.min().item():.1f}, {frames.max().item():.1f}]")

    def _maybe_durs(step_name: str) -> list[float]:
        return [
            benchmark_profiler.get_duration(step_name, i)
            for i in range(num_perf_runs)
            if benchmark_profiler.contains_step(step_name, i)
        ]

    encoder_times = _maybe_durs("encoder")
    prepare_latents_times = _maybe_durs("prepare_latents")
    s2v_vae_ref_times = _maybe_durs("s2v_vae_encode_ref")
    s2v_wav2vec2_times = _maybe_durs("s2v_wav2vec2")
    s2v_prep_audio_emb_times = _maybe_durs("s2v_prepare_audio_emb")
    s2v_vae_motion_times = _maybe_durs("s2v_vae_encode_motion")
    s2v_prep_cond_emb_times = _maybe_durs("s2v_prepare_cond_emb")
    denoising_times = _maybe_durs("denoising")
    vae_times = _maybe_durs("vae")
    total_times = _maybe_durs("run")

    # Audio cross-attn injection accumulator across the denoising loop —
    # reported directly by the transformer (post-inference).
    audio_inject_seconds = float(getattr(pipeline.transformer, "_audio_inject_sec_accum", 0.0))

    print("\n" + "=" * 80)
    print(f"WAN 2.2 S2V PERFORMANCE — BH-LB ({mesh_shape[0]}x{mesh_shape[1]}, {num_inference_steps} steps, {height}p)")
    print("=" * 80)
    print(f"Resolution:       {width}x{height}")
    print(f"Num Frames:       {num_frames}")
    print(f"Inference Steps:  {num_inference_steps}")
    print(f"DiT Parallel:     sp={sp_factor}, tp={tp_factor}, topology={topology}")
    print("-" * 80)

    def print_stats(name: str, times: list[float], *, per_step: bool = False, indent: int = 2) -> float:
        prefix = " " * indent
        if not times:
            print(f"{prefix}{name:36}  (no data)")
            return 0.0
        mean_time = statistics.mean(times)
        if per_step and num_inference_steps > 0:
            extra = f"   {mean_time / num_inference_steps:6.3f}s/step"
        else:
            extra = ""
        print(f"{prefix}{name:36}  mean={mean_time:8.3f}s{extra}")
        return mean_time

    print("TOP-LEVEL STAGES")
    print_stats("Text encoder (UMT5)", encoder_times)
    pl_time = print_stats("prepare_latents (total)", prepare_latents_times)
    print_stats("Denoising loop", denoising_times, per_step=True)
    print_stats("VAE decoder", vae_times)
    print_stats("TOTAL pipeline", total_times)

    print()
    print("PREPARE_LATENTS BREAKDOWN (S2V-specific)")
    print_stats("VAE encode (ref image)", s2v_vae_ref_times)
    print_stats("wav2vec2 + bucketing", s2v_wav2vec2_times)
    print_stats("prepare_audio_emb (CausalAudioEnc)", s2v_prep_audio_emb_times)
    print_stats("VAE encode (motion 73f zeros)", s2v_vae_motion_times)
    print_stats("prepare_cond_emb", s2v_prep_cond_emb_times)

    print()
    print("DENOISING-LOOP BREAKDOWN")
    if denoising_times:
        denoise_mean = statistics.mean(denoising_times)
        print(f"  {'Denoising loop (total)':36}  mean={denoise_mean:8.3f}s")
        print(
            f"  {'Audio cross-attn (cumulative)':36}  mean={audio_inject_seconds:8.3f}s"
            f"   {audio_inject_seconds / max(num_inference_steps, 1):.4f}s/step"
            f"  ({100.0 * audio_inject_seconds / max(denoise_mean, 1e-9):4.1f}% of denoise)"
        )
        block_other = denoise_mean - audio_inject_seconds
        print(
            f"  {'Block-stack (non-audio, cumulative)':36}  mean={block_other:8.3f}s"
            f"   {block_other / max(num_inference_steps, 1):.4f}s/step"
        )

    if total_times:
        total = statistics.mean(total_times)
        print()
        print("STAGE SHARE OF TOTAL PIPELINE")
        for name, times in [
            ("text encoder", encoder_times),
            ("prepare_latents", prepare_latents_times),
            ("  ↳ VAE encode (ref)", s2v_vae_ref_times),
            ("  ↳ wav2vec2", s2v_wav2vec2_times),
            ("  ↳ prepare_audio_emb", s2v_prep_audio_emb_times),
            ("  ↳ VAE encode (motion)", s2v_vae_motion_times),
            ("  ↳ prepare_cond_emb", s2v_prep_cond_emb_times),
            ("denoising loop", denoising_times),
            ("  ↳ audio injection", [audio_inject_seconds] if audio_inject_seconds > 0 else []),
            ("vae decode", vae_times),
        ]:
            if times:
                pct = 100.0 * statistics.mean(times) / total
                print(f"  {name:36}  {pct:5.1f}%")
    print("=" * 80)

    logger.info("S2V performance test completed!")
