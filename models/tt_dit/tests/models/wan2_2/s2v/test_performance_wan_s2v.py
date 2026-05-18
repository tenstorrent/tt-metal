# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Performance benchmark for the multi-clip S2V pipeline.

Reports per-clip and aggregate wall-clock for each stage emitted by
``WanPipelineS2V.__call__``:

  * ``encoder``                       — UMT5 text encoder forward (once).
  * ``prepare_latents``               — wav2vec2 + audio bucketing + ref VAE
                                        encode + initial-motion VAE encode (once).
  * ``s2v_clip_{r}_prepare_audio_emb``— on-device CausalAudioEncoder (per clip).
  * ``s2v_clip_{r}_prepare_cond_emb`` — cond/ref/motion prep (per clip).
  * ``s2v_clip_{r}_denoise``          — diffusion loop (per clip).
  * ``s2v_clip_{r}_vae_decode``       — VAE decode (per clip).
  * ``s2v_clip_{r}_vae_encode_motion``— VAE encode of next-clip motion (per clip).
"""

import os
import statistics

import numpy as np
import PIL
import pytest
import torch
from loguru import logger

import ttnn
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_dit.pipelines.wan.pipeline_wan_s2v import WanPipelineS2V

from .....utils.test import line_params

# Inputs are expected at the repo root (same pattern as test_pipeline_wan_i2v.py).
# Override with env vars when needed.
_REF_IMAGE_PATH = os.environ.get("S2V_REF_IMAGE", "./prompt_image.png")
_AUDIO_PATH = os.environ.get("S2V_AUDIO", "./prompt_audio.wav")
_PROMPT = "a person is talking"
_NEGATIVE_PROMPT = (
    "画面模糊，最差质量，画面模糊，细节模糊不清，情绪激动剧烈，手快速抖动，字幕，丑陋的，残缺的，"
    "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


@pytest.mark.timeout(3000)
@pytest.mark.parametrize(
    (
        "mesh_device",
        "mesh_shape",
        "sp_axis",
        "tp_axis",
        "num_links",
        "dynamic_load",
        "device_params",
        "topology",
        "is_fsdp",
    ),
    [
        pytest.param(
            (2, 4),
            (2, 4),
            1,
            0,
            2,
            False,
            line_params,
            ttnn.Topology.Linear,
            False,
            id="bh_2x4sp1tp0",
        ),
    ],
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
@pytest.mark.parametrize(
    "num_clips",
    [1, 4],
    ids=["clips1", "clips4"],
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
    num_clips: int,
) -> None:
    """Multi-clip performance breakdown for WanPipelineS2V."""
    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]
    guidance_scale = 4.5  # reference wan_s2v_14B.py:59

    print(
        f"Parameters: {height}x{width}, {num_clips} clips × {WanPipelineS2V._INFER_FRAMES_PIXEL} pixels, "
        f"{num_inference_steps} steps"
    )

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
        num_frames=81,  # reserved hook into create_pipeline's config-loading path
    )

    ref_image = PIL.Image.open(_REF_IMAGE_PATH)

    profiler = BenchmarkProfiler()

    ttnn.synchronize_device(mesh_device)

    logger.info(f"S2V performance run: num_clips={num_clips}, steps={num_inference_steps}")
    with profiler("run", iteration=0):
        with torch.no_grad():
            result = pipeline(
                prompt=_PROMPT,
                image_prompt=ref_image,
                audio_prompt=_AUDIO_PATH,
                negative_prompt=_NEGATIVE_PROMPT,
                height=height,
                width=width,
                num_clips=num_clips,
                num_inference_steps=num_inference_steps,
                seed=0,
                guidance_scale=guidance_scale,
                output_type="uint8",
                profiler=profiler,
                profiler_iteration=0,
            )
            ttnn.synchronize_device(mesh_device)
    logger.info(f"  Run completed in {profiler.get_duration('run', 0):.2f}s")

    frames = result.frames if hasattr(result, "frames") else (result[0] if isinstance(result, tuple) else result)
    if isinstance(frames, np.ndarray):
        print(f"Output shape: {frames.shape}, range: [{frames.min()}, {frames.max()}]")
    elif isinstance(frames, torch.Tensor):
        print(f"Output shape: {tuple(frames.shape)}, range: [{frames.min().item():.1f}, {frames.max().item():.1f}]")

    def _dur(step_name: str) -> float:
        return profiler.get_duration(step_name, 0) if profiler.contains_step(step_name, 0) else 0.0

    # Top-level (once per pipeline call) stages.
    total = _dur("run")
    encoder_t = _dur("encoder")
    prepare_latents_t = _dur("prepare_latents")
    vae_encode_ref_t = _dur("s2v_vae_encode_ref")
    wav2vec2_t = _dur("s2v_wav2vec2")
    vae_encode_initial_motion_t = _dur("s2v_vae_encode_motion")

    # Per-clip stages.
    per_clip_total = [_dur(f"s2v_clip_{r}_total") for r in range(num_clips)]
    per_clip_prepare_audio_emb = [_dur(f"s2v_clip_{r}_prepare_audio_emb") for r in range(num_clips)]
    per_clip_prepare_cond_emb = [_dur(f"s2v_clip_{r}_prepare_cond_emb") for r in range(num_clips)]
    per_clip_denoise = [_dur(f"s2v_clip_{r}_denoise") for r in range(num_clips)]
    per_clip_vae_decode = [_dur(f"s2v_clip_{r}_vae_decode") for r in range(num_clips)]
    per_clip_vae_encode_motion = [_dur(f"s2v_clip_{r}_vae_encode_motion") for r in range(num_clips)]

    # Audio cross-attn injection accumulator (cumulative across the whole run).
    audio_inject_seconds = float(getattr(pipeline.transformer, "_audio_inject_sec_accum", 0.0))

    sum_denoise = sum(per_clip_denoise)
    sum_vae_decode = sum(per_clip_vae_decode)
    sum_vae_encode_motion = sum(per_clip_vae_encode_motion)
    sum_prep_audio = sum(per_clip_prepare_audio_emb)
    sum_prep_cond = sum(per_clip_prepare_cond_emb)
    total_denoise_steps = max(1, num_clips * num_inference_steps)

    print("\n" + "=" * 88)
    print(
        f"WAN 2.2 S2V PERFORMANCE — BH-LB ({mesh_shape[0]}x{mesh_shape[1]}), "
        f"{num_clips} clip{'s' if num_clips != 1 else ''} × {num_inference_steps} steps, {height}p"
    )
    print("=" * 88)
    print(f"Resolution:        {width}x{height}")
    print(f"Output frames:     {num_clips * WanPipelineS2V._INFER_FRAMES_PIXEL - WanPipelineS2V._S2V_VAE_CLIP0_TRIM}")
    print(f"DiT Parallel:      sp={sp_factor}, tp={tp_factor}, topology={topology}")
    print("-" * 88)

    def row(name: str, value: float, *, per_step: bool = False, indent: int = 2) -> None:
        prefix = " " * indent
        if per_step and total_denoise_steps > 0:
            extra = f"   {value / total_denoise_steps:6.3f}s/step"
        else:
            extra = ""
        print(f"{prefix}{name:38}  {value:8.3f}s{extra}")

    print("TOP-LEVEL")
    row("Text encoder (UMT5)", encoder_t)
    row("prepare_latents (one-time, audio+ref)", prepare_latents_t)
    row("Sum denoise (across clips)", sum_denoise, per_step=True)
    row("Sum VAE decode (across clips)", sum_vae_decode)
    row("Sum VAE motion-encode (across clips)", sum_vae_encode_motion)
    row("TOTAL pipeline", total)

    print()
    print("ONE-TIME PREPARE_LATENTS BREAKDOWN")
    row("VAE encode (ref image)", vae_encode_ref_t)
    row("wav2vec2 + bucketing", wav2vec2_t)
    row("VAE encode (initial zero motion)", vae_encode_initial_motion_t)

    print()
    print("PER-CLIP BREAKDOWN")
    print(
        f"  {'clip':4} {'total':>9} {'prep_audio':>11} {'prep_cond':>10} {'denoise':>9} {'vae_dec':>9} {'vae_mot_enc':>12}"
    )
    for r in range(num_clips):
        print(
            f"  {r:4d} {per_clip_total[r]:>8.2f}s "
            f"{per_clip_prepare_audio_emb[r]:>10.2f}s "
            f"{per_clip_prepare_cond_emb[r]:>9.2f}s "
            f"{per_clip_denoise[r]:>8.2f}s "
            f"{per_clip_vae_decode[r]:>8.2f}s "
            f"{per_clip_vae_encode_motion[r]:>11.2f}s"
        )
    if num_clips > 1:
        print(
            f"  {'mean':>4} {statistics.mean(per_clip_total):>8.2f}s "
            f"{statistics.mean(per_clip_prepare_audio_emb):>10.2f}s "
            f"{statistics.mean(per_clip_prepare_cond_emb):>9.2f}s "
            f"{statistics.mean(per_clip_denoise):>8.2f}s "
            f"{statistics.mean(per_clip_vae_decode):>8.2f}s "
            f"{statistics.mean(per_clip_vae_encode_motion):>11.2f}s"
        )

    print()
    print("DENOISE-LOOP AGGREGATES")
    print(
        f"  Total denoise:                       {sum_denoise:8.3f}s   {sum_denoise / total_denoise_steps:6.3f}s/step"
    )
    print(
        f"  Audio cross-attn (cumulative):       {audio_inject_seconds:8.3f}s   "
        f"{audio_inject_seconds / total_denoise_steps:6.4f}s/step  "
        f"({100.0 * audio_inject_seconds / max(sum_denoise, 1e-9):4.1f}% of denoise)"
    )
    block_other = sum_denoise - audio_inject_seconds
    print(
        f"  Block-stack (non-audio, cumulative): {block_other:8.3f}s   {block_other / total_denoise_steps:6.4f}s/step"
    )

    if total > 0:
        print()
        print("STAGE SHARE OF TOTAL PIPELINE")
        for name, value in [
            ("text encoder", encoder_t),
            ("prepare_latents (one-time)", prepare_latents_t),
            ("  ↳ VAE encode (ref)", vae_encode_ref_t),
            ("  ↳ wav2vec2", wav2vec2_t),
            ("  ↳ VAE encode (initial motion)", vae_encode_initial_motion_t),
            ("per-clip prepare_audio_emb (sum)", sum_prep_audio),
            ("per-clip prepare_cond_emb (sum)", sum_prep_cond),
            ("per-clip denoise (sum)", sum_denoise),
            ("  ↳ audio injection", audio_inject_seconds),
            ("per-clip vae_decode (sum)", sum_vae_decode),
            ("per-clip vae_encode_motion (sum)", sum_vae_encode_motion),
        ]:
            if value > 0:
                print(f"  {name:38}  {100.0 * value / total:5.1f}%")
    print("=" * 88)

    logger.info("S2V performance test completed!")
