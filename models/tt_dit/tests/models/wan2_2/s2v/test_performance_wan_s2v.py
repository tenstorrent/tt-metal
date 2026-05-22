# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Multi-clip S2V pipeline performance benchmark."""

from __future__ import annotations

import os

import numpy as np
import PIL
import pytest
import torch
from loguru import logger

import ttnn
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_dit.pipelines.wan.pipeline_wan_s2v import WanPipelineS2V
from models.tt_dit.utils.video import export_to_video_with_audio

from .....utils.test import line_params, ring_params_8k

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
        pytest.param((2, 4), (2, 4), 1, 0, 2, False, line_params, ttnn.Topology.Linear, False, id="bh_2x4sp1tp0"),
        pytest.param((4, 8), (4, 8), 1, 0, 2, False, ring_params_8k, ttnn.Topology.Ring, False, id="bh_4x8sp1tp0"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "width, height",
    [
        (832, 480),
        (1280, 720),
    ],
    ids=["resolution_480p", "resolution_720p"],
)
@pytest.mark.parametrize("num_inference_steps", [1, 5, 40], ids=["steps1", "steps5", "steps40"])
@pytest.mark.parametrize("num_clips", [1, 4], ids=["clips1", "clips4"])
def test_pipeline_performance_s2v(
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
    """Multi-clip performance breakdown."""
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
        num_frames=81,
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

    output_path = f"wan_s2v_perf_{width}x{height}_{num_clips}c_{num_inference_steps}s.mp4"
    export_to_video_with_audio(frames[0], output_path, audio_path=_AUDIO_PATH, fps=16)
    logger.info(f"Saved video with audio: {output_path}")

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

    sum_denoise = sum(per_clip_denoise)
    sum_vae_decode = sum(per_clip_vae_decode)
    sum_vae_encode_motion = sum(per_clip_vae_encode_motion)
    sum_prep_audio = sum(per_clip_prepare_audio_emb)
    sum_prep_cond = sum(per_clip_prepare_cond_emb)
    total_denoise_steps = max(1, num_clips * num_inference_steps)

    print(
        f"\nS2V PERF {mesh_shape[0]}x{mesh_shape[1]} {width}x{height} "
        f"{num_clips}c x {num_inference_steps}s  sp={sp_factor} tp={tp_factor}"
    )
    for name, value in [
        ("encoder", encoder_t),
        ("prepare_latents", prepare_latents_t),
        ("  vae_encode_ref", vae_encode_ref_t),
        ("  wav2vec2", wav2vec2_t),
        ("  vae_encode_motion0", vae_encode_initial_motion_t),
        ("denoise (sum)", sum_denoise),
        ("vae_decode (sum)", sum_vae_decode),
        ("vae_encode_motion (sum)", sum_vae_encode_motion),
        ("TOTAL", total),
    ]:
        print(f"  {name:30}  {value:8.3f}s")
    print(f"  per-step denoise: {sum_denoise / total_denoise_steps:.3f}s")
    for r in range(num_clips):
        print(
            f"  clip {r}: total={per_clip_total[r]:.2f}s denoise={per_clip_denoise[r]:.2f}s "
            f"vae_dec={per_clip_vae_decode[r]:.2f}s"
        )
