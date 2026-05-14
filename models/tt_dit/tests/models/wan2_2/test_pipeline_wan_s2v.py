# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import itertools
import os

import numpy as np
import PIL
import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.pipelines.wan.pipeline_wan_s2v import WanPipelineS2V

from ....utils.test import line_params, ring_params


@pytest.mark.parametrize(
    "no_prompt",
    [{"1": True, "0": False}.get(os.environ.get("NO_PROMPT"), False)],
)
@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, device_params, topology, is_fsdp, sdpa_t_fracture_w_only",
    [
        # WH (ring) on 4x8
        [(4, 8), (4, 8), 1, 0, 4, False, ring_params, ttnn.Topology.Ring, True, False],
        # BH (linear) on 4x8 — primary S2V target.
        [(4, 8), (4, 8), 1, 0, 2, False, line_params, ttnn.Topology.Linear, False, False],
    ],
    ids=[
        "wh_4x8sp1tp0",
        "bh_4x8sp1tp0",
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "width, height",
    [
        (832, 480),
        (1280, 720),
    ],
    ids=[
        "resolution_480p",
        "resolution_720p",
    ],
)
def test_pipeline_inference(
    mesh_device,
    mesh_shape,
    sp_axis,
    tp_axis,
    num_links,
    dynamic_load,
    topology,
    width,
    height,
    is_fsdp,
    sdpa_t_fracture_w_only,
    no_prompt,
):
    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    ref_image = PIL.Image.open("./ref_image.png")
    audio_path = "./prompt_audio.wav"
    negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

    num_frames = 81
    # Default 40 for production, but allow a small-step smoke run via
    # ``NUM_INFERENCE_STEPS=N`` to keep the test inside pytest's 300s default
    # timeout. The hard-coded 40 makes this take 30+ min on BH 4×8.
    num_inference_steps = int(os.environ.get("NUM_INFERENCE_STEPS", "40"))
    guidance_scale = 5.0
    guidance_scale_2 = 5.0

    pipeline = WanPipelineS2V.create_pipeline(
        mesh_device=mesh_device,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_links=num_links,
        dynamic_load=dynamic_load,
        topology=topology,
        is_fsdp=is_fsdp,
        sdpa_t_fracture_w_only=sdpa_t_fracture_w_only,
        height=height,
        width=width,
        num_frames=num_frames,
    )

    # Canonical reference S2V example pair (wan2_2_ref/README.md §S2V):
    #   --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard."
    #   --image examples/i2v_input.JPG  --audio examples/talk.wav
    prompt = os.environ.get(
        "S2V_PROMPT",
        "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard.",
    )

    def run(*, prompt, number, seed):
        logger.info(f"Running inference with prompt: '{prompt}'")
        logger.info(f"  audio_path: {audio_path}")
        logger.info(f"  parameters: {height}x{width}, {num_frames} frames, {num_inference_steps} steps")

        with torch.no_grad():
            result = pipeline(
                prompt=prompt,
                image_prompt=ref_image,
                audio_prompt=audio_path,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                seed=seed,
                guidance_scale=guidance_scale,
                guidance_scale_2=guidance_scale_2,
                output_type="uint8",
            )

        frames = result.frames if hasattr(result, "frames") else (result[0] if isinstance(result, tuple) else result)

        logger.info(f"  output shape: {getattr(frames, 'shape', 'Unknown')}")
        if isinstance(frames, np.ndarray):
            logger.info(f"  video data range: [{frames.min():.3f}, {frames.max():.3f}]")
        elif isinstance(frames, torch.Tensor):
            logger.info(f"  video data range: [{frames.min().item():.3f}, {frames.max().item():.3f}]")

        frames = frames[0]  # strip batch
        output_filename = f"wan_s2v_{width}x{height}_{number}.mp4"
        try:
            from models.tt_dit.utils.video import export_to_video

            export_to_video(frames, output_filename, fps=16)
            logger.info(f"Saved video to: {output_filename}")
        except ImportError:
            logger.info("Could not export video — imageio_ffmpeg not available")

        # Mux the input audio into the output MP4 (matches the reference's
        # ``wan.utils.utils.merge_video_audio`` step). Without this, the MP4
        # is silent. ffmpeg re-encodes the video with the audio side-by-side.
        import os
        import subprocess

        if os.path.exists(audio_path):
            try:
                from imageio_ffmpeg import get_ffmpeg_exe

                base, ext = os.path.splitext(output_filename)
                temp_output = f"{base}_temp{ext}"
                cmd = [
                    get_ffmpeg_exe(),
                    "-y",
                    "-i",
                    output_filename,
                    "-i",
                    audio_path,
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-shortest",  # trim to shorter of the two streams
                    temp_output,
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                os.replace(temp_output, output_filename)
                logger.info(f"Muxed audio from {audio_path} into {output_filename}")
            except (ImportError, subprocess.CalledProcessError, FileNotFoundError) as e:
                logger.warning(f"Audio mux failed (output remains silent): {e}")

    if no_prompt:
        run(prompt=prompt, number=0, seed=42)
    else:
        for i in itertools.count():
            new_prompt = input("Enter the input prompt, or q to exit: ")
            if new_prompt:
                prompt = new_prompt
            if prompt[0] == "q":
                break
            run(prompt=prompt, number=i, seed=i)
