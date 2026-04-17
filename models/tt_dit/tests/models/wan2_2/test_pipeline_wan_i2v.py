# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import gc
import itertools
import os

import numpy as np
import PIL
import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.pipelines.wan.pipeline_wan_i2v import ImagePrompt, WanPipelineI2V

from ....utils.test import line_params, ring_params


@pytest.mark.parametrize(
    "no_prompt",
    [{"1": True, "0": False}.get(os.environ.get("NO_PROMPT"), False)],
)
@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, device_params, topology, is_fsdp",
    [
        [(2, 2), (2, 2), 0, 1, 2, False, line_params, ttnn.Topology.Linear, True],
        [(2, 4), (2, 4), 0, 1, 1, True, line_params, ttnn.Topology.Linear, True],
        # BH on 2x4 with dynamic_load to avoid init-time DRAM OOM
        [(2, 4), (2, 4), 1, 0, 2, True, line_params, ttnn.Topology.Linear, False],
        # WH (ring) on 4x8
        [(4, 8), (4, 8), 1, 0, 4, False, ring_params, ttnn.Topology.Ring, True],
        # BH (linear) on 4x8
        [(4, 8), (4, 8), 1, 0, 2, False, line_params, ttnn.Topology.Linear, False],
    ],
    ids=[
        "2x2sp0tp1",
        "2x4sp0tp1",
        "bh_2x4sp1tp0",
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
    no_prompt,
):
    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    num_frames = int(os.environ.get("WAN_NUM_FRAMES", "81"))
    assert (num_frames - 1) % 4 == 0, f"num_frames must satisfy (n-1) % 4 == 0; got {num_frames}"
    num_inference_steps = int(os.environ.get("WAN_STEPS", "40"))
    guidance_scale = float(os.environ.get("WAN_GUIDANCE", "3.5"))
    guidance_scale_2 = float(os.environ.get("WAN_GUIDANCE_2", "3.5"))
    seed_env = int(os.environ.get("WAN_SEED", "42"))
    output_path_env = os.environ.get("WAN_OUTPUT_PATH", "")

    first_image_path = os.environ.get("WAN_FIRST_IMAGE", "./prompt_image.png")
    last_image_path = os.environ.get("WAN_LAST_IMAGE", "")
    pil_image = PIL.Image.open(first_image_path).convert("RGB")
    image_prompt = [ImagePrompt(image=pil_image, frame_pos=0)]
    if last_image_path:
        pil_last = PIL.Image.open(last_image_path).convert("RGB")
        image_prompt.append(ImagePrompt(image=pil_last, frame_pos=num_frames - 1))
    negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

    pipeline = WanPipelineI2V.create_pipeline(
        mesh_device=mesh_device,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_links=num_links,
        dynamic_load=dynamic_load,
        topology=topology,
        is_fsdp=is_fsdp,
        target_height=height,
        target_width=width,
        num_frames=num_frames,
    )

    prompt = os.environ.get("WAN_PROMPT", "The cat in the hat runs up the hill to the house.")

    def run(*, prompt, number, seed):
        logger.info(f"Running inference with prompt: '{prompt}'")
        logger.info(f"Parameters: {height}x{width}, {num_frames} frames, {num_inference_steps} steps")

        with torch.no_grad():
            result = pipeline(
                prompt=prompt,
                image_prompt=image_prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                seed=seed,
                guidance_scale=guidance_scale,
                guidance_scale_2=guidance_scale_2,
            )

        if hasattr(result, "frames"):
            frames = result.frames
        else:
            frames = result[0] if isinstance(result, tuple) else result

        logger.info(f"Inference completed successfully")
        logger.info(f"  Output shape: {frames.shape if hasattr(frames, 'shape') else 'Unknown'}")
        logger.info(f"  Output type: {type(frames)}")

        if isinstance(frames, np.ndarray):
            logger.info(f"  Video data range: [{frames.min():.3f}, {frames.max():.3f}]")
        elif isinstance(frames, torch.Tensor):
            logger.info(f"  Video data range: [{frames.min().item():.3f}, {frames.max().item():.3f}]")

        # Remove batch dimension
        frames = frames[0]
        output_filename = output_path_env or f"wan_i2v_{width}x{height}_{number}.mp4"
        del result
        gc.collect()
        try:
            import av

            video_u8 = frames
            if video_u8.dtype != np.uint8:
                scale = 255.0 if float(video_u8.max()) <= 1.0 else 1.0
                video_u8 = np.clip(video_u8 * scale, 0, 255).round().astype(np.uint8)
            video_u8 = np.ascontiguousarray(video_u8)
            T, H, W, _ = video_u8.shape
            # yuv420p requires even dimensions
            H2, W2 = H - (H % 2), W - (W % 2)
            if (H2, W2) != (H, W):
                video_u8 = np.ascontiguousarray(video_u8[:, :H2, :W2, :])
            from fractions import Fraction as _Fraction

            _tb = _Fraction(1, 16)
            container = av.open(output_filename, mode="w")
            try:
                stream = container.add_stream("h264", rate=16)
                stream.width = W2
                stream.height = H2
                stream.pix_fmt = "yuv420p"
                stream.time_base = _tb
                stream.codec_context.time_base = _tb
                stream.codec_context.framerate = _Fraction(16, 1)
                for i_pts, frame_rgb in enumerate(video_u8):
                    vf = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
                    vf.pts = i_pts
                    vf.time_base = _tb
                    for packet in stream.encode(vf):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            finally:
                container.close()
            logger.info(f"Saved video to: {output_filename}")
        except Exception as e:
            logger.info(f"Video save failed: {type(e).__name__}: {e}")

    if no_prompt:
        run(prompt=prompt, number=0, seed=seed_env)
    else:
        for i in itertools.count():
            new_prompt = input("Enter the input prompt, or q to exit: ")
            if new_prompt:
                prompt = new_prompt
            if prompt[0] == "q":
                break
            run(prompt=prompt, number=i, seed=i)
