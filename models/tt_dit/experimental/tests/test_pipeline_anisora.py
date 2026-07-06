# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import itertools
import os

import numpy as np
import PIL
import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.experimental.pipelines.pipeline_anisora import AniSoraPipeline
from models.tt_dit.pipelines.wan.pipeline_wan_i2v import ImagePrompt
from models.tt_dit.utils.test import ring_params


@pytest.mark.parametrize(
    "no_prompt",
    [{"1": True, "0": False}.get(os.environ.get("NO_PROMPT"), False)],
)
@pytest.mark.parametrize(
    "mesh_device, mesh_shape, num_links, dynamic_load, device_params, topology, is_fsdp",
    [
        # BH Galaxy 4x8 Ring — sole supported config in the first AniSora PR.
        [(4, 8), (4, 8), 2, False, ring_params, ttnn.Topology.Ring, False],
    ],
    ids=["bh_4x8sp1tp0_ring"],
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

    pil_image = PIL.Image.open(os.environ.get("PROMPT_IMAGE", "./prompt_image.png"))
    image_prompt = [ImagePrompt(image=pil_image, frame_pos=0)]
    negative_prompt = ""

    # AniSora upstream defaults (anisoraV3.2/wan/configs/wan_i2v_A14B.py):
    #   sample_steps=40, boundary=0.9, sample_guide_scale=(3.5, 3.5).
    num_frames = 81
    num_inference_steps = int(os.environ.get("NUM_STEPS", "40"))
    guidance_scale = 3.5
    guidance_scale_2 = 3.5

    pipeline = AniSoraPipeline.create_pipeline(
        mesh_device=mesh_device,
        num_links=num_links,
        dynamic_load=dynamic_load,
        topology=topology,
        is_fsdp=is_fsdp,
        height=height,
        width=width,
        num_frames=num_frames,
    )

    prompt = "An anime girl smiling, soft lighting, cinematic."

    def run(*, prompt, number, seed):
        logger.info(f"Running AniSora inference with prompt: '{prompt}'")
        logger.info(f"Parameters: {height}x{width}, {num_frames} frames, {num_inference_steps} steps")

        with torch.no_grad():
            result = pipeline(
                prompts=[prompt],
                image_prompt=image_prompt,
                negative_prompts=[negative_prompt],
                num_inference_steps=num_inference_steps,
                seed=seed,
                guidance_scale=guidance_scale,
                guidance_scale_2=guidance_scale_2,
                output_type="uint8",
            )

        if hasattr(result, "frames"):
            frames = result.frames
        else:
            frames = result[0] if isinstance(result, tuple) else result

        logger.info("AniSora inference completed successfully")
        logger.info(f"  Output shape: {frames.shape if hasattr(frames, 'shape') else 'Unknown'}")
        logger.info(f"  Output type: {type(frames)}")

        if isinstance(frames, np.ndarray):
            logger.info(f"  Video data range: [{frames.min():.3f}, {frames.max():.3f}]")
        elif isinstance(frames, torch.Tensor):
            logger.info(f"  Video data range: [{frames.min().item():.3f}, {frames.max().item():.3f}]")

        frames = frames[0]
        output_filename = f"wan_anisora_i2v_{width}x{height}_{number}.mp4"
        try:
            from models.tt_dit.utils.video import export_to_video

            export_to_video(frames, output_filename, fps=16)
            logger.info(f"Saved video to: {output_filename}")
        except ImportError:
            logger.info("Could not export video - imageio_ffmpeg not available")

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


# ---------------------------------------------------------------------------
# Random-weight smoke test
# ---------------------------------------------------------------------------
# Runs the full pipeline end-to-end without downloading the AniSora safetensors
# or the two ~28 GB transformer subfolders from Wan-AI. Tokenizer / UMT5 text
# encoder / VAE / scheduler still come from Wan-AI/Wan2.2-I2V-A14B-Diffusers
# (~12 GB total) and require TT_DIT_ALLOW_HF_DOWNLOAD=1 the first time.
# Output frames are garbage by design — this only validates compile / mesh /
# shape / CCL plumbing, not numeric quality. Uses 4 sampling steps to keep the
# smoke test short; the 40-step real-weights config is exercised above.


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, num_links, dynamic_load, device_params, topology, is_fsdp",
    [
        [(4, 8), (4, 8), 2, False, ring_params, ttnn.Topology.Ring, False],
    ],
    ids=["bh_4x8sp1tp0_ring"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "width, height",
    [
        (832, 480),
    ],
    ids=["resolution_480p"],
)
def test_pipeline_inference_random_weights(
    mesh_device,
    mesh_shape,
    num_links,
    dynamic_load,
    topology,
    width,
    height,
    is_fsdp,
):
    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    pil_image = PIL.Image.open(os.environ.get("PROMPT_IMAGE", "./prompt_image.png"))
    image_prompt = [ImagePrompt(image=pil_image, frame_pos=0)]

    num_frames = 81
    num_inference_steps = 4

    os.environ["TT_DIT_RANDOM_WEIGHTS"] = "1"
    try:
        pipeline = AniSoraPipeline.create_pipeline(
            mesh_device=mesh_device,
            num_links=num_links,
            dynamic_load=dynamic_load,
            topology=topology,
            is_fsdp=is_fsdp,
            height=height,
            width=width,
            num_frames=num_frames,
        )
    finally:
        os.environ.pop("TT_DIT_RANDOM_WEIGHTS", None)

    prompt = "smoke test — random weights, output is meaningless"
    logger.info(f"Random-weight smoke test: {height}x{width}, {num_frames} frames, {num_inference_steps} steps")

    with torch.no_grad():
        result = pipeline(
            prompts=[prompt],
            image_prompt=image_prompt,
            negative_prompts=[""],
            num_inference_steps=num_inference_steps,
            seed=42,
            guidance_scale=3.5,
            guidance_scale_2=3.5,
            output_type="uint8",
        )

    if hasattr(result, "frames"):
        frames = result.frames
    else:
        frames = result[0] if isinstance(result, tuple) else result

    if isinstance(frames, np.ndarray):
        rng_str = f"[{frames.min():.3f}, {frames.max():.3f}]"
    elif isinstance(frames, torch.Tensor):
        rng_str = f"[{frames.min().item():.3f}, {frames.max().item():.3f}]"
    else:
        rng_str = "n/a"

    shape = frames.shape if hasattr(frames, "shape") else "Unknown"
    logger.info(f"Random-weight smoke test PASSED. Output shape={shape}, range={rng_str}")
