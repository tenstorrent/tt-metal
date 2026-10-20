# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import itertools
import math
import os
from pathlib import Path

import numpy as np
import PIL
import pytest
import torch
from loguru import logger
from PIL import Image

import ttnn
from models.tt_dit.parallel.config import DiTParallelConfig, EncoderParallelConfig, VaeHWParallelConfig
from models.tt_dit.pipelines.wan.pipeline_wan import WanPipelineConfig
from models.tt_dit.pipelines.wan.pipeline_wan_i2v import ImagePrompt, WanPipelineI2V
from models.tt_dit.utils.vbench import assert_vbench_quality

from ....utils.test import (
    line_params_req_exact_devices,
    ring_params_8k_req_exact_devices,
    ring_params_req_exact_devices,
    skip_if_unsupported_num_links,
)
from .common import check_first_frame_matches_seed, check_output_sanity


# The official I2V example (Wan-Video/Wan2.2 README and wan/configs/wan_i2v_A14B.py). Image, prompt,
# size, flow shift, boundary and seed match the diffusers reference run, and so does the noise: both
# draw it from the CPU generator.
OFFICIAL_IMAGE_URL = (
    "https://raw.githubusercontent.com/Wan-Video/Wan2.2/1ea34ff48f87168174e12956e200b1d908b1c5ff/examples/i2v_input.JPG"
)
OFFICIAL_IMAGE_SHA256 = "077e3d965090c9028c69c00931675f42e1acc815c6eb450ab291b3b72d211a8e"
OFFICIAL_PROMPT = (
    "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline "
    "gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring "
    "crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally "
    "relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's "
    "intricate details and the refreshing atmosphere of the seaside."
)
OFFICIAL_FLOW_SHIFT = 5.0
OFFICIAL_BOUNDARY_RATIO = 0.9  # the checkpoint's model_index.json
OFFICIAL_SEED = 0


def official_image() -> Image.Image:
    path = Path(torch.hub.get_dir()) / "wan2.2_i2v_input.jpg"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(OFFICIAL_IMAGE_URL, str(path), hash_prefix=OFFICIAL_IMAGE_SHA256)
    return Image.open(path).convert("RGB")


def official_size(max_area: int, image: Image.Image) -> tuple[int, int]:
    """Wan's image2video.py: the image's aspect ratio at max_area, in whole latent patches."""
    aspect = image.height / image.width
    lat_h = round(math.sqrt(max_area * aspect) // 8 // 2 * 2)
    lat_w = round(math.sqrt(max_area / aspect) // 8 // 2 * 2)
    return lat_w * 8, lat_h * 8


@pytest.mark.parametrize(
    "no_prompt",
    [{"1": True, "0": False}.get(os.environ.get("NO_PROMPT"), False)],
)
@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, device_params, topology, is_fsdp",
    [
        [(2, 2), (2, 2), 0, 1, 2, False, line_params_req_exact_devices, ttnn.Topology.Linear, True],
        [(2, 4), (2, 4), 0, 1, 1, True, line_params_req_exact_devices, ttnn.Topology.Linear, True],
        [(2, 4), (2, 4), 1, 0, 2, True, line_params_req_exact_devices, ttnn.Topology.Linear, False],
        [(4, 8), (4, 8), 1, 0, 4, False, ring_params_req_exact_devices, ttnn.Topology.Ring, True],
        [(4, 8), (4, 8), 1, 0, 2, False, line_params_req_exact_devices, ttnn.Topology.Linear, False],
        [(4, 32), (4, 32), 1, 0, 2, False, ring_params_8k_req_exact_devices, ttnn.Topology.Ring, False],
    ],
    ids=[
        "2x2sp0tp1nl2_linear_is_fsdp1",
        "2x4sp0tp1nl1_linear_is_fsdp1",
        "2x4sp1tp0nl2_linear_is_fsdp0",  # BH on 2x4 with dynamic_load to avoid init-time DRAM OOM
        "4x8sp1tp0nl4_ring_is_fsdp1",  # WH (ring) on 4x8
        "4x8sp1tp0nl2_linear_is_fsdp0",  # BH (linear) on 4x8
        "4x32sp1tp0nl2_ring_is_fsdp0",  # BH (ring) on 4x32 quad galaxy
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "max_area",
    [
        832 * 480,
        1280 * 720,
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
    max_area,
    is_fsdp,
    no_prompt,
    request,
):
    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    skip_if_unsupported_num_links(mesh_device, num_links)

    if no_prompt:
        test_image = official_image()
    else:
        test_image = PIL.Image.open("./prompt_image.png").convert("RGB")
    width, height = official_size(max_area, test_image)
    test_image = test_image.resize((width, height))
    image_prompt = [ImagePrompt(image=test_image, frame_pos=0)]
    negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

    num_frames = 81
    num_inference_steps = 40
    guidance_scale = 3.5
    guidance_scale_2 = 3.5

    h_factor = tuple(mesh_device.shape)[tp_axis]
    w_factor = tuple(mesh_device.shape)[sp_axis]
    parallel_config = DiTParallelConfig.from_tuples(cfg=(1, 0), sp=(w_factor, sp_axis), tp=(h_factor, tp_axis))
    vae_parallel_config = VaeHWParallelConfig.from_tuples(height=(h_factor, tp_axis), width=(w_factor, sp_axis))
    encoder_parallel_config = EncoderParallelConfig.from_tuple((h_factor, tp_axis))

    pipeline = WanPipelineI2V(
        device=mesh_device,
        config=WanPipelineConfig.default(
            mesh_shape=mesh_device.shape,
            dit_parallel_config=parallel_config,
            vae_parallel_config=vae_parallel_config,
            encoder_parallel_config=encoder_parallel_config,
            num_links=num_links,
            dynamic_load=dynamic_load,
            topology=topology,
            is_fsdp=is_fsdp,
            checkpoint_name="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            model_type="i2v",
            height=height,
            width=width,
            num_frames=num_frames,
        ),
    )

    prompt = OFFICIAL_PROMPT

    def run(*, prompt, number, seed):
        logger.info(f"Running inference with prompt: '{prompt}'")
        logger.info(f"Parameters: {height}x{width}, {num_frames} frames, {num_inference_steps} steps")

        with torch.no_grad():
            frames = pipeline(
                prompts=[prompt],
                negative_prompts=[negative_prompt],
                image_prompt=image_prompt,
                num_inference_steps=num_inference_steps,
                seed=seed,
                guidance_scale=guidance_scale,
                guidance_scale_2=guidance_scale_2,
                flow_shift=OFFICIAL_FLOW_SHIFT,
                boundary_ratio=OFFICIAL_BOUNDARY_RATIO,
                output_type="uint8",
            )

        logger.info(f"Inference completed successfully")
        logger.info(f"  Output shape: {frames.shape if hasattr(frames, 'shape') else 'Unknown'}")
        logger.info(f"  Output type: {type(frames)}")

        if isinstance(frames, np.ndarray):
            logger.info(f"  Video data range: [{frames.min():.3f}, {frames.max():.3f}]")
        elif isinstance(frames, torch.Tensor):
            logger.info(f"  Video data range: [{frames.min().item():.3f}, {frames.max().item():.3f}]")

        # Remove batch dimension
        frames = frames[0]
        if int(ttnn.distributed_context_get_rank()) == 0:
            check_output_sanity(frames, num_frames=num_frames, height=height, width=width)
            check_first_frame_matches_seed(frames, seed_image=test_image, width=width, height=height)
        output_filename = f"wan_i2v_{width}x{height}_{number}.mp4"
        try:
            from models.tt_dit.utils.video import export_to_video

            export_to_video(frames, output_filename, fps=16)
            logger.info(f"Saved video to: {output_filename}")
        except ImportError:
            logger.info("Could not export video - imageio_ffmpeg not available")

    # VBench gate for I2V uses ONLY subject_consistency + background_consistency. They measure
    # intra-video feature consistency (does the content stay coherent frame-to-frame), which still
    # catches a pipeline that emits incoherent/garbage output. The other VBench dimensions were dropped
    # while the CI seed was a synthetic fractal, for which `dynamic_degree` (RAFT optical-flow "is it
    # moving") reads ~0 and `imaging_quality` (MUSIQ, trained on natural photos) reads ~0.30 regardless
    # of correctness. Floors are provisional: set below one observed i2v run on the fractal seed
    # (subject≈0.75, background≈0.89); tighten/loosen as runs on the official image accrue.
    vbench_thresholds_by_area = {
        (1280 * 720): {
            "subject_consistency": 0.70,
            "background_consistency": 0.80,
        },
        (832 * 480): {
            "subject_consistency": 0.70,
            "background_consistency": 0.82,
        },
    }

    def check_output_with_vbench(prompt, number):
        if int(ttnn.distributed_context_get_rank()) == 0:
            output_filename = f"wan_i2v_{width}x{height}_{number}.mp4"
            thresholds = vbench_thresholds_by_area[max_area]
            assert_vbench_quality(output_filename, prompt=prompt, thresholds=thresholds)

    if no_prompt:
        run(prompt=prompt, number=0, seed=OFFICIAL_SEED)
        check_output_with_vbench(prompt, 0)
    else:
        for i in itertools.count():
            new_prompt = input("Enter the input prompt, or q to exit: ")
            if new_prompt:
                prompt = new_prompt
            if prompt[0] == "q":
                break
            run(prompt=prompt, number=i, seed=i)
            check_output_with_vbench(prompt, i)
