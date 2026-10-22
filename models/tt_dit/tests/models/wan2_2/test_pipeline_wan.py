# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import itertools
import os

import numpy as np
import PIL
import pytest
import torch
from loguru import logger
from PIL import Image

import ttnn
from models.tt_dit.parallel.config import DiTParallelConfig, EncoderParallelConfig, VaeHWParallelConfig
from models.tt_dit.pipelines.events import log_event_section
from models.tt_dit.pipelines.wan.pipeline_wan import WanPipeline, WanPipelineConfig
from models.tt_dit.pipelines.wan.pipeline_wan_i2v import ImagePrompt, WanPipelineI2V
from models.tt_dit.pipelines.wan.quant_config import QuantConfig, set_quant_config
from models.tt_dit.tests.dataset_eval.clip_encoder import CLIPEncoder
from models.tt_dit.utils.vbench import assert_vbench_quality

from ....utils.test import (
    line_params_req_exact_devices,
    ring_params_8k_req_exact_devices,
    ring_params_req_exact_devices,
    skip_if_unsupported_num_links,
)
from .common import check_first_frame_matches_seed, check_output_sanity, create_fractal_image

PROMPTS = {
    "t2v": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    "i2v": "The cat in the hat runs up the hill to the house.",
}

# These thresholds are deliberately loose, they're just to catch a broken pipeline
# (garbage, static or blurry output)
#
# This test runs a single prompt at a single seed, and is therefore too sensitive
# to numeric fluctuation to support strict thresholds; the multi-prompt, multi-seed gate
# is in test_performance_wan.py. RUN_VBENCH=0 skips this one where that gate runs instead.
VBENCH_THRESHOLDS = {
    "t2v": {
        720: {
            "subject_consistency": 0.8,
            "background_consistency": 0.8,
            "motion_smoothness": 0.9,
            "dynamic_degree": 1.0,
            "imaging_quality": 0.5,
        },
        480: {
            "subject_consistency": 0.8,
            "background_consistency": 0.8,
            "motion_smoothness": 0.9,
            "dynamic_degree": 1.0,
            "imaging_quality": 0.45,
        },
    },
    # I2V uses ONLY subject_consistency + background_consistency. They measure intra-video feature
    # consistency (does the content stay coherent frame-to-frame), which still catches a pipeline
    # that emits incoherent/garbage output. The other VBench dimensions are dropped because they're
    # meaningless for this test's synthetic seed: the CI seed is a fractal, so `dynamic_degree`
    # (RAFT optical-flow "is it moving") reads ~0 (a fractal can't be animated naturally) and
    # `imaging_quality` (MUSIQ, trained on natural photos) reads ~0.30 regardless of correctness --
    # neither reflects a real defect here. Floors are provisional: set below one observed i2v run
    # (subject≈0.75, background≈0.89); tighten/loosen as more runs accrue.
    "i2v": {
        720: {
            "subject_consistency": 0.70,
            "background_consistency": 0.80,
        },
        480: {
            "subject_consistency": 0.70,
            "background_consistency": 0.82,
        },
    },
}


@pytest.mark.parametrize(
    "no_prompt",
    [{"1": True, "0": False}.get(os.environ.get("NO_PROMPT"), False)],
)
@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, device_params, topology, is_fsdp, quant_config_name",
    [
        [(2, 2), (2, 2), 0, 1, 2, False, line_params_req_exact_devices, ttnn.Topology.Linear, True, None],
        [(2, 4), (2, 4), 0, 1, 1, True, line_params_req_exact_devices, ttnn.Topology.Linear, True, None],
        [(2, 4), (2, 4), 1, 0, 2, True, line_params_req_exact_devices, ttnn.Topology.Linear, False, None],
        [(4, 8), (4, 8), 1, 0, 4, False, ring_params_req_exact_devices, ttnn.Topology.Ring, True, None],
        [(4, 8), (4, 8), 1, 0, 2, False, line_params_req_exact_devices, ttnn.Topology.Linear, False, None],
        [(4, 8), (4, 8), 1, 0, 2, False, ring_params_req_exact_devices, ttnn.Topology.Ring, False, None],
        [(4, 32), (4, 32), 1, 0, 2, False, ring_params_8k_req_exact_devices, ttnn.Topology.Ring, False, None],
        [(2, 4), (2, 4), 0, 1, 1, True, line_params_req_exact_devices, ttnn.Topology.Linear, True, "all_bf8_lofi"],
    ],
    ids=[
        "2x2sp0tp1nl2_linear_is_fsdp1",
        "2x4sp0tp1nl1_linear_is_fsdp1",
        "2x4sp1tp0nl2_linear_is_fsdp0",  # BH on 2x4 with dynamic_load to avoid init-time DRAM OOM
        "4x8sp1tp0nl4_ring_is_fsdp1",  # WH (ring) on 4x8
        "4x8sp1tp0nl2_linear_is_fsdp0",  # BH (linear) on 4x8
        "4x8sp1tp0nl2_ring_is_fsdp0",  # BH (ring) on 4x8
        "4x32sp1tp0nl2_ring_is_fsdp0",  # BH (ring) on 4x32 quad galaxy
        "2x4sp0tp1nl1_linear_is_fsdp1_bf8_lofi",  # FSDP on 2x4 with bf8 weights+activations, LoFi linear, bf8 HiFi2 SDPA
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
@pytest.mark.parametrize("model_type", ["t2v", "i2v"], ids=["t2v", "i2v"])
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
    quant_config_name,
    model_type,
    no_prompt,
    request,
):
    # The BH 4x8 ring and the bf8 LoFi rows were only brought up for T2V.
    if model_type == "i2v" and (
        quant_config_name is not None or (mesh_shape == (4, 8) and topology == ttnn.Topology.Ring and not is_fsdp)
    ):
        pytest.skip("not brought up for I2V")

    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    skip_if_unsupported_num_links(mesh_device, num_links)

    if (not is_fsdp) and (not ttnn.device.is_blackhole()):
        pytest.skip("FSDP=False unsupported on non-blackhole systems due to memory constraints")

    if tuple(mesh_device.shape) == (2, 2) and height > 480:
        pytest.skip("2x2 lacks the DRAM capacity for 720p")

    num_frames = 81
    num_inference_steps = 40

    if model_type == "i2v":
        pipeline_cls = WanPipelineI2V
        seed_image = create_fractal_image(width, height) if no_prompt else PIL.Image.open("./prompt_image.png")
        image_prompt = [ImagePrompt(image=seed_image, frame_pos=0)]
        guidance_scale, guidance_scale_2 = 3.5, 3.5
    else:
        pipeline_cls = WanPipeline
        image_prompt = None
        guidance_scale, guidance_scale_2 = 4.0, 3.0

    h_factor = tuple(mesh_device.shape)[tp_axis]
    w_factor = tuple(mesh_device.shape)[sp_axis]
    parallel_config = DiTParallelConfig.from_tuples(cfg=(1, 0), sp=(w_factor, sp_axis), tp=(h_factor, tp_axis))
    vae_parallel_config = VaeHWParallelConfig.from_tuples(height=(h_factor, tp_axis), width=(w_factor, sp_axis))
    encoder_parallel_config = EncoderParallelConfig.from_tuple((h_factor, tp_axis))

    pipeline = pipeline_cls(
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
            model_type=model_type,
            checkpoint_name=(
                "Wan-AI/Wan2.2-I2V-A14B-Diffusers" if model_type == "i2v" else "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
            ),
            height=height,
            width=width,
            num_frames=num_frames,
        ),
    )

    if quant_config_name is not None:
        qc = getattr(QuantConfig, quant_config_name)()
        set_quant_config(pipeline, qc)

    prompt = PROMPTS[model_type]

    def output_filename(number):
        return f"wan_{model_type}_{width}x{height}_{number}.mp4"

    def run(*, prompt, number, seed):
        logger.info(f"Running {model_type} inference with prompt: '{prompt}'")
        logger.info(f"Parameters: {height}x{width}, {num_frames} frames, {num_inference_steps} steps")

        with torch.no_grad():
            frames = pipeline(
                prompts=[prompt],
                image_prompt=image_prompt,
                num_inference_steps=num_inference_steps,
                seed=seed,
                guidance_scale=guidance_scale,
                guidance_scale_2=guidance_scale_2,
                output_type="uint8",
                on_event=log_event_section,
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
            if model_type == "i2v":
                check_output_sanity(frames, num_frames=num_frames, height=height, width=width)
                check_first_frame_matches_seed(frames, seed_image=seed_image, width=width, height=height)
            try:
                from models.tt_dit.utils.video import export_to_video

                export_to_video(frames, output_filename(number), fps=16)
                logger.info(f"Saved video to: {output_filename(number)}")
            except ImportError:
                logger.info("Could not export video - imageio_ffmpeg not available")
        else:
            logger.info(f"Skipping video export on rank {ttnn.distributed_context_get_rank()}")

        return frames

    def check_output_with_clip(prompt, frames, clip_threshold=36.0):
        # T2V only: the I2V CI seed is a fractal, which CLIP can't relate to a natural-language prompt.
        if model_type != "t2v":
            return
        if int(ttnn.distributed_context_get_rank()) == 0:
            # Sample ~8 evenly-spaced frames from the video
            total_frames = frames.shape[0]
            indices = np.linspace(0, total_frames - 1, min(8, total_frames), dtype=int)
            sampled = [frames[i] for i in indices]

            clip_encoder = CLIPEncoder()
            scores = []
            for frame_data in sampled:
                if isinstance(frame_data, torch.Tensor):
                    frame_data = frame_data.numpy()
                pil_img = Image.fromarray(frame_data.astype(np.uint8))
                score = clip_encoder.get_clip_score(prompt, pil_img).item() * 100.0
                scores.append(score)

            clip_min = min(scores)
            clip_max = max(scores)
            clip_mean = sum(scores) / len(scores)
            logger.info(f"CLIP scores: min={clip_min:.2f}, max={clip_max:.2f}, mean={clip_mean:.2f}")

            assert clip_mean >= clip_threshold, (
                f"Mean CLIP score {clip_mean:.2f} is below threshold {clip_threshold:.2f}. "
                f"Per-frame scores: {[f'{s:.2f}' for s in scores]}"
            )

    def check_output_with_vbench(prompt, number):
        if os.environ.get("RUN_VBENCH", "1") == "0":
            logger.info("RUN_VBENCH=0, skipping VBench quality gate")
            return
        if int(ttnn.distributed_context_get_rank()) == 0:
            thresholds = VBENCH_THRESHOLDS[model_type][height]
            assert_vbench_quality(output_filename(number), prompt=prompt, thresholds=thresholds)

    if no_prompt:
        frames = run(prompt=prompt, number=0, seed=42)
        check_output_with_clip(prompt, frames)
        check_output_with_vbench(prompt, 0)
    else:
        for i in itertools.count():
            new_prompt = input("Enter the input prompt, or q to exit: ")
            if new_prompt:
                prompt = new_prompt
            if prompt[0] == "q":
                break
            frames = run(prompt=prompt, number=i, seed=i)
            check_output_with_clip(prompt, frames)
            check_output_with_vbench(prompt, i)
