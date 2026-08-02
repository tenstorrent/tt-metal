# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import itertools
import os

import numpy as np
import pytest
import torch
from loguru import logger
from PIL import Image

import ttnn
from models.tt_dit.parallel.config import DiTParallelConfig, EncoderParallelConfig, VaeHWParallelConfig
from models.tt_dit.pipelines.events import log_event_section
from models.tt_dit.pipelines.wan.pipeline_wan import WanPipeline, WanPipelineConfig
from models.tt_dit.pipelines.wan.quant_config import QuantConfig, set_quant_config
from models.tt_dit.tests.dataset_eval.clip_encoder import CLIPEncoder
from models.tt_dit.utils.vbench import assert_vbench_quality

from ....utils.test import line_params_req_exact_devices, ring_params_req_exact_devices, skip_if_unsupported_num_links


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
        [(4, 32), (4, 32), 1, 0, 2, False, ring_params_req_exact_devices, ttnn.Topology.Ring, False, None],
        [(2, 4), (2, 4), 0, 1, 1, True, line_params_req_exact_devices, ttnn.Topology.Linear, True, "all_bf8_lofi"],
    ],
    ids=[
        "2x2sp0tp1nl2_linear_is_fsdp1",
        "2x4sp0tp1nl1_linear_is_fsdp1",
        "2x4sp1tp0nl2_linear_is_fsdp0",  # BH on 2x4
        "4x8sp1tp0nl4_ring_is_fsdp1",  # WH (ring) on 4x8
        "4x8sp1tp0nl2_linear_is_fsdp0",  # BH (linear) on 4x8
        "4x8sp1tp0nl2_ring_is_fsdp0",  # BH (ring) on 4x8
        "4x32sp1tp0nl2_ring_is_fsdp0",
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
    no_prompt,
):
    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    skip_if_unsupported_num_links(mesh_device, num_links)

    if (not is_fsdp) and (not ttnn.device.is_blackhole()):
        pytest.skip("FSDP=False unsupported on non-blackhole systems due to memory constraints")

    num_frames = 81
    num_inference_steps = 40

    h_factor = tuple(mesh_device.shape)[tp_axis]
    w_factor = tuple(mesh_device.shape)[sp_axis]
    parallel_config = DiTParallelConfig.from_tuples(cfg=(1, 0), sp=(w_factor, sp_axis), tp=(h_factor, tp_axis))
    vae_parallel_config = VaeHWParallelConfig.from_tuples(height=(h_factor, tp_axis), width=(w_factor, sp_axis))
    encoder_parallel_config = EncoderParallelConfig.from_tuple((h_factor, tp_axis))

    pipeline = WanPipeline(
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
            checkpoint_name="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            height=height,
            width=width,
            num_frames=num_frames,
        ),
    )

    if quant_config_name is not None:
        qc = getattr(QuantConfig, quant_config_name)()
        set_quant_config(pipeline, qc)

    prompt = "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."

    def run(*, prompt, number, seed):
        logger.info(f"Running inference with prompt: '{prompt}'")
        logger.info(f"Parameters: {height}x{width}, {num_frames} frames, {num_inference_steps} steps")

        with torch.no_grad():
            frames = pipeline(
                prompts=[prompt],
                num_inference_steps=num_inference_steps,
                seed=seed,
                guidance_scale=4.0,
                guidance_scale_2=3.0,
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
        output_filename = f"wan_t2v_{width}x{height}_{number}.mp4"
        if int(ttnn.distributed_context_get_rank()) == 0:
            try:
                from models.tt_dit.utils.video import export_to_video

                export_to_video(frames, output_filename, fps=16)
                logger.info(f"Saved video to: {output_filename}")
            except ImportError:
                logger.info("Could not export video - imageio_ffmpeg not available")
        else:
            logger.info(f"Skipping video export on rank {ttnn.distributed_context_get_rank()}")

        return frames

    def check_output_with_clip(prompt, frames, clip_threshold=36.0):
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

    vbench_thresholds_by_height = {
        720: {
            "subject_consistency": 0.92,
            "background_consistency": 0.93,
            "motion_smoothness": 0.955,
            "dynamic_degree": 1.0,
            "imaging_quality": 0.645,
        },
        480: {
            "subject_consistency": 0.94,
            "background_consistency": 0.96,
            "motion_smoothness": 0.97,
            "dynamic_degree": 1.0,
            "imaging_quality": 0.545,
        },
    }

    def check_output_with_vbench(prompt, number):
        if int(ttnn.distributed_context_get_rank()) == 0:
            output_filename = f"wan_t2v_{width}x{height}_{number}.mp4"
            thresholds = vbench_thresholds_by_height[height]
            assert_vbench_quality(output_filename, prompt=prompt, thresholds=thresholds)

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
