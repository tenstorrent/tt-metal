# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import json
import os
import pytest
import statistics
import torch
import ttnn
from loguru import logger
from models.experimental.tt_dit.pipelines.wan.pipeline_wan import WanPipeline
from models.experimental.tt_dit.parallel.config import DiTParallelConfig, VaeHWParallelConfig, ParallelFactor
from ....utils.test import line_params, ring_params
from models.experimental.tt_dit.tests.dataset_eval.utils.clip_encoder import CLIPEncoder
import models.experimental.tt_dit.tests.dataset_eval.utils.data_helper as data_helper


OUT_ROOT, RESULTS_FILE_NAME = "test_reports", "wan_test_results.json"


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, device_params, topology",
    [
        [(2, 4), (2, 4), 0, 1, 1, True, line_params, ttnn.Topology.Linear],
        # WH (ring) on 4x8
        [(4, 8), (4, 8), 1, 0, 4, False, ring_params, ttnn.Topology.Ring],
        # BH (linear) on 4x8
        [(4, 8), (4, 8), 1, 0, 2, False, line_params, ttnn.Topology.Linear],
    ],
    ids=[
        "2x4sp0tp1",
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
@pytest.mark.parametrize("captions_path", ["models/experimental/stable_diffusion_xl_base/coco_data/captions.tsv"])
@pytest.mark.parametrize("coco_statistics_path", ["models/experimental/stable_diffusion_xl_base/coco_data/val2014.npz"])
def test_accuracy_wan(
    mesh_device,
    mesh_shape,
    sp_axis,
    tp_axis,
    num_links,
    dynamic_load,
    device_params,
    topology,
    width,
    height,
    captions_path,
    coco_statistics_path,
    evaluation_range,
):
    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    sp_factor = mesh_device.shape[sp_axis]
    tp_factor = mesh_device.shape[tp_axis]

    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=sp_factor),
        cfg_parallel=None,
    )
    vae_parallel_config = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[sp_axis], mesh_axis=sp_axis),
        width_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[tp_axis], mesh_axis=tp_axis),
    )

    pipeline = WanPipeline(
        mesh_device=mesh_device,
        parallel_config=parallel_config,
        vae_parallel_config=vae_parallel_config,
        num_links=num_links,
        use_cache=True,
        boundary_ratio=0.875,
        dynamic_load=dynamic_load,
        topology=topology,
    )

    num_frames = 81
    num_inference_steps = 40
    guidance_scale = 3.0
    guidance_scale_2 = 4.0

    generator = torch.Generator("cpu").manual_seed(0)

    start_from, num_prompts = evaluation_range
    prompts = data_helper.get_prompts(captions_path, start_from, num_prompts)
    logger.info(f"start inference from prompt index: {start_from} to {start_from + num_prompts}")
    negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

    videos = []
    all_timings = []
    for prompt in prompts:
        logger.info(f"Processing prompt: {prompt}")
        logger.info(f"Prompt number: {start_from + len(videos) + 1}")

        frames = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            guidance_scale_2=guidance_scale_2,
            output_type="pil",
        ).frames[0]

        videos.append(frames)
        all_timings.append(pipeline.timing_data)

    clip = CLIPEncoder()
    clip_scores = [
        [100 * clip.get_clip_score(prompts[i], img).item() for img in video] for i, video in enumerate(videos)
    ]
    clip_score_stats = [(min(c), max(c), statistics.mean(c), statistics.stdev(c)) for c in clip_scores]

    for p, c in zip(prompts, clip_score_stats):
        logger.info(f"Prompt: {p}, CLIP statistics (min, max, mean, stddev): {c}")

    video_mean_scores = [c[2] for c in clip_score_stats]
    min_clip = min(video_mean_scores)
    max_clip = max(video_mean_scores)
    mean_clip = statistics.mean(video_mean_scores)
    stddev_clip = statistics.stdev(video_mean_scores)

    mean_text_encoder_time = statistics.mean([t["text_encoder"] for t in all_timings])
    mean_denoising_time = statistics.mean([t["denoising"] for t in all_timings])
    mean_vae_time = statistics.mean([t["vae"] for t in all_timings])
    mean_total_time = statistics.mean([t["total"] for t in all_timings])

    data = {
        "model": "wan",
        "metadata": {
            "device": os.environ.get("MESH_DEVICE", "unknown"),
            "mesh_shape": tuple(mesh_device.shape),
            "start_from": start_from,
            "num_prompts": num_prompts,
            "image_width": width,
            "image_height": height,
            "guidance_scale": guidance_scale,
            "guidance_scale_2": guidance_scale_2,
            "num_inference_steps": num_inference_steps,
            "num_frames": num_frames,
            "sp_factor": sp_factor,
            "tp_factor": tp_factor,
        },
        "benchmarks_summary": [
            {
                "min_clip": min_clip,
                "max_clip": max_clip,
                "mean_clip": mean_clip,
                "stddev_clip": stddev_clip,
                "mean_text_encoder_time": mean_text_encoder_time,
                "mean_denoising_time": mean_denoising_time,
                "mean_vae_time": mean_vae_time,
                "mean_total_time": mean_total_time,
            }
        ],
    }

    os.makedirs(OUT_ROOT, exist_ok=True)
    with open(f"{OUT_ROOT}/{RESULTS_FILE_NAME}", "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"test results saved to {OUT_ROOT}/{RESULTS_FILE_NAME}")

    # Synchronize devices
    ttnn.synchronize_device(mesh_device)
