# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import json
import os
import pytest
import statistics
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

    videos = []
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
        ).frames[0]

        videos.append(frames)

    clip = CLIPEncoder()
    clip_scores = [
        [100 * clip.get_clip_score(prompts[i], img).item() for img in video] for i, video in enumerate(videos)
    ]
    clip_score_stats = [(min(c), max(c), statistics.mean(c), statistics.stdev(c)) for c in clip_scores]

    for p, c in zip(prompt, clip_score_stats):
        print(f"PROMPT: {p} STATISTICS: {c}")
    # average_clip_score = sum(clip_scores) / len(clip_scores)

    # fvd_score = calculate_fvd_score(videos, coco_statistics_path)

    # print(f"FID score: {fid_score}")
    # print(f"Average CLIP Score: {average_clip_score}")
    # print(f"Standard Deviation of CLIP Scores: {deviation_clip_score}")

    data = {
        "model": "wan",
        "metadata": {
            "device": os.environ.get("MESH_DEVICE", "unknown"),
            "mesh_shape": mesh_device.shape,
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
                # "average_denoising_time": profiler.get("denoising_loop"),
                # "average_vae_time": profiler.get("vae_decode"),
                # "average_inference_time": profiler.get("denoising_loop") + profiler.get("vae_decode"),
                # "min_inference_time": min(
                #    i + j for i, j in zip(profiler.times["denoising_loop"], profiler.times["vae_decode"])
                # ),
                # "max_inference_time": max(
                #    i + j for i, j in zip(profiler.times["denoising_loop"], profiler.times["vae_decode"])
                # ),
                # "average_clip": average_clip_score,
                # "deviation_clip": deviation_clip_score,
                # "fvd_score": fvd_score,
            }
        ],
    }

    os.makedirs(OUT_ROOT, exist_ok=True)
    with open(f"{OUT_ROOT}/{RESULTS_FILE_NAME}", "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"test results saved to {OUT_ROOT}/{RESULTS_FILE_NAME}")

    # Synchronize devices
    ttnn.synchronize_device(mesh_device)
