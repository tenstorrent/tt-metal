# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import json
import os
import pytest
import statistics
import torch
import ttnn
from loguru import logger
from ....pipelines.mochi.pipeline_mochi import MochiPipeline as TTMochiPipeline
from ....parallel.config import DiTParallelConfig, MochiVAEParallelConfig, ParallelFactor
from models.experimental.tt_dit.tests.dataset_eval.utils.clip_encoder import CLIPEncoder
import models.experimental.tt_dit.tests.dataset_eval.utils.data_helper as data_helper


OUT_ROOT, RESULTS_FILE_NAME = "test_reports", "mochi_test_results.json"


@pytest.mark.parametrize(
    "mesh_device, sp_axis, tp_axis, vae_mesh_shape, vae_sp_axis, vae_tp_axis, num_links",
    [
        [(2, 4), 0, 1, (1, 8), 0, 1, 1],
        [(4, 8), 1, 0, (4, 8), 0, 1, 4],
    ],
    ids=[
        "2x4sp0tp1",
        "4x8sp1tp0",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize(
    "image_w, image_h, guidance_scale, num_inference_steps, num_frames",
    [
        (848, 480, 3.5, 50, 168),
    ],
)
@pytest.mark.parametrize("captions_path", ["models/experimental/stable_diffusion_xl_base/coco_data/captions.tsv"])
@pytest.mark.parametrize("coco_statistics_path", ["models/experimental/stable_diffusion_xl_base/coco_data/val2014.npz"])
def test_accuracy_mochi(
    mesh_device,
    image_w,
    image_h,
    guidance_scale,
    num_inference_steps,
    num_frames,
    sp_axis,
    tp_axis,
    vae_mesh_shape,
    vae_sp_axis,
    vae_tp_axis,
    num_links,
    captions_path,
    coco_statistics_path,
    evaluation_range,
):
    sp_factor = mesh_device.shape[sp_axis]
    tp_factor = mesh_device.shape[tp_axis]

    # Create parallel config
    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis),
        sequence_parallel=ParallelFactor(factor=sp_factor, mesh_axis=sp_axis),
    )

    if vae_mesh_shape[vae_sp_axis] == 1:
        w_parallel_factor = 1
    else:
        w_parallel_factor = 2

    vae_parallel_config = MochiVAEParallelConfig(
        time_parallel=ParallelFactor(factor=vae_mesh_shape[vae_tp_axis], mesh_axis=vae_tp_axis),
        w_parallel=ParallelFactor(factor=w_parallel_factor, mesh_axis=vae_sp_axis),
        h_parallel=ParallelFactor(factor=vae_mesh_shape[vae_sp_axis] // w_parallel_factor, mesh_axis=vae_sp_axis),
    )
    assert vae_parallel_config.h_parallel.factor * vae_parallel_config.w_parallel.factor == vae_mesh_shape[vae_sp_axis]
    assert vae_parallel_config.h_parallel.mesh_axis == vae_parallel_config.w_parallel.mesh_axis

    # Create the TT Mochi pipeline
    pipeline = TTMochiPipeline(
        mesh_device=mesh_device,
        vae_mesh_shape=vae_mesh_shape,
        parallel_config=parallel_config,
        vae_parallel_config=vae_parallel_config,
        num_links=num_links,
        use_cache=True,
        use_reference_vae=False,
        model_name="genmo/mochi-1-preview",
    )

    generator = torch.Generator("cpu").manual_seed(0)

    start_from, num_prompts = evaluation_range
    prompts = data_helper.get_prompts(captions_path, start_from, num_prompts)
    logger.info(f"start inference from prompt index: {start_from} to {start_from + num_prompts}")

    videos = []
    all_timings = []
    for prompt in prompts:
        logger.info(f"Processing prompt: {prompt}")
        logger.info(f"Prompt number: {start_from + len(videos) + 1}")

        frames = pipeline(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_frames=num_frames,
            height=image_h,
            width=image_w,
            generator=generator,
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
        "model": "mochi",
        "metadata": {
            "device": os.environ.get("MESH_DEVICE", "unknown"),
            "mesh_shape": tuple(mesh_device.shape),
            "vae_mesh_shape": vae_mesh_shape,
            "start_from": start_from,
            "num_prompts": num_prompts,
            "image_width": image_w,
            "image_height": image_h,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "num_frames": num_frames,
            "sp_factor": sp_factor,
            "tp_factor": tp_factor,
            "vae_sp_factor": vae_mesh_shape[sp_axis],
            "vae_tp_factor": vae_mesh_shape[tp_axis],
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
