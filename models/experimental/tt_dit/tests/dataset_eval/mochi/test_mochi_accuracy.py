# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import os
from loguru import logger
import statistics
import json

from models.experimental.tt_dit.tests.dataset_eval.utils.clip_encoder import CLIPEncoder
import ttnn
import models.experimental.tt_dit.tests.dataset_eval.utils.data_helper as data_helper


OUT_ROOT, RESULTS_FILE_NAME = "test_reports", "mochi_test_results.json"


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "vae_mesh_shape", "vae_sp_axis", "vae_tp_axis", "topology", "num_links"),
    [
        pytest.param((2, 4), 0, 1, (1, 8), 0, 1, ttnn.Topology.Linear, 1, id="dit_2x4sp0tp1_vae_1x8sp0tp1"),
        pytest.param((4, 8), 1, 0, (4, 8), 0, 1, ttnn.Topology.Linear, 4, id="dit_4x8sp1tp0_vae_4x8sp0tp1"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "model_name, image_w, image_h, guidance_scale, num_inference_steps, num_frames",
    [
        ("genmo/mochi-1-preview", 848, 480, 3.5, 50, 168),
    ],
)
@pytest.mark.parametrize("captions_path", ["models/experimental/stable_diffusion_xl_base/coco_data/captions.tsv"])
@pytest.mark.parametrize("coco_statistics_path", ["models/experimental/stable_diffusion_xl_base/coco_data/val2014.npz"])
def test_accuracy_mochi(
    mesh_device,
    model_name,
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
    topology,
    num_links,
    captions_path,
    coco_statistics_path,
    evaluation_range,
):
    start_from, num_prompts = evaluation_range
    prompts = data_helper.flux_get_prompts(captions_path, start_from, num_prompts)
    logger.info(f"start inference from prompt index: {start_from} to {start_from + num_prompts}")

    sp_factor = mesh_device.shape[sp_axis]
    tp_factor = mesh_device.shape[tp_axis]

    # Create the TT Mochi pipeline
    pipeline = TTMochiPipeline(
        mesh_device=mesh_device,
        vae_mesh_shape=vae_mesh_shape,
        parallel_config=parallel_config,
        vae_parallel_config=vae_parallel_config,
        num_links=num_links,
        use_cache=True,
        use_reference_vae=False,
        model_name=model_name,
    )

    generator = torch.Generator("cpu").manual_seed(0)

    videos = []
    for prompt in prompts:
        logger.info(f"Processing prompt: {prompt}")
        logger.info(f"Prompt number: {start_from + len(videos) + 1}")

        frames = tt_pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_frames=num_frames,
            height=image_h,
            width=image_w,
            generator=generator,
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
        "model": "mochi",
        "metadata": {
            "device": os.environ.get("MESH_DEVICE", "unknown"),
            "mesh_shape": mesh_device.shape,
            "vae_mesh_shape": vae_mesh_shape,
            "model_name": model_name,
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
