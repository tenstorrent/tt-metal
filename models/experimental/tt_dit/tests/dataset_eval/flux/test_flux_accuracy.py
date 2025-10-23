# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import os
from loguru import logger
import statistics
import json

from models.experimental.tt_dit.tests.dataset_eval.utils.clip_encoder import CLIPEncoder
from models.experimental.tt_dit.tests.dataset_eval.utils.fid_score import calculate_fid_score
from models.experimental.tt_dit.pipelines.flux1.pipeline_flux1 import Flux1Pipeline
import ttnn
from models.common.utility_functions import profiler
import models.experimental.tt_dit.tests.dataset_eval.utils.data_helper as data_helper


OUT_ROOT, RESULTS_FILE_NAME = "test_reports", "flux_test_results.json"


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 33000000}],
    indirect=True,
)
@pytest.mark.parametrize(
    ("mesh_device", "sp", "tp", "encoder_tp", "vae_tp", "topology", "num_links"),
    [
        # pytest.param((2, 4), (2, 0), (4, 1), (4, 1), (4, 1), ttnn.Topology.Linear, 1, id="2x4sp0tp1"),
        pytest.param((4, 8), (4, 0), (8, 1), (4, 0), (4, 0), ttnn.Topology.Linear, 4, id="4x8sp0tp1"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "model_name, image_w, image_h, guidance_scale, num_inference_steps",
    [
        ("schnell", 1024, 1024, 1.0, 4),
        # ("dev", 1024, 1024, 3.5, 28),
    ],
)
@pytest.mark.parametrize("captions_path", ["models/experimental/stable_diffusion_xl_base/coco_data/captions.tsv"])
@pytest.mark.parametrize("coco_statistics_path", ["models/experimental/stable_diffusion_xl_base/coco_data/val2014.npz"])
def test_accuracy_flux(
    mesh_device,
    model_name,
    image_w,
    image_h,
    guidance_scale,
    num_inference_steps,
    sp,
    tp,
    encoder_tp,
    vae_tp,
    topology,
    num_links,
    captions_path,
    coco_statistics_path,
    evaluation_range,
    model_location_generator,
):
    start_from, num_prompts = evaluation_range
    prompts = data_helper.get_prompts(captions_path, start_from, num_prompts)
    logger.info(f"start inference from prompt index: {start_from} to {start_from + num_prompts}")

    sp_factor, sp_axis = sp
    tp_factor, tp_axis = tp

    # Enable T5 based on device configuration
    enable_t5_text_encoder = True  # Flux typically uses T5

    pipeline = Flux1Pipeline.create_pipeline(
        checkpoint_name=model_location_generator(f"black-forest-labs/FLUX.1-{model_name}"),
        mesh_device=mesh_device,
        dit_sp=sp,
        dit_tp=tp,
        encoder_tp=encoder_tp,
        vae_tp=vae_tp,
        enable_t5_text_encoder=enable_t5_text_encoder,
        use_torch_t5_text_encoder=False,
        use_torch_clip_text_encoder=False,
        num_links=num_links,
        topology=topology,
    )

    images = []
    for prompt in prompts:
        logger.info(f"Processing prompt: {prompt}")
        logger.info(f"Prompt number: {start_from + len(images) + 1}")
        negative_prompt = ""

        profiler.start("denoising_loop")
        profiler.start("vae_decode")

        generated_images = pipeline(
            prompt_1=[prompt],
            prompt_2=[prompt],
            negative_prompt_1=[negative_prompt],
            negative_prompt_2=[negative_prompt],
            num_inference_steps=num_inference_steps,
            width=image_w,
            height=image_h,
            guidance_scale=guidance_scale,
            seed=0,
            traced=True,
        )

        profiler.end("denoising_loop")
        profiler.end("vae_decode")

        images.append(generated_images[0])

    clip = CLIPEncoder()
    clip_scores = [100 * clip.get_clip_score(prompts[i], img).item() for i, img in enumerate(images)]
    average_clip_score = sum(clip_scores) / len(clip_scores)

    deviation_clip_score = "N/A"
    fid_score = "N/A"
    if num_prompts >= 2 and os.path.isfile(coco_statistics_path):
        deviation_clip_score = statistics.stdev(clip_scores)
        fid_score = calculate_fid_score(images, coco_statistics_path)
    elif num_prompts >= 2 and not os.path.isfile(coco_statistics_path):
        logger.warning(f"fid skipped: stats file not found at {coco_statistics_path}")

    print(f"FID score: {fid_score}")
    print(f"Average CLIP Score: {average_clip_score}")
    print(f"Standard Deviation of CLIP Scores: {deviation_clip_score}")

    data = {
        "model": "flux",
        "metadata": {
            "device": "TG",
            "model_name": model_name,
            "start_from": start_from,
            "num_prompts": num_prompts,
            "image_width": image_w,
            "image_height": image_h,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "sp_factor": sp_factor,
            "tp_factor": tp_factor,
            "encoder_tp_factor": encoder_tp[0],
            "vae_tp_factor": vae_tp[0],
        },
        "benchmarks_summary": [
            {
                "device": "TG",
                "model": "flux",
                "average_denoising_time": profiler.get("denoising_loop"),
                "average_vae_time": profiler.get("vae_decode"),
                "average_inference_time": profiler.get("denoising_loop") + profiler.get("vae_decode"),
                "min_inference_time": min(
                    i + j for i, j in zip(profiler.times["denoising_loop"], profiler.times["vae_decode"])
                ),
                "max_inference_time": max(
                    i + j for i, j in zip(profiler.times["denoising_loop"], profiler.times["vae_decode"])
                ),
                "average_clip": average_clip_score,
                "deviation_clip": deviation_clip_score,
                "fid_score": fid_score,
            }
        ],
    }

    os.makedirs(OUT_ROOT, exist_ok=True)
    with open(f"{OUT_ROOT}/{RESULTS_FILE_NAME}", "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"test results saved to {OUT_ROOT}/{RESULTS_FILE_NAME}")

    # Synchronize devices
    ttnn.synchronize_device(mesh_device)
