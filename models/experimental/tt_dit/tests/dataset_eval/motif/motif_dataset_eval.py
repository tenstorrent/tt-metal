# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import csv
import os
import urllib
from loguru import logger
import statistics
import json

from models.experimental.tt_dit.tests.dataset_eval.utils.clip_encoder import CLIPEncoder
from models.experimental.tt_dit.tests.dataset_eval.utils.fid_score import calculate_fid_score
from models.experimental.tt_dit.pipelines.motif.pipeline_motif import MotifPipeline
from models.experimental.tt_dit.pipelines.stable_diffusion_35_large.pipeline_stable_diffusion_35_large import (
    TimingCollector,
)
import ttnn


COCO_CAPTIONS_DOWNLOAD_PATH = "https://github.com/mlcommons/inference/raw/4b1d1156c23965172ae56eacdd8372f8897eb771/text_to_image/coco2014/captions/captions_source.tsv"
OUT_ROOT, RESULTS_FILE_NAME = "test_reports", "motif_test_results.json"


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 31000000}],
    indirect=True,
)
@pytest.mark.parametrize(
    ("mesh_device", "cfg_config", "sp_config", "tp_config", "encoder_tp", "vae_tp", "topology", "num_links"),
    [
        pytest.param((4, 8), (2, 1), (4, 0), (4, 1), (4, 1), (4, 1), ttnn.Topology.Linear, 4, id="4x8_cfg2_sp4_tp4"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "image_w, image_h, guidance_scale, num_inference_steps",
    [
        (1024, 1024, 4.0, 20),
    ],
)
@pytest.mark.parametrize("captions_path", ["models/experimental/stable_diffusion_xl_base/coco_data/captions.tsv"])
@pytest.mark.parametrize("coco_statistics_path", ["models/experimental/stable_diffusion_xl_base/coco_data/val2014.npz"])
def test_accuracy_motif(
    mesh_device,
    image_w,
    image_h,
    guidance_scale,
    num_inference_steps,
    cfg_config,
    sp_config,
    tp_config,
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
    prompts = motif_get_prompts(captions_path, start_from, num_prompts)
    logger.info(f"start inference from prompt index: {start_from} to {start_from + num_prompts}")

    cfg_factor, cfg_axis = cfg_config
    sp_factor, sp_axis = sp_config
    tp_factor, tp_axis = tp_config

    # Enable T5 for Motif
    enable_t5_text_encoder = True

    # Create pipeline
    pipeline = MotifPipeline.create_pipeline(
        mesh_device=mesh_device,
        dit_cfg=cfg_config,
        dit_sp=sp_config,
        dit_tp=tp_config,
        encoder_tp=encoder_tp,
        vae_tp=vae_tp,
        enable_t5_text_encoder=enable_t5_text_encoder,
        use_torch_t5_text_encoder=False,
        use_torch_clip_text_encoder=False,
        num_links=num_links,
        topology=topology,
        width=image_w,
        height=image_h,
    )

    # Create timing collector for detailed timing information
    timing_collector = TimingCollector()
    pipeline.timing_collector = timing_collector

    images = []
    all_timings = []
    for prompt in prompts:
        logger.info(f"Processing prompt: {prompt}")
        logger.info(f"Prompt number: {start_from + len(images) + 1}")
        negative_prompt = ""

        # Reset timing collector for this prompt
        timing_collector.reset()

        generated_images = pipeline(
            prompt_1=[prompt],
            prompt_2=[prompt],
            prompt_3=[prompt],
            negative_prompt_1=[negative_prompt],
            negative_prompt_2=[negative_prompt],
            negative_prompt_3=[negative_prompt],
            num_inference_steps=num_inference_steps,
            cfg_scale=guidance_scale,
            seed=0,
            traced=True,
        )

        images.append(generated_images[0])

        # Collect timing data for this prompt
        timing_data = timing_collector.get_timing_data()
        all_timings.append(timing_data)
        logger.info(f"Prompt {start_from + len(images)} completed in {timing_data.total_time:.2f}s")

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

    # Calculate timing statistics
    total_times = [t.total_time for t in all_timings]
    vae_times = [t.vae_decoding_time for t in all_timings]
    # Calculate denoising time as total - vae for each run
    denoising_times = [t.total_time - t.vae_decoding_time - t.total_encoding_time for t in all_timings]

    data = {
        "model": "motif-image-6b",
        "metadata": {
            "device": "TG",
            "start_from": start_from,
            "num_prompts": num_prompts,
            "image_width": image_w,
            "image_height": image_h,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "cfg_factor": cfg_factor,
            "sp_factor": sp_factor,
            "tp_factor": tp_factor,
            "encoder_tp_factor": encoder_tp[0],
            "vae_tp_factor": vae_tp[0],
            "t5_enabled": enable_t5_text_encoder,
        },
        "benchmarks_summary": [
            {
                "device": "TG",
                "model": "motif-image-6b",
                "average_denoising_time": sum(denoising_times) / len(denoising_times) if denoising_times else 0,
                "average_vae_time": sum(vae_times) / len(vae_times) if vae_times else 0,
                "average_inference_time": sum(total_times) / len(total_times) if total_times else 0,
                "min_inference_time": min(total_times) if total_times else 0,
                "max_inference_time": max(total_times) if total_times else 0,
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


def motif_get_prompts(captions_path, start_from, num_prompts):
    assert (
        0 <= start_from < 5000 and start_from + num_prompts <= 5000
    ), "start_from must be between 0 and 4999, and start_from + num_prompts must not exceed 5000."

    prompts = []
    if not os.path.isfile(captions_path):
        logger.info(f"file {captions_path} not found. downloading...")
        os.makedirs(os.path.dirname(captions_path), exist_ok=True)
        urllib.request.urlretrieve(COCO_CAPTIONS_DOWNLOAD_PATH, captions_path)
        logger.info("download complete.")

    with open(captions_path, "r") as tsv_file:
        reader = csv.reader(tsv_file, delimiter="\t")
        next(reader)  # skip header
        for index, row in enumerate(reader):
            if index < start_from:
                continue
            if index >= start_from + num_prompts:
                break
            prompts.append(row[2])
    return prompts
