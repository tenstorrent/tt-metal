# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import csv
import urllib
import json
import time
import statistics
import pytest
import ttnn
from loguru import logger

from .clip_encoder import CLIPEncoder

# from .fid_score import calculate_fid_score
from models.experimental.stable_diffusion_xl_base.utils.fid_score import calculate_fid_score

from ...pipelines.flux1.pipeline_flux1 import Flux1Pipeline
from ...pipelines.motif.pipeline_motif import MotifPipeline
from ...pipelines.stable_diffusion_35_large.pipeline_stable_diffusion_35_large import StableDiffusion3Pipeline
from ...pipelines.stable_diffusion_35_large.pipeline_stable_diffusion_35_large import TimingCollector
from models.common.utility_functions import profiler

# COCO captions download path
COCO_CAPTIONS_DOWNLOAD_PATH = "https://github.com/mlcommons/inference/raw/4b1d1156c23965172ae56eacdd8372f8897eb771/text_to_image/coco2014/captions/captions_source.tsv"


def get_pipeline(mesh_device, model_id, model_location_generator):
    pipeline_map = {
        "flux1.dev": lambda: Flux1Pipeline.create_pipeline(
            mesh_device=mesh_device, checkpoint_name=model_location_generator(f"black-forest-labs/FLUX.1-dev")
        ),
        "flux1.schnell": lambda: Flux1Pipeline.create_pipeline(
            mesh_device=mesh_device, checkpoint_name=model_location_generator(f"black-forest-labs/FLUX.1-schnell")
        ),
        "sd35.large": lambda: StableDiffusion3Pipeline.create_pipeline(
            mesh_device=mesh_device,
            model_checkpoint_path=model_location_generator(f"stabilityai/stable-diffusion-3.5-large"),
        ),
        "motif.image.6b.preview": lambda: MotifPipeline.create_pipeline(
            mesh_device=mesh_device,
            model_checkpoint_path=model_location_generator(f"Motif-Technologies/Motif-Image-6B-Preview"),
        ),
    }
    return pipeline_map[model_id]()


def get_prompts(captions_path, start_from, num_prompts):
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


@pytest.mark.parametrize(
    "mesh_device",
    [
        (1, 2),
        (2, 2),
        (2, 4),
        (4, 8),
    ],
    ids=[
        "1x2",
        "2x2",
        "2x4",
        "4x8",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 50000000}],
    indirect=True,
)
def test_tt_dit_accuracy(
    *,
    mesh_device: ttnn.MeshDevice,
    num_inference_steps,
    model_id,
    model_location_generator,
    evaluation_range,
) -> None:
    """Accuracy test for TT-Metal DiT pipelines with CLIP and FID score evaluation.
    It relies on the default configurations set in each pipeline.
    """

    captions_path = "coco_captions/captions.tsv"
    coco_statistics_path = "coco_statistics/val2014.npz"
    start_from, num_prompts = evaluation_range
    prompts = get_prompts(captions_path, start_from, num_prompts)
    logger.info(f"start inference from prompt index: {start_from} to {start_from + num_prompts}")

    logger.info(f"  Mesh shape: {mesh_device.shape}")
    logger.info(f"  Model ID: {model_id}")
    logger.info(f"  Inference steps: {num_inference_steps}")

    # Create pipeline similar to test_performance_flux1.py
    pipeline = get_pipeline(mesh_device, model_id, model_location_generator)

    # Setup timing collector
    timer = TimingCollector()
    pipeline.timing_collector = timer

    images = []
    total_times = []

    logger.info(f"Starting generation of {len(prompts)} images...")

    # Generate images for each prompt
    for i, prompt in enumerate(prompts):
        logger.info(f"Generating image {i+1}/{len(prompts)}: {prompt[:50]}...")

        start_total = time.time()
        profiler.start("denoising_loop")

        # Generate image using TT-Metal pipeline
        generated_images = pipeline.run_single_prompt(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            seed=0,
            traced=True,
        )

        profiler.end("denoising_loop")
        end_total = time.time()

        total_time = end_total - start_total
        total_times.append(total_time)

        images.append(generated_images[0])
        logger.info(f"Image {i+1} completed in {total_time:.2f}s")

        # Save image for inspection
        os.makedirs("generated_images", exist_ok=True)
        generated_images[0].save(f"generated_images/tt_dit_image_{i+1}.png")

    # Calculate metrics
    logger.info("Calculating CLIP scores...")
    clip = CLIPEncoder()
    clip_scores = [100 * clip.get_clip_score(prompts[i], img).item() for i, img in enumerate(images)]
    average_clip_score = sum(clip_scores) / len(clip_scores)

    deviation_clip_score = "N/A"
    fid_score = "N/A"

    # Calculate CLIP standard deviation if we have multiple prompts
    if num_prompts >= 2:
        deviation_clip_score = statistics.stdev(clip_scores)
        logger.info(f"CLIP scores individual: {[f'{score:.2f}' for score in clip_scores]}")

        # Calculate FID score if we have the COCO statistics file
        logger.info("Calculating FID score...")
        try:
            fid_score = calculate_fid_score(images, coco_statistics_path)
        except Exception as e:
            logger.error(f"Error calculating FID score: {e}")
    else:
        logger.info("FID calculation requires at least 2 images")

    # Performance metrics
    average_inference_time = sum(total_times) / len(total_times) if total_times else 0
    min_inference_time = min(total_times) if total_times else 0
    max_inference_time = max(total_times) if total_times else 0

    # Get timing data from pipeline
    timing_data = timer.get_timing_data()
    average_denoising_time = profiler.get("denoising_loop")
    average_vae_time = timing_data.vae_decoding_time if timing_data else 0.0

    # Results
    logger.info("=== RESULTS ===")
    logger.info(f"Average CLIP Score: {average_clip_score:.2f}")
    logger.info(f"CLIP Score Std Dev: {deviation_clip_score}")
    logger.info(f"Average Inference Time: {average_inference_time:.2f}s")
    logger.info(f"Min/Max Inference Time: {min_inference_time:.2f}s / {max_inference_time:.2f}s")
    logger.info(f"FID Score: {fid_score}")

    print(f"FID score: {fid_score}")
    print(f"Average CLIP Score: {average_clip_score}")
    print(f"Standard Deviation of CLIP Scores: {deviation_clip_score}")

    # Ensure values are JSON serializable
    clip_std_serializable = float(deviation_clip_score) if deviation_clip_score != "N/A" else "N/A"
    fid_serializable = float(fid_score) if fid_score != "N/A" else "N/A"

    data = {
        "model": model_id,
        "metadata": {
            "device": f"TT-Metal-{mesh_device.shape[0]}x{mesh_device.shape[1]}",
            "model_name": model_id,
            "start_from": start_from,
            "num_prompts": num_prompts,
            "num_inference_steps": num_inference_steps,
            "backend": "tt-metal",
            "mesh_shape": list(mesh_device.shape),
        },
        "benchmarks_summary": [
            {
                "device": f"TT-Metal-{mesh_device.shape[0]}x{mesh_device.shape[1]}",
                "model": model_id,
                "average_denoising_time": average_denoising_time,
                "average_vae_time": average_vae_time,
                "average_inference_time": average_inference_time,
                "min_inference_time": min_inference_time,
                "max_inference_time": max_inference_time,
                "average_clip": average_clip_score,
                "deviation_clip": clip_std_serializable,
                "fid_score": fid_serializable,
                "individual_clip_scores": [float(score) for score in clip_scores],
            }
        ],
    }

    out_root = "test_reports"
    os.makedirs(out_root, exist_ok=True)
    file_path = f"{out_root}/test_{model_id.replace('.', '')}_results.json"
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"test results saved to {file_path}")

    # Synchronize devices
    pipeline.synchronize_devices()

    print(f"\n🎉 Test completed successfully!")
    print(f"📊 Average CLIP Score: {average_clip_score:.2f}")
    print(f"⚡ Average Inference Time: {average_inference_time:.2f}s")
    print(f"🖼️  Images saved to: generated_images/")
