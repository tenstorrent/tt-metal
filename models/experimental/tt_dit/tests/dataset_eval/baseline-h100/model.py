# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import csv

# must be the very first lines!
import os

# os.environ["TRANSFORMERS_NO_TF"] = "1"     # block tensorflow backend
# os.environ["TRANSFORMERS_NO_FLAX"] = "1"   # block flax/jax backend
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # silence tf if it somehow loads
import urllib
import torch
import time
from loguru import logger
import statistics
import json

from clip_encoder import CLIPEncoder
from fid_score import calculate_fid_score
from diffusers import FluxPipeline


class SimpleProfiler:
    def __init__(self):
        self.times = {}
        self.start_times = {}

    def start(self, name):
        self.start_times[name] = time.time()
        if name not in self.times:
            self.times[name] = []

    def end(self, name):
        if name in self.start_times:
            elapsed = time.time() - self.start_times[name]
            self.times[name].append(elapsed)
            del self.start_times[name]

    def get(self, name):
        if name in self.times and self.times[name]:
            return sum(self.times[name]) / len(self.times[name])
        return 0.0


profiler = SimpleProfiler()

# currently same as sdxl
COCO_CAPTIONS_DOWNLOAD_PATH = "https://github.com/mlcommons/inference/raw/4b1d1156c23965172ae56eacdd8372f8897eb771/text_to_image/coco2014/captions/captions_source.tsv"
OUT_ROOT, RESULTS_FILE_NAME = "test_reports", "flux_test_results.json"


@pytest.mark.parametrize(
    "device_params",
    [{"device": "cuda" if torch.cuda.is_available() else "cpu"}],
    indirect=True,
)
@pytest.mark.parametrize(
    "model_name, image_w, image_h, guidance_scale, num_inference_steps",
    [
        # ("dev", 1024, 1024, 3.5, 28),  # Full resolution on H100
        ("schnell", 1024, 1024, 1.0, 4)
    ],
)
@pytest.mark.parametrize("captions_path", ["captions.tsv"])
@pytest.mark.parametrize("coco_statistics_path", ["val2014.npz"])
def test_accuracy_model(
    device_params,
    model_name,
    image_w,
    image_h,
    guidance_scale,
    num_inference_steps,
    captions_path,
    coco_statistics_path,
    evaluation_range,
):
    start_from, num_prompts = evaluation_range
    prompts = flux_get_prompts(captions_path, start_from, num_prompts)
    logger.info(f"start inference from prompt index: {start_from} to {start_from + num_prompts}")

    device = device_params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    # Initialize Diffusers pipeline
    model_id = f"black-forest-labs/FLUX.1-{model_name}"
    logger.info(f"Loading model: {model_id}")

    pipeline = FluxPipeline.from_pretrained(
        model_id,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,  # Use bfloat16 for H100
        use_safetensors=True,
    )

    # Move to device manually
    pipeline = pipeline.to(device)
    logger.info(f"Pipeline loaded on {device}")

    if device == "cuda":
        # Enable optimizations for H100
        try:
            pipeline.enable_attention_slicing()
            logger.info("✓ Attention slicing enabled")
        except Exception as e:
            logger.warning(f"Could not enable attention slicing: {e}")

        # Optional: Enable VAE slicing for memory efficiency
        try:
            if hasattr(pipeline, "enable_vae_slicing"):
                pipeline.enable_vae_slicing()
                logger.info("✓ VAE slicing enabled")
        except Exception as e:
            logger.warning(f"Could not enable VAE slicing: {e}")

        # Enable memory efficient attention for Flux
        try:
            pipeline.enable_memory_efficient_attention()
            logger.info("✓ Memory efficient attention enabled")
        except Exception as e:
            logger.warning(f"Could not enable memory efficient attention: {e}")

    # Set generator for reproducibility
    generator = torch.Generator(device=device).manual_seed(0)

    images = []
    total_times = []

    logger.info(f"Starting generation of {len(prompts)} images at {image_w}x{image_h}...")

    for i, prompt in enumerate(prompts):
        logger.info(f"Generating image {i+1}/{len(prompts)}: {prompt[:50]}...")
        negative_prompt = ""

        start_total = time.time()
        profiler.start("denoising_loop")

        # Generate image
        with torch.no_grad():
            generated_images = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=image_h,
                width=image_w,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type="pil",
            ).images

        profiler.end("denoising_loop")
        end_total = time.time()

        total_time = end_total - start_total
        total_times.append(total_time)

        images.append(generated_images[0])
        logger.info(f"Image {i+1} completed in {total_time:.2f}s")

        # Optional: Save image for inspection
        os.makedirs("generated_images", exist_ok=True)
        generated_images[0].save(f"generated_images/image_{i+1}.png")

    # Calculate metrics
    logger.info("Calculating CLIP scores...")
    clip = CLIPEncoder()  # Remove device parameter, it auto-detects
    clip_scores = [100 * clip.get_clip_score(prompts[i], img).item() for i, img in enumerate(images)]
    average_clip_score = sum(clip_scores) / len(clip_scores)

    deviation_clip_score = "N/A"
    fid_score = "N/A"

    # Calculate CLIP standard deviation if we have multiple prompts
    if num_prompts >= 2:
        deviation_clip_score = statistics.stdev(clip_scores)
        logger.info(f"CLIP scores individual: {[f'{score:.2f}' for score in clip_scores]}")

    # Calculate FID score if we have the COCO statistics file
    if num_prompts >= 2 and os.path.isfile(coco_statistics_path):
        logger.info("Calculating FID score...")
        fid_score = calculate_fid_score(images, coco_statistics_path)
    elif num_prompts >= 2 and not os.path.isfile(coco_statistics_path):
        logger.warning(f"FID calculation skipped: COCO stats file not found at {coco_statistics_path}")
        logger.info("To enable FID calculation, download COCO validation statistics:")
        logger.info("wget http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_imagenet_256_hdf5.npz -O val2014.npz")
    elif num_prompts < 2:
        logger.info("FID calculation requires at least 2 images")

    # Performance metrics
    average_inference_time = sum(total_times) / len(total_times) if total_times else 0
    min_inference_time = min(total_times) if total_times else 0
    max_inference_time = max(total_times) if total_times else 0

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
        "model": "flux",
        "metadata": {
            "device": device.upper(),
            "model_name": model_name,
            "actual_model": model_id,
            "start_from": start_from,
            "num_prompts": num_prompts,
            "image_width": image_w,
            "image_height": image_h,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "backend": "diffusers",
            "dtype": str(torch.bfloat16 if device == "cuda" else torch.float32),
            "optimizations": ["attention_slicing", "vae_slicing", "memory_efficient_attention"],
        },
        "benchmarks_summary": [
            {
                "device": device.upper(),
                "model": "flux",
                "average_denoising_time": profiler.get("denoising_loop"),
                "average_vae_time": 0.0,  # VAE time is included in denoising for diffusers
                "average_inference_time": average_inference_time,
                "min_inference_time": min_inference_time,
                "max_inference_time": max_inference_time,
                "average_clip": average_clip_score,
                "deviation_clip": clip_std_serializable,
                "fid_score": fid_serializable,
                "individual_clip_scores": [float(score) for score in clip_scores],  # Add individual scores
            }
        ],
    }

    os.makedirs(OUT_ROOT, exist_ok=True)
    with open(f"{OUT_ROOT}/{RESULTS_FILE_NAME}", "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"test results saved to {OUT_ROOT}/{RESULTS_FILE_NAME}")

    # Cleanup
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"\n🎉 Test completed successfully!")
    print(f"📊 Average CLIP Score: {average_clip_score:.2f}")
    print(f"⚡ Average Inference Time: {average_inference_time:.2f}s")
    print(f"🖼️  Images saved to: generated_images/")


def flux_get_prompts(captions_path, start_from, num_prompts):
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


# Fixture for device parameters
@pytest.fixture
def device_params(request):
    return request.param


# Fixture for evaluation range
# @pytest.fixture
# def evaluation_range():
#     # Default evaluation range - can be overridden
#     return (0, 5)  # start_from=0, num_prompts=5 for better statistics
