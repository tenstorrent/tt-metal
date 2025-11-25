# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import csv

# must be the very first lines!
import os
import urllib
import torch
import time
from loguru import logger
import statistics
import json
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clip_encoder import CLIPEncoder
from fid_score import calculate_fid_score

from diffusers import StableDiffusion3Pipeline


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

COCO_CAPTIONS_DOWNLOAD_PATH = "https://github.com/mlcommons/inference/raw/4b1d1156c23965172ae56eacdd8372f8897eb771/text_to_image/coco2014/captions/captions_source.tsv"
OUT_ROOT, RESULTS_FILE_NAME = "test_reports", "sd35_test_results.json"


@pytest.mark.parametrize(
    "device_params",
    [{"device": "cuda" if torch.cuda.is_available() else "cpu"}],
    indirect=True,
)
@pytest.mark.parametrize(
    "model_name, image_w, image_h, guidance_scale, num_inference_steps, max_sequence_length",
    [
        ("stable-diffusion-3.5-large", 1024, 1024, 3.5, 40, 256),
    ],
)
@pytest.mark.parametrize("captions_path", ["captions.tsv"])
@pytest.mark.parametrize("coco_statistics_path", ["val2014.npz"])
def test_accuracy_sd35(
    device_params,
    model_name,
    image_w,
    image_h,
    guidance_scale,
    num_inference_steps,
    max_sequence_length,
    captions_path,
    coco_statistics_path,
    evaluation_range,
):
    start_from, num_prompts = evaluation_range
    prompts = sd35_get_prompts(captions_path, start_from, num_prompts)
    logger.info(f"start inference from prompt index: {start_from} to {start_from + num_prompts}")

    device = device_params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    model_id = "./sd35_large"
    logger.info(f"Loading model: {model_id}")
    logger.info(f"T5 text encoder enabled with max_sequence_length={max_sequence_length}")

    pipeline = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,  # Use bfloat16 for H100
        use_safetensors=True,
    )

    pipeline = pipeline.to(device)
    logger.info(f"Pipeline loaded on {device}")

    if device == "cuda":
        # enable optimizations for H100
        try:
            pipeline.enable_attention_slicing()
            logger.info("✓ Attention slicing enabled")
        except Exception as e:
            logger.warning(f"Could not enable attention slicing: {e}")

        # enable VAE slicing for memory efficiency
        try:
            if hasattr(pipeline, "enable_vae_slicing"):
                pipeline.enable_vae_slicing()
                logger.info("✓ VAE slicing enabled")
        except Exception as e:
            logger.warning(f"Could not enable VAE slicing: {e}")

    # set generator for reproducibility
    generator = torch.Generator(device=device).manual_seed(0)

    images = []
    total_times = []

    logger.info(f"Starting generation of {len(prompts)} images at {image_w}x{image_h}...")

    for i, prompt in enumerate(prompts):
        logger.info(f"Generating image {i+1}/{len(prompts)}: {prompt[:50]}...")

        start_total = time.time()
        profiler.start("denoising_loop")

        # generate image with triple prompt (SD3.5 uses 3 text encoders)
        with torch.no_grad():
            generated_images = pipeline(
                prompt=prompt,
                prompt_2=prompt,
                prompt_3=prompt,
                negative_prompt="",
                negative_prompt_2="",
                negative_prompt_3="",
                height=image_h,
                width=image_w,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                max_sequence_length=max_sequence_length,
                generator=generator,
                output_type="pil",
            ).images

        profiler.end("denoising_loop")
        end_total = time.time()

        total_time = end_total - start_total
        total_times.append(total_time)

        images.append(generated_images[0])
        logger.info(f"Image {i+1} completed in {total_time:.2f}s")

        # optional: save image
        os.makedirs("generated_images", exist_ok=True)
        generated_images[0].save(f"generated_images/image_{i+1}.png")

    # calculate metrics
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
    if num_prompts >= 2 and os.path.isfile(coco_statistics_path):
        logger.info("Calculating FID score...")
        fid_score = calculate_fid_score(images, coco_statistics_path)
    elif num_prompts >= 2 and not os.path.isfile(coco_statistics_path):
        logger.warning(f"FID calculation skipped: COCO stats file not found at {coco_statistics_path}")
        logger.info("To enable FID calculation, download COCO validation statistics:")
        logger.info(
            "wget https://github.com/mlcommons/inference/raw/4b1d1156c23965172ae56eacdd8372f8897eb771/text_to_image/tools/val2014.npz"
        )
    elif num_prompts < 2:
        logger.info("FID calculation requires at least 2 images")

    average_inference_time = sum(total_times) / len(total_times) if total_times else 0
    min_inference_time = min(total_times) if total_times else 0
    max_inference_time = max(total_times) if total_times else 0

    # results
    logger.info("=== RESULTS ===")
    logger.info(f"Average CLIP Score: {average_clip_score:.2f}")
    logger.info(f"CLIP Score Std Dev: {deviation_clip_score}")
    logger.info(f"Average Inference Time: {average_inference_time:.2f}s")
    logger.info(f"Min/Max Inference Time: {min_inference_time:.2f}s / {max_inference_time:.2f}s")
    logger.info(f"FID Score: {fid_score}")

    print(f"FID score: {fid_score}")
    print(f"Average CLIP Score: {average_clip_score}")
    print(f"Standard Deviation of CLIP Scores: {deviation_clip_score}")

    # check if T5 encoder is being used
    t5_enabled = hasattr(pipeline, "text_encoder_3") and pipeline.text_encoder_3 is not None
    logger.info(f"T5 text encoder enabled: {t5_enabled}")

    # ensure values are JSON serializable
    clip_std_serializable = float(deviation_clip_score) if deviation_clip_score != "N/A" else "N/A"
    fid_serializable = float(fid_score) if fid_score != "N/A" else "N/A"

    data = {
        "model": model_name,
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
            "max_sequence_length": max_sequence_length,
            "t5_enabled": t5_enabled,
            "backend": "diffusers",
            "dtype": str(torch.bfloat16 if device == "cuda" else torch.float32),
            "optimizations": ["attention_slicing", "vae_slicing"],
        },
        "benchmarks_summary": [
            {
                "device": device.upper(),
                "model": model_name,
                "average_denoising_time": profiler.get("denoising_loop"),
                "average_vae_time": 0.0,  # VAE time is included in denoising for diffusers
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

    os.makedirs(OUT_ROOT, exist_ok=True)
    with open(f"{OUT_ROOT}/{RESULTS_FILE_NAME}", "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"test results saved to {OUT_ROOT}/{RESULTS_FILE_NAME}")

    # cleanup
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"\n🎉 Test completed successfully!")
    print(f"📊 Average CLIP Score: {average_clip_score:.2f}")
    print(f"⚡ Average Inference Time: {average_inference_time:.2f}s")
    print(f"🖼️  Images saved to: generated_images/")


def sd35_get_prompts(captions_path, start_from, num_prompts):
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


# fixture for device parameters
@pytest.fixture
def device_params(request):
    return request.param


# python -m pytest model.py::test_accuracy_sd35 -v --start-from=0 --num-prompts=5 -s
