# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import csv
import urllib
import json
import statistics
import pytest
import ttnn
from loguru import logger
import statistics

from .clip_encoder import CLIPEncoder

# from .fid_score import calculate_fid_score
from models.experimental.stable_diffusion_xl_base.utils.fid_score import calculate_fid_score
from models.perf.benchmarking_utils import BenchmarkProfiler


# COCO captions download path
COCO_CAPTIONS_DOWNLOAD_PATH = "https://github.com/mlcommons/inference/raw/4b1d1156c23965172ae56eacdd8372f8897eb771/text_to_image/coco2014/captions/captions_source.tsv"


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


def get_img_evals(model_id, model_setup, images, prompts, coco_statistics_path):
    # Calculate metrics
    logger.info("Calculating CLIP scores...")
    num_prompts = len(prompts)
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

    evals = [
        {
            "model": model_id,
            "average_clip": average_clip_score,
            "clip_standard_deviation": deviation_clip_score,
            "num_prompts": num_prompts,
            "accuracy_check": check_accuracy(num_prompts, average_clip_score, "clip", model_setup),
        },
        {
            "model": model_id,
            "fid_score": fid_score,
            "num_prompts": num_prompts,
            "accuracy_check": check_accuracy(num_prompts, fid_score, "fid", model_setup),
        },
    ]
    return evals


def get_vid_evals(model_id, model_setup, videos, prompts):
    num_prompts = len(prompts)
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

    evals = [
        {
            "model": model_id,
            "average_clip": mean_clip,
            "min_clip": min_clip,
            "max_clip": max_clip,
            "clip_standard_deviation": stddev_clip,
            "num_prompts": num_prompts,
            "accuracy_check": check_accuracy(num_prompts, mean_clip, "clip", model_setup),
        }
    ]
    return evals


def check_accuracy(num_prompts, score, score_key, model_info):
    accuracy_status = 1
    prompt_data = model_info["accuracy"].get(str(num_prompts), {})
    range_key = f"{score_key}_valid_range"
    if range_key in prompt_data:
        score_range = prompt_data[range_key]
        if score >= score_range[0] and score <= score_range[1]:
            return 2
        else:
            return 3

    return accuracy_status


@pytest.mark.parametrize(
    "mesh_device,device_name",
    [
        [(1, 2), "p300"],
        [(2, 2), "qbge"],
        [(2, 4), "t3K"],
        [(4, 8), "galaxy"],
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
    # [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 50000000}],
    # [{"fabric_config": ttnn.FabricConfig.FABRIC_1D,"trace_region_size": 50000000}],
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
def test_tt_dit_accuracy(
    *,
    mesh_device: ttnn.MeshDevice,
    device_name: str,
    num_inference_steps,
    model_id,
    model_location_generator,
    dit_pipeline,
    model_metadata,
    evaluation_range,
) -> None:
    """Accuracy test for TT-Metal DiT pipelines with CLIP and FID score evaluation.
    It relies on the default configurations set in each pipeline.
    """

    benchmark_profiler = BenchmarkProfiler()
    if ttnn.device.is_blackhole() and device_name == "galaxy":
        device_name = "bh_" + device_name
    captions_path = "coco_captions/captions.tsv"
    coco_statistics_path = "coco_statistics/val2014.npz"
    start_from, num_prompts = evaluation_range
    prompts = get_prompts(captions_path, start_from, num_prompts)
    logger.info(f"start inference from prompt index: {start_from} to {start_from + num_prompts}")

    logger.info(f"  Mesh shape: {mesh_device.shape}")
    logger.info(f"  Model ID: {model_id}")
    logger.info(f"  Inference steps: {num_inference_steps}")

    # Create pipeline similar to test_performance_flux1.py
    pipeline = dit_pipeline.create_pipeline(
        mesh_device=mesh_device, checkpoint_name=model_location_generator(model_metadata["hf_id"])
    )

    images = []

    logger.info(f"Starting generation of {len(prompts)} images...")

    # Generate images for each prompt
    is_video = False
    for i, prompt in enumerate(prompts):
        logger.info(f"Generating image {i+1}/{len(prompts)}: {prompt[:50]}...")

        # Generate image using TT-Metal pipeline
        with benchmark_profiler("inference", iteration=i):
            generated_images = pipeline.run_single_prompt(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                seed=0,
                traced=True,
                profiler=benchmark_profiler,
                profiler_iteration=i,
                **model_metadata.get("extra_args", {}),
            )

        images.append(generated_images[0])
        logger.info(f"Image {i+1} completed in {benchmark_profiler.get_duration('inference', i):.2f}s")

        # Save image for inspection
        if not isinstance(generated_images[0], list):  # Save only images for fid computation
            os.makedirs("generated_images", exist_ok=True)
            generated_images[0].save(f"generated_images/tt_dit_image_{i+1}.png")
            is_video = False

    # Performance metrics
    average_inference_time = benchmark_profiler.get_duration_average("inference", 1)  # Skip warmup iterations
    min_inference_time = min([benchmark_profiler.get_duration("inference", i) for i in range(1, num_prompts)])
    max_inference_time = max([benchmark_profiler.get_duration("inference", i) for i in range(1, num_prompts)])

    data = {
        "model": model_id,
        "metadata": {
            "device": f"{device_name}",
            "model_name": model_id,
            "start_from": start_from,
            "num_prompts": num_prompts,
            "num_inference_steps": num_inference_steps,
            "mesh_shape": list(mesh_device.shape),
        },
        "benchmarks_summary": [
            {
                "device": f"{device_name}",
                "model": model_id,
                "average_denoising_time": average_inference_time,
                "average_vae_time": benchmark_profiler.get_duration_average("vae", 1),
                "average_encoding_time": benchmark_profiler.get_duration_average("encoder", 1),
                "average_denoising_time": benchmark_profiler.get_duration_average("denoising", 1),
                "min_inference_time": min_inference_time,
                "max_inference_time": max_inference_time,
            }
        ],
        "evals": get_vid_evals(model_id, model_metadata, images, prompts)
        if is_video
        else get_img_evals(model_id, model_metadata, images, prompts, coco_statistics_path),
    }

    out_root = "test_reports"
    os.makedirs(out_root, exist_ok=True)
    file_path = f"{out_root}/sdxl_test_results.json"
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"test results saved to {file_path}")

    # Synchronize devices
    pipeline.synchronize_devices()

    print(f"\n🎉 Test completed successfully!")
    print(f"⚡ Average Inference Time: {average_inference_time:.2f}s")
