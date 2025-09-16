# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import csv
from models.experimental.stable_diffusion_xl_base.demo.demo import test_demo
from models.experimental.stable_diffusion_xl_base.utils.clip_encoder import CLIPEncoder
import os
import urllib
from loguru import logger
import statistics
from models.experimental.stable_diffusion_xl_base.utils.fid_score import calculate_fid_score
from models.experimental.stable_diffusion_xl_base.tests.test_common import SDXL_L1_SMALL_SIZE, SDXL_TRACE_REGION_SIZE
import json
from models.common.utility_functions import profiler
from models.experimental.stable_diffusion_xl_base.conftest import get_device_name

test_demo.__test__ = False
COCO_CAPTIONS_DOWNLOAD_PATH = "https://github.com/mlcommons/inference/raw/4b1d1156c23965172ae56eacdd8372f8897eb771/text_to_image/coco2014/captions/captions_source.tsv"
OUT_ROOT, RESULTS_FILE_NAME = "test_reports", "sdxl_test_results.json"


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": SDXL_L1_SMALL_SIZE, "trace_region_size": SDXL_TRACE_REGION_SIZE}], indirect=True
)
@pytest.mark.parametrize(
    "num_inference_steps",
    ((20),),
)
@pytest.mark.parametrize(
    "guidance_scale",
    ((8.0),),
)
@pytest.mark.parametrize(
    "negative_prompt",
    (("normal quality, low quality, worst quality, low res, blurry, nsfw, nude"),),
)
@pytest.mark.parametrize(
    "vae_on_device",
    [
        (True),
        (False),
    ],
    ids=("device_vae", "host_vae"),
)
@pytest.mark.parametrize(
    "capture_trace",
    [
        (True),
        (False),
    ],
    ids=("with_trace", "no_trace"),
)
@pytest.mark.parametrize(
    "encoders_on_device",
    [
        (True),
        (False),
    ],
    ids=("device_encoders", "host_encoders"),
)
@pytest.mark.parametrize("captions_path", ["models/experimental/stable_diffusion_xl_base/coco_data/captions.tsv"])
@pytest.mark.parametrize("coco_statistics_path", ["models/experimental/stable_diffusion_xl_base/coco_data/val2014.npz"])
def test_accuracy_sdxl(
    mesh_device,
    is_ci_env,
    num_inference_steps,
    vae_on_device,
    capture_trace,
    encoders_on_device,
    captions_path,
    coco_statistics_path,
    evaluation_range,
    guidance_scale,
    negative_prompt,
):
    start_from, num_prompts = evaluation_range

    prompts = sdxl_get_prompts(
        captions_path,
        start_from,
        num_prompts,
    )

    logger.info(f"Start inference from prompt index: {start_from} to {start_from + num_prompts}")

    images = test_demo(
        mesh_device,
        is_ci_env,
        prompts,
        negative_prompt,
        num_inference_steps,
        vae_on_device,
        encoders_on_device,
        capture_trace,
        evaluation_range,
        guidance_scale,
    )

    clip = CLIPEncoder()

    clip_scores = []

    for idx, image in enumerate(images):
        clip_scores.append(100 * clip.get_clip_score(prompts[idx], image).item())

    average_clip_score = sum(clip_scores) / len(clip_scores)

    deviation_clip_score = "N/A"
    fid_score = "N/A"

    if num_prompts >= 2:
        deviation_clip_score = statistics.stdev(clip_scores)
        fid_score = calculate_fid_score(images, coco_statistics_path)
    else:
        logger.info("FID score is not calculated for less than 2 prompts.")

    print(f"FID score: {fid_score}")
    print(f"Average CLIP Score: {average_clip_score}")
    print(f"Standard Deviation of CLIP Scores: {deviation_clip_score}")

    data = {
        "model": "sdxl",  # For compatibility with current processes
        "metadata": {
            "device": get_device_name(),
            "device_vae": vae_on_device,
            "capture_trace": capture_trace,
            "encoders_on_device": encoders_on_device,
            "num_inference_steps": num_inference_steps,
            "start_from": start_from,
            "num_prompts": num_prompts,
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "model_name": "sdxl",
        },
        "benchmarks_summary": [
            {
                "device": get_device_name(),
                "model": "sdxl",
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
    trace_flag = "with_trace" if capture_trace else "no_trace"
    vae_flag = "device_vae" if vae_on_device else "host_vae"
    encoders_flag = "device_encoders" if encoders_on_device else "host_encoders"
    new_file_name = f"sdxl_test_results_{trace_flag}_{vae_flag}_{encoders_flag}_{num_inference_steps}.json"
    with open(f"{OUT_ROOT}/{new_file_name}", "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"Test results saved to {OUT_ROOT}/{new_file_name}")

    with open(
        f"{OUT_ROOT}/{RESULTS_FILE_NAME}", "w"
    ) as f:  # this is for CI and test_sdxl_accuracy_with_reset.py compatibility
        json.dump(data, f, indent=4)

    logger.info(f"Test results saved to {OUT_ROOT}/{RESULTS_FILE_NAME}")

    check_clip_scores(start_from, num_prompts, prompts, clip_scores)


def sdxl_get_prompts(
    captions_path,
    start_from,
    num_prompts,
):
    assert (
        0 <= start_from < 5000 and start_from + num_prompts <= 5000
    ), "start_from must be between 0 and 4999, and start_from + num_prompts must not exceed 5000."

    prompts = []

    if not os.path.isfile(captions_path):
        logger.info(f"File {captions_path} not found. Downloading...")
        os.makedirs(os.path.dirname(captions_path), exist_ok=True)
        urllib.request.urlretrieve(COCO_CAPTIONS_DOWNLOAD_PATH, captions_path)
        logger.info("Download complete.")

    with open(captions_path, "r") as tsv_file:
        reader = csv.reader(tsv_file, delimiter="\t")
        next(reader)
        for index, row in enumerate(reader):
            if index < start_from:
                continue
            if index >= start_from + num_prompts:
                break
            prompts.append(row[2])

    return prompts


def check_clip_scores(start_from, num_prompts, prompts, clip_scores):
    assert len(clip_scores) == num_prompts == len(prompts), f"Expected {num_prompts} CLIP scores and prompts."
    num_of_very_low_clip_scores = 0
    for idx, score in enumerate(clip_scores):
        if clip_scores[idx] < 27:
            if clip_scores[idx] < 20:
                logger.error(
                    f"Very low CLIP score detected for image {start_from + idx + 1}: {score}, prompt: {prompts[idx]},  \
                        this indicates a fragmented image or noise or prompt mismatch or something else very wrong."
                )
                num_of_very_low_clip_scores += 1
            else:
                logger.warning(
                    f"Low CLIP score detected for image {start_from + idx + 1}: {score}, prompt: {prompts[idx]}"
                )

    assert num_of_very_low_clip_scores == 0, f"Found {num_of_very_low_clip_scores} images with very low CLIP scores"
