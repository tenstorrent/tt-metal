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
from models.experimental.stable_diffusion_xl_base.tests.test_common import SDXL_L1_SMALL_SIZE
import json
from models.utility_functions import profiler

test_demo.__test__ = False
COCO_CAPTIONS_DOWNLOAD_PATH = "https://github.com/mlcommons/inference/raw/4b1d1156c23965172ae56eacdd8372f8897eb771/text_to_image/coco2014/captions/captions_source.tsv"


@pytest.mark.parametrize("device_params", [{"l1_small_size": SDXL_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize(
    "num_inference_steps",
    ((50),),
)
@pytest.mark.parametrize(
    "vae_on_device",
    [
        (True),
        (False),
    ],
    ids=("device_vae", "host_vae"),
)
@pytest.mark.parametrize("captions_path", ["models/experimental/stable_diffusion_xl_base/coco_data/captions.tsv"])
@pytest.mark.parametrize("coco_statistics_path", ["models/experimental/stable_diffusion_xl_base/coco_data/val2014.npz"])
def test_accuracy_sdxl(
    mesh_device,
    is_ci_env,
    num_inference_steps,
    vae_on_device,
    captions_path,
    coco_statistics_path,
    evaluation_range,
):
    start_from, num_prompts = evaluation_range

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
        for row in reader:
            prompts.append(row[2])

    logger.info(f"Start inference from prompt index: {start_from} to {start_from + num_prompts}")

    images = test_demo(
        mesh_device,
        is_ci_env,
        prompts[start_from : start_from + num_prompts],
        num_inference_steps,
        vae_on_device,
        evaluation_range,
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
            "device": "N150",
            "device_vae": vae_on_device,
            "start_from": start_from,
            "num_prompts": num_prompts,
            "model_name": "sdxl",
        },
        "benchmarks_summary": [
            {
                "device": "N150",
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

    out_root, file_name = "test_reports", "sdxl_test_results.json"
    os.makedirs(out_root, exist_ok=True)

    with open(f"{out_root}/{file_name}", "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"Test results saved to {out_root}/{file_name}")
