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

test_demo.__test__ = False
COCO_CAPTIONS_DOWNLOAD_PATH = "https://github.com/mlcommons/inference/raw/4b1d1156c23965172ae56eacdd8372f8897eb771/text_to_image/coco2014/captions/captions_source.tsv"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 6 * 16384}], indirect=True)
@pytest.mark.parametrize(
    "num_inference_steps",
    ((50),),
)
@pytest.mark.parametrize(
    "classifier_free_guidance",
    [
        (True),
    ],
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
    device,
    use_program_cache,
    is_ci_env,
    num_inference_steps,
    classifier_free_guidance,
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
        device,
        use_program_cache,
        is_ci_env,
        prompts[start_from : start_from + num_prompts],
        num_inference_steps,
        classifier_free_guidance,
        vae_on_device,
        evaluation_range,
    )

    clip = CLIPEncoder()

    clip_scores = []

    for idx, image in enumerate(images):
        clip_scores.append(100 * clip.get_clip_score(prompts[idx], image).item())

    average_clip_score = sum(clip_scores) / len(clip_scores)

    fid_value = calculate_fid_score(images, coco_statistics_path)

    print(f"FID score: {fid_value:.4f}")
    print(f"Average CLIP Score: {average_clip_score:.4f}")
    print(f"Standard Deviation of CLIP Scores: {statistics.stdev(clip_scores):.4f}")
