# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import os
import csv
import urllib
import json

COCO_CAPTIONS_DOWNLOAD_PATH = "https://github.com/mlcommons/inference/raw/4b1d1156c23965172ae56eacdd8372f8897eb771/text_to_image/coco2014/captions/captions_source.tsv"
OUT_ROOT, RESULTS_FILE_NAME = "test_reports", "sdxl_test_results.json"


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


def save_json_results(data, capture_trace, vae_on_device, encoders_on_device, use_cfg_parallel, num_inference_steps):
    os.makedirs(OUT_ROOT, exist_ok=True)
    trace_flag = "with_trace" if capture_trace else "no_trace"
    vae_flag = "device_vae" if vae_on_device else "host_vae"
    encoders_flag = "device_encoders" if encoders_on_device else "host_encoders"
    new_file_name = (
        f"sdxl_test_results_{trace_flag}_{vae_flag}_{encoders_flag}_{use_cfg_parallel}_{num_inference_steps}.json"
    )
    
    with open(f"{OUT_ROOT}/{new_file_name}", "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Test results saved to {OUT_ROOT}/{new_file_name}")

    with open(
        f"{OUT_ROOT}/{RESULTS_FILE_NAME}", "w"
    ) as f:  # this is for CI and test_sdxl_accuracy_with_reset.py compatibility
        json.dump(data, f, indent=4)
    logger.info(f"Test results saved to {OUT_ROOT}/{RESULTS_FILE_NAME}")