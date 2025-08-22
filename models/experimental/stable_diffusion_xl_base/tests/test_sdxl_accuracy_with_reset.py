# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import subprocess
import os
import statistics
import json
from PIL import Image

from loguru import logger
from models.experimental.stable_diffusion_xl_base.utils.fid_score import calculate_fid_score
from models.experimental.stable_diffusion_xl_base.tests.test_sdxl_accuracy import sdxl_get_prompts, check_clip_scores
from models.experimental.stable_diffusion_xl_base.utils.clip_encoder import CLIPEncoder
from models.experimental.stable_diffusion_xl_base.tests.test_sdxl_accuracy import OUT_ROOT, RESULTS_FILE_NAME

from conftest import is_6u, is_galaxy
from models.experimental.stable_diffusion_xl_base.conftest import get_device_name

IS_RING_6U_LOCAL = os.environ.get("RING_6U", "0") == "1"
DEVICE_NAME_LOCAL = get_device_name()

NEW_JSON_FILE_NAME = "sdxl_test_results_with_reset.json"
READ_JSON_FILE_NAME = RESULTS_FILE_NAME

IMAGES_PATH, IMAGE_NAME_BASE = "output", "output"


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
@pytest.mark.skipif(is_6u() or IS_RING_6U_LOCAL, reason="skip when 6u, as it does not support reset")
def test_accuracy_with_reset(
    vae_on_device,
    encoders_on_device,
    capture_trace,
    captions_path,
    coco_statistics_path,
    evaluation_range,
    reset_config,
):
    start_from, num_prompts = evaluation_range
    prompts = sdxl_get_prompts(captions_path, start_from, num_prompts)  # also asserts evaluation_range

    reset_bool, reset_period = reset_config
    if reset_bool:
        assert reset_period >= 2, "reset_period should be at least 2, so FID can be calculated"
    else:
        reset_period = num_prompts

    vae_str = "device_vae" if vae_on_device else "host_vae"
    trace_str = "with_trace" if capture_trace else "no_trace"
    encoders_str = "device_encoders" if encoders_on_device else "host_encoders"

    logger.info(
        f"Running test_accuracy_with_reset with vae_on_device={vae_on_device}, capture_trace={capture_trace}, reset_bool={reset_bool}, reset_period={reset_period}"
    )
    logger.info(f"start_from: {start_from}, num_prompts: {num_prompts}")

    total_denoising_time, total_vae_time = 0.0, 0.0
    min_inference_time, max_inference_time = float("inf"), float("-inf")

    for current_start in range(start_from, start_from + num_prompts, reset_period):
        current_num_prompts = min(reset_period, start_from + num_prompts - current_start)

        env = ["env"]
        prefix_for_throttle = [
            "TT_MM_THROTTLE_PERF=5"
        ]  # for galaxies it is a must, for other machines beeing super safe
        command = (
            env
            + prefix_for_throttle
            + [
                "pytest",
                "models/experimental/stable_diffusion_xl_base/tests/test_sdxl_accuracy.py",
                "--start-from",
                str(current_start),
                "--num-prompts",
                str(current_num_prompts),
                "-k",
                f"{vae_str} and {trace_str} and {encoders_str}",
            ]
        )
        subprocess.run(command, check=True)

        json_file_path = f"{OUT_ROOT}/{READ_JSON_FILE_NAME}"
        with open(json_file_path, "r") as f:
            data = json.load(f)

            total_denoising_time += data["benchmarks_summary"][0]["average_denoising_time"] * current_num_prompts
            total_vae_time += data["benchmarks_summary"][0]["average_vae_time"] * current_num_prompts
            min_inference_time = min(data["benchmarks_summary"][0]["min_inference_time"], min_inference_time)
            max_inference_time = max(data["benchmarks_summary"][0]["max_inference_time"], max_inference_time)

        if reset_bool and current_start + reset_period < start_from + num_prompts:
            if is_galaxy():
                subprocess.run(["tt-smi", "-r", "/opt/tt_metal_infra/host-scripts/reset.json"], check=True)
            else:
                subprocess.run(["tt-smi", "-r"], check=True)
            logger.info(
                f"reset done for device {DEVICE_NAME_LOCAL} after {current_start + current_num_prompts} prompts"
            )
            logger.info("sleeping for 60 seconds to allow reset to complete safely")
            subprocess.run(
                ["sleep", "60"], check=True
            )  # for galaxies it is a must, for other machines beeing super safe

    images = sdxl_collect_images(start_from, num_prompts)

    clip = CLIPEncoder()

    clip_scores = []

    for idx, image in enumerate(images):
        clip_scores.append(100 * clip.get_clip_score(prompts[idx], image).item())
    check_clip_scores(start_from, num_prompts, prompts, clip_scores)

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

    average_denoising_time = total_denoising_time / num_prompts
    average_vae_time = total_vae_time / num_prompts

    data = {
        "model": "sdxl",  # For compatibility with current processes
        "metadata": {
            "device": DEVICE_NAME_LOCAL,
            "device_vae": vae_on_device,
            "start_from": start_from,
            "num_prompts": num_prompts,
            "model_name": "sdxl",
        },
        "benchmarks_summary": [
            {
                "device": DEVICE_NAME_LOCAL,
                "model": "sdxl",
                "average_denoising_time": average_denoising_time,
                "average_vae_time": average_vae_time,
                "average_inference_time": average_denoising_time + average_vae_time,
                "min_inference_time": min_inference_time,
                "max_inference_time": max_inference_time,
                "average_clip": average_clip_score,
                "deviation_clip": deviation_clip_score,
                "fid_score": fid_score,
            }
        ],
    }

    os.makedirs(OUT_ROOT, exist_ok=True)

    with open(f"{OUT_ROOT}/{NEW_JSON_FILE_NAME}", "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"Test results saved to {OUT_ROOT}/{NEW_JSON_FILE_NAME}")


def sdxl_collect_images(start_from, num_prompts):
    assert (
        0 <= start_from < 5000 and start_from + num_prompts <= 5000
    ), "start_from must be between 0 and 4999, and start_from + num_prompts must not exceed 5000."

    collected_images = []
    for index in range(start_from, start_from + num_prompts):
        current_filename_path = f"{IMAGES_PATH}/{IMAGE_NAME_BASE}{index+1}.png"
        img = Image.open(current_filename_path).convert("RGB")
        collected_images.append(img)
    return collected_images
