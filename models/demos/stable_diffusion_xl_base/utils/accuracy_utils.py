# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import csv
import json
import os
import statistics
import urllib

from loguru import logger

from models.common.utility_functions import profiler
from models.demos.stable_diffusion_xl_base.utils.clip_encoder import CLIPEncoder
from models.demos.stable_diffusion_xl_base.utils.clip_fid_ranges import (
    STANDARD_DATASET_SIZES,
    accuracy_check_clip,
    accuracy_check_fid,
    get_appr_delta_metric,
    get_model_targets,
    using_full_dataset,
)
from models.demos.stable_diffusion_xl_base.utils.fid_score import calculate_fid_score

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


def check_clip_scores(model_name, evaluation_range, prompts, clip_scores):
    start_from, num_prompts = evaluation_range
    targets = get_model_targets(model_name)
    warning_threshold, error_threshold = (
        targets["clip_score_thresholds"]["warning"],
        targets["clip_score_thresholds"]["error"],
    )

    assert len(clip_scores) == num_prompts == len(prompts), f"Expected {num_prompts} CLIP scores and prompts."
    logger.info(
        f"CLIP score error threshold: {error_threshold}, warning threshold: {warning_threshold}, for model {model_name}"
    )
    num_of_very_low_clip_scores = 0
    for idx, score in enumerate(clip_scores):
        if clip_scores[idx] < warning_threshold:
            if clip_scores[idx] < error_threshold:
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


def calculate_accuracy_metrics(images, prompts, coco_statistics_path):
    assert len(images) == len(
        prompts
    ), f"Expected same number of images and prompts, got {len(images)} images and {len(prompts)} prompts."

    clip = CLIPEncoder()

    clip_scores = []

    for idx, image in enumerate(images):
        clip_scores.append(100 * clip.get_clip_score(prompts[idx], image).item())

    average_clip_score = sum(clip_scores) / len(clip_scores)

    deviation_clip_score = "N/A"
    fid_score = "N/A"

    if len(prompts) >= 2:
        deviation_clip_score = statistics.stdev(clip_scores)
        fid_score = calculate_fid_score(images, coco_statistics_path)
    else:
        logger.info("FID score is not calculated for less than 2 prompts.")

    accuracy_metrics = {
        "clip_scores": clip_scores,
        "average_clip_score": average_clip_score,
        "deviation_clip_score": deviation_clip_score,
        "fid_score": fid_score,
    }

    return accuracy_metrics


def get_benchmark_summary(metadata):
    targets = get_model_targets(metadata["model_name"])
    avg_gen_end_to_end = profiler.get("end_to_end_generation")
    return [
        {
            "model": metadata["model_name"],
            "device": metadata["device"],
            "avg_gen_time": avg_gen_end_to_end,
            "target_checks": {
                "functional": {
                    "avg_gen_time": targets["perf"]["functional"],
                    "avg_gen_time_check": 2 if targets["perf"]["functional"] >= avg_gen_end_to_end else 3,
                },
                "complete": {
                    "avg_gen_time": targets["perf"]["complete"],
                    "avg_gen_time_check": 2 if targets["perf"]["complete"] >= avg_gen_end_to_end else 3,
                },
                "target": {
                    "avg_gen_time": targets["perf"]["target"],
                    "avg_gen_time_check": 2 if targets["perf"]["target"] >= avg_gen_end_to_end else 3,
                },
            },
            "average_denoising_time": profiler.get("denoising_loop"),
            "average_vae_time": profiler.get("vae_decode"),
            "min_gen_time": min(profiler.times["end_to_end_generation"]),
            "max_gen_time": max(profiler.times["end_to_end_generation"]),
            "average_encoding_time": profiler.get("encode_prompts"),
            "average_read_output_tensor": profiler.get("read_output_tensor"),
            "average_prepare_input_tensors": profiler.get("prepare_input_tensors"),
            "average_allocate_input_tensors": profiler.get("allocate_input_tensors"),
            "average_create_user_tensors": profiler.get("create_user_tensors"),
            "average_diffusion_pipeline_from_pretrained": profiler.get("diffusion_pipeline_from_pretrained"),
            "average_all_gather": profiler.get("all_gather"),
            "average_noise_pred_interleaved": profiler.get("noise_pred_interleaved"),
            "average_unet_uncond": profiler.get("unet_uncond"),
            "average_unet_text": profiler.get("unet_text"),
            "average_after_all_gather": profiler.get("after_all_gather"),
            "average_cfg_parallel_unet": profiler.get("cfg_parallel_unet"),
        }
    ]


def create_report_json(metadata, accuracy_metrics):
    model_name, device_name = metadata["model_name"], metadata["device"]
    average_clip_score, deviation_clip_score = (
        accuracy_metrics["average_clip_score"],
        accuracy_metrics["deviation_clip_score"],
    )
    fid_score, num_prompts = accuracy_metrics["fid_score"], metadata["num_prompts"]

    report_json = {
        "model": model_name,
        "metadata": metadata,
        "benchmarks_summary": get_benchmark_summary(metadata),
        "evals": [
            {
                "model": model_name,
                "device": device_name,
                "average_clip": average_clip_score,
                "deviation_clip": deviation_clip_score,
                "delta_clip": get_appr_delta_metric(model_name, average_clip_score, num_prompts, score_type="clip"),
                "fid_score": fid_score,
                "delta_fid": get_appr_delta_metric(model_name, fid_score, num_prompts, score_type="fid"),
                "accuracy_check": min(
                    accuracy_check_fid(model_name, fid_score, num_prompts, mode="approx"),
                    accuracy_check_clip(model_name, average_clip_score, num_prompts, mode="approx"),
                ),
            }
        ],
    }

    return report_json


def accuracy_assert(metadata, accuracy_metrics):
    model_name = metadata["model_name"]
    num_prompts = metadata["num_prompts"]
    clip_score = accuracy_metrics["average_clip_score"]
    fid_score = accuracy_metrics.get("fid_score", None)

    if num_prompts not in STANDARD_DATASET_SIZES:
        logger.info(f"Skipping accuracy range check: {num_prompts} prompts is not a standard dataset size")
        return

    targets = get_model_targets(model_name)
    using_full = using_full_dataset(model_name, num_prompts)
    label = "full dataset" if using_full else f"{num_prompts} prompts"

    clip_key = "clip_valid_range_full_dataset" if using_full else "clip_valid_range_100"
    fid_key = "fid_valid_range_full_dataset" if using_full else "fid_valid_range_100"

    clip_range = targets["accuracy"][clip_key]
    fid_range = targets["accuracy"][fid_key] if fid_key in targets["accuracy"] else None

    logger.info(f"Accuracy check | {model_name} | {label}")
    logger.info(f"  CLIP valid range : {clip_range},  obtained: {clip_score:.6f}")
    if fid_range:
        logger.info(f"  FID  valid range : {fid_range},  obtained: {fid_score:.6f}")

    failures = []

    if not (clip_range[0] <= clip_score <= clip_range[1]):
        dist = clip_range[0] - clip_score if clip_score < clip_range[0] else clip_score - clip_range[1]
        logger.debug(f"  [FAIL] CLIP {clip_score:.6f} outside {clip_range}, distance from boundary: {dist:.6f}")
        failures.append("CLIP out of valid range")

    if fid_range and not (fid_range[0] <= fid_score <= fid_range[1]):
        dist = fid_range[0] - fid_score if fid_score < fid_range[0] else fid_score - fid_range[1]
        logger.debug(f"  [FAIL] FID  {fid_score:.6f} outside {fid_range}, distance from boundary: {dist:.6f}")
        failures.append("FID out of valid range")

    assert not failures, "Accuracy test FAILED: " + ", ".join(failures)


def save_report_json(report_json, metadata):
    os.makedirs(OUT_ROOT, exist_ok=True)

    model_name = metadata["model_name"].removesuffix("-tp")
    trace_flag = "with_trace" if metadata["capture_trace"] else "no_trace"
    vae_flag = "device_vae" if metadata["device_vae"] else "host_vae"
    encoders_flag = "device_encoders" if metadata["encoders_on_device"] else "host_encoders"
    cfg_parallel_flag = "cfg_parallel" if metadata["use_cfg_parallel"] else "no_cfg_parallel"

    new_file_name = f"{model_name}-{trace_flag}-{vae_flag}-{encoders_flag}-{cfg_parallel_flag}.json"
    with open(f"{OUT_ROOT}/{new_file_name}", "w") as f:
        json.dump(report_json, f, indent=4)

    logger.info(f"Test results saved to {OUT_ROOT}/{new_file_name}")

    with open(f"{OUT_ROOT}/{RESULTS_FILE_NAME}", "w") as f:  # this is for CI compatibility
        json.dump(report_json, f, indent=4)

    logger.info(f"Test results saved to {OUT_ROOT}/{RESULTS_FILE_NAME}")
