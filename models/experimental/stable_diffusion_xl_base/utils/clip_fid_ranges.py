# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import json

TARGET_JSON_PATH = "models/experimental/stable_diffusion_xl_base/targets/targets.json"
SDXL_DATASET_SIZE, SDXL_INPAINT_DATASET_SIZE, QUICK_TEST_DATASET_SIZE = 5000, 1260, 100
STANDARD_DATASET_SIZES = {SDXL_DATASET_SIZE, SDXL_INPAINT_DATASET_SIZE, QUICK_TEST_DATASET_SIZE}


def get_model_targets(model_name):
    with open(TARGET_JSON_PATH) as f:
        model_targets_json = json.load(f)

    model_targets = model_targets_json[model_name.removesuffix("-tp")]
    if model_name.endswith("-tp"):
        model_targets["perf"] = model_targets.pop("perf-tp")

    return model_targets


get_approx = lambda range_tuple: (range_tuple[0] * 0.97, range_tuple[1] * 1.03)


def using_full_dataset(model_name, num_prompts):
    if model_name in {
        "sdxl",
        "sdxl-tp",
        "sdxl-base-refiner",
        "sdxl-base-refiner-tp",
        "sdxl-img2img",
        "sdxl-img2img-tp",
    }:
        return num_prompts == SDXL_DATASET_SIZE
    elif model_name in {"sdxl-inpaint", "sdxl-inpaint-tp"}:
        return num_prompts == SDXL_INPAINT_DATASET_SIZE
    assert False, f"Unknown model name {model_name}"


def accuracy_check_fid(model_name, score, num_prompts, mode):
    # code 0 - invalid input, code 3 - out of range, code 2 - within range, this is for CI dashboard compatibility
    assert mode in {"valid", "approx", "delta"}, "mode should be either valid, approx, or delta"
    if num_prompts not in STANDARD_DATASET_SIZES or score == -1:
        return 0

    using_full_dataset_flag = using_full_dataset(model_name, num_prompts)
    targets = get_model_targets(model_name)
    if mode == "valid":
        range_tuple = (
            targets["accuracy"]["fid_valid_range_full_dataset"]
            if using_full_dataset_flag
            else targets["accuracy"]["fid_valid_range_100"]
        )
    elif mode == "approx":
        range_tuple = (
            get_approx(targets["accuracy"]["fid_valid_range_full_dataset"])
            if using_full_dataset_flag
            else get_approx(targets["accuracy"]["fid_valid_range_100"])
        )
    elif mode == "delta":
        delta_score = get_appr_delta_metric(model_name, score, num_prompts, "fid")
        return 2 if delta_score <= 0.5 else 3

    return 2 if score >= range_tuple[0] and score <= range_tuple[1] else 3


def accuracy_check_clip(model_name, score, num_prompts, mode):
    # code 0 - invalid input, code 3 - out of range, code 2 - within range, this is for CI dashboard compatibility
    assert mode in {"valid", "approx", "delta"}, "mode should be either valid, approx, or delta"
    if num_prompts not in STANDARD_DATASET_SIZES or score == -1:
        return 0

    using_full_dataset_flag = using_full_dataset(model_name, num_prompts)
    targets = get_model_targets(model_name)
    if mode == "valid":
        range_tuple = (
            targets["accuracy"]["clip_valid_range_full_dataset"]
            if using_full_dataset_flag
            else targets["accuracy"]["clip_valid_range_100"]
        )
    elif mode == "approx":
        range_tuple = (
            get_approx(targets["accuracy"]["clip_valid_range_full_dataset"])
            if using_full_dataset_flag
            else get_approx(targets["accuracy"]["clip_valid_range_100"])
        )
    elif mode == "delta":
        delta_score = get_appr_delta_metric(model_name, score, num_prompts, "clip")
        return 2 if delta_score <= 0.5 else 3

    return 2 if score >= range_tuple[0] and score <= range_tuple[1] else 3


def get_appr_delta_metric(model_name, score, num_prompts, score_type):
    """
    Intuition:
        Delta metric quantifies how far (as a percentage of the true metric interval)
        the given score is from the center of the valid range, it shouldnt exceed 50%
    Formula:
        abs(avg(range) - score) / ((max - min) * 1.06)
    Note:
        1.06 is used to adjust the range to the approximate range (+/- 3%)
    """
    assert score_type in {"fid", "clip"}, "score_type should be either fid or clip"
    if num_prompts not in STANDARD_DATASET_SIZES:
        return -1

    using_full_dataset_flag = using_full_dataset(model_name, num_prompts)
    targets = get_model_targets(model_name)
    if score_type == "fid":
        valid_range_tuple = (
            targets["accuracy"]["fid_valid_range_full_dataset"]
            if using_full_dataset_flag
            else targets["accuracy"]["fid_valid_range_100"]
        )
    else:
        valid_range_tuple = (
            targets["accuracy"]["clip_valid_range_full_dataset"]
            if using_full_dataset_flag
            else targets["accuracy"]["clip_valid_range_100"]
        )

    avg_val = (valid_range_tuple[0] + valid_range_tuple[1]) / 2
    return abs(avg_val - score) / ((valid_range_tuple[1] - valid_range_tuple[0]) * 1.06)
