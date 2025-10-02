# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import json

TARGET_JSON_PATH = "models/experimental/stable_diffusion_xl_base/targets/targets.json"

with open(TARGET_JSON_PATH) as f:
    targets = json.load(f)

get_approx = lambda range_tuple: (range_tuple[0] * 0.97, range_tuple[1] * 1.03)


def accuracy_check_fid(score, num_prompts, mode):
    # code 0 - invalid input, code 3 - out of range, code 2 - within range, this is for CI dashboard compatibility
    assert mode in {"valid", "approx", "delta"}, "mode should be either valid, approx, or delta"
    if num_prompts not in {100, 5000} or score == -1:
        return 0

    if mode == "valid":
        range_tuple = (
            targets["accuracy"]["fid_valid_range_5000"]
            if num_prompts == 5000
            else targets["accuracy"]["fid_valid_range_100"]
        )
    elif mode == "approx":
        range_tuple = (
            get_approx(targets["accuracy"]["fid_valid_range_5000"])
            if num_prompts == 5000
            else get_approx(targets["accuracy"]["fid_valid_range_100"])
        )
    elif mode == "delta":
        delta_score = get_appr_delta_metric(score, num_prompts, "fid")
        return 2 if delta_score <= 0.5 else 3

    return 2 if score >= range_tuple[0] and score <= range_tuple[1] else 3


def accuracy_check_clip(score, num_prompts, mode):
    # code 0 - invalid input, code 3 - out of range, code 2 - within range, this is for CI dashboard compatibility
    assert mode in {"valid", "approx", "delta"}, "mode should be either valid, approx, or delta"
    if num_prompts not in {100, 5000} or score == -1:
        return 0

    if mode == "valid":
        range_tuple = (
            targets["accuracy"]["clip_valid_range_5000"]
            if num_prompts == 5000
            else targets["accuracy"]["clip_valid_range_100"]
        )
    elif mode == "approx":
        range_tuple = (
            get_approx(targets["accuracy"]["clip_valid_range_5000"])
            if num_prompts == 5000
            else get_approx(targets["accuracy"]["clip_valid_range_100"])
        )
    elif mode == "delta":
        delta_score = get_appr_delta_metric(score, num_prompts, "clip")
        return 2 if delta_score <= 0.5 else 3

    return 2 if score >= range_tuple[0] and score <= range_tuple[1] else 3


def get_appr_delta_metric(score, num_prompts, score_type):
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
    if num_prompts not in {100, 5000}:
        return -1

    if score_type == "fid":
        valid_range_tuple = (
            targets["accuracy"]["fid_valid_range_5000"]
            if num_prompts == 5000
            else targets["accuracy"]["fid_valid_range_100"]
        )
    else:
        valid_range_tuple = (
            targets["accuracy"]["clip_valid_range_5000"]
            if num_prompts == 5000
            else targets["accuracy"]["clip_valid_range_100"]
        )

    avg_val = (valid_range_tuple[0] + valid_range_tuple[1]) / 2
    return abs(avg_val - score) / ((valid_range_tuple[1] - valid_range_tuple[0]) * 1.06)
