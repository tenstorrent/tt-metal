# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# THRESHOLDS 5000 PROMPTS
FID_VALID_RANGE_5000 = (23.01085758, 23.95007626)
CLIP_VALID_RANGE_5000 = (31.68631873, 31.81331801)

FID_SCORE_APPROX_RANGE_5000 = (22.3205318526, 24.6685785478)  # Approximate ranges (+/- 3%) from valid range
CLIP_SCORE_APPROX_RANGE_5000 = (30.7357291681, 32.7677175503)  # Approximate ranges (+/- 3%) from valid range

FID_DELTA_RANGE_106_PERCENT_5000 = (22.982681019599998, 23.978252820399998)
CLIP_DELTA_RANGE_106_PERCENT_5000 = (31.6825087516, 31.8171279884)  # intuition: DELTA_RANGE_100_PERCENT == VALID_RANGE

# THRESHOLDS 100 PROMPTS - calculated from nvidia results
FID_VALID_RANGE_100 = (181.1513318972489, 184.97865376919088)
CLIP_VALID_RANGE_100 = (31.65430683222675, 32.15949391210697)

FID_APPROX_RANGE_100 = (179.20542453436752, 186.9656737959027)
CLIP_APPROX_RANGE_100 = (30.70704558232445, 33.12187298301411)

FID_DELTA_RANGE_106_PERCENT_100 = (181.03651224109063, 185.09347342534915)
CLIP_DELTA_RANGE_106_PERCENT_100 = (31.639151219830346, 32.17464952450337)


def accuracy_check_fid(score, num_prompts, mode):
    assert mode in {"valid", "approx", "delta"}, "mode should be either 'valid' or 'approx' or 'delta'"
    if num_prompts not in {100, 5000} or score == -1:
        return 0

    if mode == "valid":
        range_tuple = FID_VALID_RANGE_5000 if num_prompts == 5000 else FID_VALID_RANGE_100
    elif mode == "approx":
        range_tuple = FID_SCORE_APPROX_RANGE_5000 if num_prompts == 5000 else FID_APPROX_RANGE_100
    elif mode == "delta":
        range_tuple = FID_DELTA_RANGE_106_PERCENT_5000 if num_prompts == 5000 else FID_DELTA_RANGE_106_PERCENT_100

    return 3 if score >= range_tuple[0] and score <= range_tuple[1] else 2


def accuracy_check_clip(score, num_prompts, mode):
    assert mode in {"valid", "approx", "delta"}, "mode should be either 'valid', 'approx', or 'delta'"
    if num_prompts not in {100, 5000} or score == -1:
        return 0

    if mode == "valid":
        range_tuple = CLIP_VALID_RANGE_5000 if num_prompts == 5000 else CLIP_VALID_RANGE_100
    elif mode == "approx":
        range_tuple = CLIP_SCORE_APPROX_RANGE_5000 if num_prompts == 5000 else CLIP_APPROX_RANGE_100
    elif mode == "delta":
        range_tuple = CLIP_DELTA_RANGE_106_PERCENT_5000 if num_prompts == 5000 else CLIP_DELTA_RANGE_106_PERCENT_100

    return 3 if score >= range_tuple[0] and score <= range_tuple[1] else 2


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
    assert score_type in {"fid", "clip"}, "score_type should be either 'fid' or 'clip'"
    if num_prompts not in {100, 5000}:
        return -1

    if score_type == "fid":
        valid_range_tuple = FID_VALID_RANGE_5000 if num_prompts == 5000 else FID_VALID_RANGE_100
    else:
        valid_range_tuple = CLIP_VALID_RANGE_5000 if num_prompts == 5000 else CLIP_VALID_RANGE_100

    avg_val = (valid_range_tuple[0] + valid_range_tuple[1]) / 2
    return abs(avg_val - score) / ((valid_range_tuple[1] - valid_range_tuple[0]) * 1.06)
