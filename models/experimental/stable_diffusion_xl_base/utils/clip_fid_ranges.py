# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# THRESHOLDS 5000 PROMPTS
FID_VALID_RANGE_5000 = (23.01085758, 23.95007626)
CLIP_VALID_RANGE_5000 = (31.68631873, 31.81331801)

FID_SCORE_APPROX_RANGE_5000 = (22.3205318526, 24.6685785478)  # Approximate ranges (+/- 3%) from valid range
CLIP_SCORE_APPROX_RANGE_5000 = (30.7357291681, 32.7677175503)  # Approximate ranges (+/- 3%) from valid range

CLIP_DELTA_RANGE_106_PERCENT_5000 = (31.6825087516, 31.8171279884)  # intuition: DELTA_RANGE_100_PERCENT == VALID_RANGE
FID_DELTA_RANGE_106_PERCENT_5000 = (22.982681019599998, 23.978252820399998)

# THRESHOLDS 100 PROMPTS - calculated from nvidia results
FID_VALID_RANGE_100 = (179.47917071312727, 186.80485117586642)
CLIP_VALID_RANGE_100 = (31.844935079606522, 31.972570096511394)

FID_APPROX_RANGE_100 = (178.78884498572728, 187.5233534636664)
CLIP_APPROX_RANGE_100 = (30.894345517706522, 32.92696963681139)

CLIP_DELTA_RANGE_106_PERCENT_100 = (31.84110602909938, 31.97639914701854)
FID_DELTA_RANGE_106_PERCENT_100 = (179.2594002992451, 187.02462158974862)

DELTA_UPPER_THRESHOLD = 1.0  # delta metric


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
        the given score is from the center of the valid range.
    Formula:
        abs(avg(range) - score) / ((max - min) * 1.06)
    Note:
        1.06 is used to adjust the range to the approximate range (+/- 3%)
    Usage:
        # For CLIP score:
        delta_clip = get_appr_delta_metric(clip_score, CLIP_SCORE_RANGE)

        # For FID score:
        delta_fid = get_appr_delta_metric(fid_score, FID_SCORE_RANGE)
    """
    assert score_type in {"fid", "clip"}, "score_type should be either 'fid' or 'clip'"
    if num_prompts not in {100, 5000}:
        return -1

    if score_type == "fid":
        valid_range_tuple = FID_VALID_RANGE_5000 if num_prompts == 5000 else FID_VALID_RANGE_100
    else:  # score_type == "clip"
        valid_range_tuple = CLIP_VALID_RANGE_5000 if num_prompts == 5000 else CLIP_VALID_RANGE_100

    min_val, max_val = valid_range_tuple
    avg_val = (min_val + max_val) / 2
    return abs(avg_val - score) * 2 / ((max_val - min_val) * 1.06)
