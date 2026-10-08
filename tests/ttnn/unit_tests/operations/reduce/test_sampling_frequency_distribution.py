# SPDX-FileCopyrightText: © 2025-2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import math
import pytest
import torch
import torch.nn.functional as F

try:
    import ttnn
    from models.common.utility_functions import is_wormhole_b0, is_blackhole
    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False


def chi_square_goodness_of_fit(observed, expected):
    """
    Compute Pearson Chi-Square test statistic.
    chi2 = sum((O_i - E_i)^2 / E_i)
    """
    assert len(observed) == len(expected)
    chi2 = 0.0
    for o, e in zip(observed, expected):
        assert e > 0, f"Expected count must be positive, got {e}"
        chi2 += ((float(o) - float(e)) ** 2) / float(e)
    return chi2


def normal_z_score(observed_count, expected_prob, total_trials):
    """
    Compute standard z-score for binomial observation:
    z = (O - N*p) / sqrt(N * p * (1 - p))
    """
    n_p = total_trials * expected_prob
    variance = total_trials * expected_prob * (1.0 - expected_prob)
    std_dev = math.sqrt(variance)
    return (observed_count - n_p) / std_dev


@pytest.mark.skipif(not TTNN_AVAILABLE, reason="ttnn not available")
def test_sampling_two_candidate_low_probability_frequency(device):
    """
    Regression test for tenstorrent/tt-metal#59732 (,000 bounty):
    Verify that tokens with low probabilities (below the BF16 mantissa resolution of 1/128 ~ 0.0078125)
    are sampled accurately without truncation or zero-frequency bias.
    """
    num_users = 32
    vocab_size = 64
    trials_per_batch = 100
    total_trials = num_users * trials_per_batch

    # Construct logits where token 0 has high probability (~0.998) and token 1 has tail probability p ~ 0.002
    # In bfloat16 (7 mantissa bits), 0.002 falls below the 1/128 step and was previously never sampled (count = 0).
    target_p_tail = 0.002
    logit_0 = math.log(1.0 - target_p_tail)
    logit_1 = math.log(target_p_tail)

    logits_row = torch.full((vocab_size,), -1e4, dtype=torch.float32)
    logits_row[0] = logit_0
    logits_row[1] = logit_1
    input_values = logits_row.unsqueeze(0).unsqueeze(0).repeat(1, 1, num_users, 1)

    input_indices = torch.arange(vocab_size, dtype=torch.int32).unsqueeze(0).unsqueeze(0).repeat(1, 1, num_users, 1)

    k_list = [32] * num_users
    p_list = [0.0] * num_users  # pure top-k sampling

    input_values_tt = ttnn.from_torch(input_values, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    input_indices_tt = ttnn.from_torch(input_indices, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    k_tensor = ttnn.from_torch(torch.tensor(k_list), device=device, dtype=ttnn.uint32 if (is_wormhole_b0() or is_blackhole()) else ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    p_tensor = ttnn.from_torch(torch.tensor(p_list), device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    temp_tensor = ttnn.from_torch(torch.ones(num_users), device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    observed_tail_counts = 0

    for trial_idx in range(trials_per_batch):
        out_tt = ttnn.sampling(
            input_values_tt,
            input_indices_tt,
            k=k_tensor,
            p=p_tensor,
            temp=temp_tensor,
            seed=1000 + trial_idx,
        )
        out_torch = ttnn.to_torch(out_tt).view(num_users)
        observed_tail_counts += (out_torch == 1).sum().item()

    # With N = 3200 trials and p = 0.002, expected counts = 6.4
    # Under BF16 truncation, count was identically 0.
    # Under FP32 resolution, count is statistically consistent with binomial distribution.
    expected_count = total_trials * target_p_tail
    z = normal_z_score(observed_tail_counts, target_p_tail, total_trials)

    assert observed_tail_counts > 0, (
        f"Regression detected: Low-probability candidate (p={target_p_tail}) was NEVER sampled "
        f"over {total_trials} trials (expected ~{expected_count:.1f}). FP32 random threshold failed."
    )
    assert abs(z) < 3.89, (
        f"Sampling frequency significantly biased: observed {observed_tail_counts}, "
        f"expected {expected_count:.1f}, z-score = {z:.2f} (exceeds 3.89 limit for p < 1e-4)"
    )


@pytest.mark.skipif(not TTNN_AVAILABLE, reason="ttnn not available")
def test_sampling_multi_candidate_distribution_and_homogeneity(device):
    """
    Acceptance Criteria Test:
    Verify that equal-probability candidates are sampled homogeneously and tail candidates
    follow the theoretical reference distribution using Chi-Square Goodness-of-Fit.
    """
    num_users = 32
    vocab_size = 32
    trials_per_batch = 50
    total_trials = num_users * trials_per_batch

    # 4 equal tokens (0.24 each) and 1 tail token (0.04)
    probs = [0.24, 0.24, 0.24, 0.24, 0.04]
    logits = [math.log(p) for p in probs]
    logits_row = torch.full((vocab_size,), -1e4, dtype=torch.float32)
    for idx, l in enumerate(logits):
        logits_row[idx] = l

    input_values = logits_row.unsqueeze(0).unsqueeze(0).repeat(1, 1, num_users, 1)
    input_indices = torch.arange(vocab_size, dtype=torch.int32).unsqueeze(0).unsqueeze(0).repeat(1, 1, num_users, 1)

    k_list = [5] * num_users
    p_list = [0.0] * num_users

    input_values_tt = ttnn.from_torch(input_values, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    input_indices_tt = ttnn.from_torch(input_indices, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    k_tensor = ttnn.from_torch(torch.tensor(k_list), device=device, dtype=ttnn.uint32 if (is_wormhole_b0() or is_blackhole()) else ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    p_tensor = ttnn.from_torch(torch.tensor(p_list), device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    temp_tensor = ttnn.from_torch(torch.ones(num_users), device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    counts = [0] * len(probs)

    for trial_idx in range(trials_per_batch):
        out_tt = ttnn.sampling(
            input_values_tt,
            input_indices_tt,
            k=k_tensor,
            p=p_tensor,
            temp=temp_tensor,
            seed=5000 + trial_idx,
        )
        out_torch = ttnn.to_torch(out_tt).view(num_users)
        for val in out_torch.tolist():
            if 0 <= val < len(probs):
                counts[val] += 1

    expected_counts = [total_trials * p for p in probs]
    chi2 = chi_square_goodness_of_fit(counts, expected_counts)

    # For df = 4 (5 categories), chi2 critical value at alpha = 0.001 is 18.47
    assert chi2 < 18.47, (
        f"Chi-Square goodness-of-fit rejected null hypothesis: chi2 = {chi2:.2f} > 18.47. "
        f"Observed={counts}, Expected={expected_counts}"
    )

    # Verify homogeneity between equal candidates (0, 1, 2, 3)
    equal_counts = counts[:4]
    mean_equal = sum(equal_counts) / 4.0
    for idx, c in enumerate(equal_counts):
        relative_diff = abs(c - mean_equal) / mean_equal
        assert relative_diff < 0.15, (
            f"Homogeneity violation for equal candidate {idx}: count={c}, mean={mean_equal:.1f}, "
            f"relative deviation={relative_diff:.1%}"
        )
