# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from loguru import logger

# Import from local reference files
from models.demos.deepseek_v3.reference.configuration_deepseek import DeepseekV3Config
from models.demos.deepseek_v3.reference.modeling_deepseek import MoEGate


def get_valid_group_combinations(group_scores, topk_groups, atol=0.075):
    """
    Get all valid combinations of topk_groups groups considering ties.

    Args:
        group_scores: 1D tensor of group scores
        topk_groups: Number of groups to select
        atol: Absolute tolerance for tie detection

    Returns:
        List of frozensets, each representing a valid group selection
    """
    from itertools import combinations

    n_groups = group_scores.shape[0]

    # Sort scores descending
    sorted_scores, sorted_indices = torch.sort(group_scores, descending=True)

    # Find the threshold (k-th highest score)
    threshold = sorted_scores[topk_groups - 1].item()

    # Partition groups into: strictly above threshold, at threshold (tied), strictly below
    above_threshold = []
    at_threshold = []
    for g in range(n_groups):
        score = group_scores[g].item()
        if score > threshold + atol:
            above_threshold.append(g)
        elif score >= threshold - atol:
            at_threshold.append(g)
        # else: below threshold, ignore

    # We must select all groups above threshold
    # Then select remaining from tied groups
    remaining_to_select = topk_groups - len(above_threshold)

    if remaining_to_select < 0:
        # More groups above threshold than needed - shouldn't happen with correct threshold
        raise ValueError(f"More groups ({len(above_threshold)}) above threshold than topk_groups ({topk_groups})")

    if remaining_to_select > len(at_threshold):
        # Not enough tied groups - shouldn't happen
        raise ValueError(f"Not enough tied groups ({len(at_threshold)}) to fill remaining ({remaining_to_select})")

    # Generate all valid combinations
    valid_combos = []
    for tied_selection in combinations(at_threshold, remaining_to_select):
        selected = frozenset(above_threshold) | frozenset(tied_selection)
        valid_combos.append(selected)

    return valid_combos


def get_valid_expert_sets(biased_scores_row, selected_groups, n_activated_experts, experts_per_group, atol=0.1):
    """
    Get all valid sets of expert indices for a given group selection, considering ties.

    Args:
        biased_scores_row: 1D tensor of biased scores for one row (all experts)
        selected_groups: Set of selected group indices
        n_activated_experts: Number of experts to select
        experts_per_group: Number of experts per group
        atol: Absolute tolerance for tie detection

    Returns:
        List of frozensets, each representing a valid expert selection
    """
    from itertools import combinations

    # Mask out non-selected groups
    masked_scores = biased_scores_row.clone()
    n_groups = len(biased_scores_row) // experts_per_group
    for g in range(n_groups):
        if g not in selected_groups:
            masked_scores[g * experts_per_group : (g + 1) * experts_per_group] = float("-inf")

    # Find the threshold (k-th highest score among valid experts)
    valid_scores = masked_scores[masked_scores > float("-inf")]
    if len(valid_scores) < n_activated_experts:
        raise ValueError(f"Not enough valid experts ({len(valid_scores)}) for selection ({n_activated_experts})")

    sorted_scores, _ = torch.sort(valid_scores, descending=True)
    threshold = sorted_scores[n_activated_experts - 1].item()

    # Partition experts into: strictly above threshold, at threshold (tied)
    above_threshold = []
    at_threshold = []
    for e in range(len(masked_scores)):
        score = masked_scores[e].item()
        if score > threshold + atol:
            above_threshold.append(e)
        elif score >= threshold - atol and score > float("-inf"):
            at_threshold.append(e)

    remaining_to_select = n_activated_experts - len(above_threshold)

    if remaining_to_select < 0 or remaining_to_select > len(at_threshold):
        # Edge case: just return the top-k as the only valid set
        _, top_indices = torch.topk(masked_scores, n_activated_experts)
        return [frozenset(top_indices.tolist())]

    # Generate all valid combinations
    valid_sets = []
    for tied_selection in combinations(at_threshold, remaining_to_select):
        selected = frozenset(above_threshold) | frozenset(tied_selection)
        valid_sets.append(selected)

    return valid_sets if valid_sets else [frozenset(above_threshold)]


def assert_in_valid_outcomes(
    ttnn_weights,
    ttnn_indices,
    scores,
    bias,
    n_groups,
    summed_experts_per_group,
    topk_groups,
    n_activated_experts,
    route_scale,
    epsilon,
    weight_rtol=0.02,
    weight_atol=0.01,
):
    """
    Verify that TTNN's output is one of the valid outcomes considering all possible
    tie-breaking decisions at both group and expert levels.

    This function:
    1. Computes all valid group combinations (considering group score ties)
    2. For each valid group combo, computes all valid expert selections (considering expert ties)
    3. Verifies TTNN's result matches one of these valid outcomes

    Args:
        ttnn_weights: TTNN output weights tensor
        ttnn_indices: TTNN output indices tensor
        scores: Original input scores (before sigmoid)
        bias: Bias tensor added for selection
        n_groups: Number of expert groups
        summed_experts_per_group: Number of top experts summed per group
        topk_groups: Number of top groups to select
        n_activated_experts: Number of experts to select
        route_scale: Scale factor for final weights
        epsilon: Epsilon for normalization stability
        weight_rtol: Relative tolerance for weight comparison
        weight_atol: Absolute tolerance for weight comparison
    """
    # Compute intermediate values
    sigmoid_scores = torch.sigmoid(scores)
    biased_scores = sigmoid_scores + bias
    experts_per_group = scores.shape[-1] // n_groups

    # Compute group scores
    grouped_biased = biased_scores.reshape(scores.shape[:-1] + (n_groups, experts_per_group))
    top_expert_scores, _ = torch.topk(grouped_biased, summed_experts_per_group, dim=-1)
    group_scores = top_expert_scores.sum(dim=-1)  # [batch, seq, n_groups]

    # Flatten for easier iteration
    original_shape = ttnn_weights.shape
    ttnn_weights_2d = ttnn_weights.reshape(-1, n_activated_experts)
    ttnn_indices_2d = ttnn_indices.reshape(-1, n_activated_experts).to(torch.int64)
    biased_scores_2d = biased_scores.reshape(-1, scores.shape[-1])
    sigmoid_scores_2d = sigmoid_scores.reshape(-1, scores.shape[-1])
    group_scores_2d = group_scores.reshape(-1, n_groups)

    num_rows = ttnn_weights_2d.shape[0]
    stats = {"exact_match": 0, "group_tie": 0, "expert_tie": 0, "both_tie": 0}

    for row in range(num_rows):
        row_group_scores = group_scores_2d[row]
        row_biased_scores = biased_scores_2d[row]
        row_sigmoid_scores = sigmoid_scores_2d[row]
        ttnn_expert_set = frozenset(ttnn_indices_2d[row].tolist())
        ttnn_group_set = frozenset((idx // experts_per_group) for idx in ttnn_indices_2d[row].tolist())

        # Get all valid group combinations
        valid_group_combos = get_valid_group_combinations(row_group_scores, topk_groups)

        # Check if TTNN's group selection is valid
        # Note: TTNN's groups may be a SUBSET of a valid combo if not all groups
        # have experts in the final top-k selection
        group_selection_valid = any(ttnn_group_set.issubset(combo) for combo in valid_group_combos)
        if not group_selection_valid:
            logger.info(f"Row {row}: TTNN group selection not subset of any valid combination")
            logger.info(f"  TTNN groups: {sorted(ttnn_group_set)}")
            logger.info(f"  Valid group combos: {[sorted(c) for c in valid_group_combos]}")
            logger.info(f"  Group scores: {row_group_scores.tolist()}")
            logger.info(f"  TTNN indices: {ttnn_indices_2d[row].tolist()}")
            raise AssertionError(f"Row {row}: Invalid group selection")

        # Find which valid combo(s) contain TTNN's groups
        matching_combos = [combo for combo in valid_group_combos if ttnn_group_set.issubset(combo)]

        # Get all valid expert sets for each matching group combo
        # TTNN's selection is valid if it matches any valid expert set from any matching combo
        all_valid_expert_sets = []
        for combo in matching_combos:
            valid_expert_sets = get_valid_expert_sets(row_biased_scores, combo, n_activated_experts, experts_per_group)
            all_valid_expert_sets.extend(valid_expert_sets)

        # Check if TTNN's expert selection is valid
        if ttnn_expert_set not in all_valid_expert_sets:
            logger.info(f"Row {row}: TTNN expert selection not in valid sets")
            logger.info(f"  TTNN experts: {sorted(ttnn_expert_set)}")
            logger.info(f"  Matching group combos: {[sorted(c) for c in matching_combos]}")
            logger.info(f"  Sample valid expert sets: {[sorted(s) for s in all_valid_expert_sets[:5]]}...")
            raise AssertionError(f"Row {row}: Invalid expert selection for the chosen groups")

        # Compute expected weights for TTNN's selection
        ttnn_indices_row = ttnn_indices_2d[row]
        expected_unbiased = row_sigmoid_scores[ttnn_indices_row]
        expected_normalized = expected_unbiased / (expected_unbiased.sum() + epsilon)
        expected_weights = expected_normalized * route_scale

        # Sort both by index for comparison
        ttnn_sort_idx = torch.argsort(ttnn_indices_row)
        ttnn_weights_sorted = ttnn_weights_2d[row][ttnn_sort_idx]
        expected_weights_sorted = expected_weights[ttnn_sort_idx]

        if not torch.allclose(
            ttnn_weights_sorted.float(), expected_weights_sorted.float(), rtol=weight_rtol, atol=weight_atol
        ):
            max_diff = (ttnn_weights_sorted.float() - expected_weights_sorted.float()).abs().max()
            logger.info(f"Row {row}: Weights don't match expected for selected experts")
            logger.info(f"  TTNN weights:     {ttnn_weights_sorted}")
            logger.info(f"  Expected weights: {expected_weights_sorted}")
            logger.info(f"  Max diff: {max_diff}")
            raise AssertionError(f"Row {row}: Weight mismatch (max_diff={max_diff})")

        # Track statistics
        is_group_tie = len(valid_group_combos) > 1
        is_expert_tie = len(all_valid_expert_sets) > 1
        if is_group_tie and is_expert_tie:
            stats["both_tie"] += 1
        elif is_group_tie:
            stats["group_tie"] += 1
        elif is_expert_tie:
            stats["expert_tie"] += 1
        else:
            stats["exact_match"] += 1

    logger.info(
        f"✓ All {num_rows} rows passed exhaustive validation: "
        f"{stats['exact_match']} exact, {stats['group_tie']} group ties, "
        f"{stats['expert_tie']} expert ties, {stats['both_tie']} both"
    )


def generate_distinct_sigmoid_inputs(shape, min_val=0.05, max_val=0.95, dtype=torch.bfloat16):
    """
    Generate random bfloat16 tensor where values are guaranteed to be distinct
    after sigmoid WITHIN EACH ROW (last dimension). This is achieved by:
    1. Generating distinct bfloat16 values in sigmoid output space (0, 1)
    2. Applying logit (inverse sigmoid) to get the inputs

    Args:
        shape: Output tensor shape
        min_val: Minimum sigmoid output value (avoid 0 for numerical stability)
        max_val: Maximum sigmoid output value (avoid 1 for numerical stability)
        dtype: Output dtype (default bfloat16)

    Returns:
        Tensor of shape `shape` where sigmoid(output) has distinct values per row
    """
    row_size = shape[-1]  # Number of elements per row (e.g., total_experts)
    num_rows = torch.tensor(shape[:-1]).prod().item()  # Total number of rows

    # Generate enough unique candidates for one row
    # Use more points than needed to ensure we have enough after deduplication
    num_candidates = row_size * 4
    candidates = torch.linspace(min_val, max_val, num_candidates, dtype=torch.float32)
    candidates_bf16 = candidates.to(dtype)

    # Get unique bfloat16 values
    unique_bf16 = candidates_bf16.unique()

    if unique_bf16.numel() < row_size:
        raise ValueError(
            f"Cannot generate {row_size} distinct bfloat16 sigmoid outputs per row in range "
            f"[{min_val}, {max_val}]. Only {unique_bf16.numel()} distinct values available."
        )

    # Generate distinct values for each row independently
    all_rows = []
    for _ in range(num_rows):
        # Randomly select row_size unique values for this row
        perm = torch.randperm(unique_bf16.numel())[:row_size]
        sigmoid_outputs = unique_bf16[perm]

        # Apply logit (inverse sigmoid) to get pre-sigmoid inputs
        # logit(p) = log(p / (1-p))
        # Do this in float32 for precision, then convert back
        sigmoid_outputs_f32 = sigmoid_outputs.float()
        row_inputs = torch.log(sigmoid_outputs_f32 / (1 - sigmoid_outputs_f32))
        all_rows.append(row_inputs)

    # Stack all rows and reshape to desired shape
    inputs = torch.stack(all_rows).to(dtype).reshape(shape)

    return inputs


def grouped_gate_golden(
    scores, bias, route_scale, epsilon, n_groups, summed_experts_per_group, topk_groups, n_activated_experts
):
    # first run sigmoid on scores
    scores = torch.sigmoid(scores)

    # then add bias (used for selection only)
    biased_scores = scores + bias
    # index3 = 3
    # logger.info(f"group 3 scores: {biased_scores[:, :, :, index3*32:(index3+1)*32]}")
    # index6 = 6
    # logger.info(f"group 6 scores: {biased_scores[:, :, :, index6*32:(index6+1)*32]}")

    # then reshape based on number of groups
    grouped_scores = biased_scores.reshape(scores.shape[:-1] + (n_groups, scores.shape[-1] // n_groups))

    # then sort the scores within each group
    top_p_experts_scores, _ = torch.topk(grouped_scores, summed_experts_per_group, dim=-1)

    # then sum the scores of the top p experts in each group
    summed_scores = top_p_experts_scores.sum(dim=-1, keepdim=False)
    logger.info(f"summed_scores: {summed_scores}")

    # find the top k groups
    _, top_k_groups_indices = torch.topk(summed_scores, topk_groups, dim=-1)
    logger.info(f"top_k_groups_indices: {top_k_groups_indices}")
    # logger.info(f"top_k_groups_indices: {top_k_groups_indices}")

    # Create a mask for valid groups
    # We initialize a mask of allowed groups and fill others with -inf
    group_mask = torch.ones(grouped_scores.shape[:-1], dtype=torch.bool, device=scores.device)
    group_mask.scatter_(-1, top_k_groups_indices, False)  # Set selected groups to False (keep)

    # Fill ignored groups with -inf
    masked_grouped_scores = grouped_scores.masked_fill(group_mask.unsqueeze(-1), float("-inf"))

    # reshape back to the original shape
    masked_scores = masked_grouped_scores.reshape(scores.shape)

    # then run topk to find expert indices
    _, top_k_experts_indices = torch.topk(masked_scores, n_activated_experts, dim=-1)

    # then gather the UNBIASED scores (original sigmoid output) based on the top k experts indices
    # The reference uses 'original_scores' (no bias) for the final weights
    chosen_scores = torch.gather(scores, dim=-1, index=top_k_experts_indices)
    logger.info(f"chosen_scores: {chosen_scores[-1, -1, -1, :]}")

    # normalize the chosen scores
    normalized_scores = chosen_scores / (chosen_scores.sum(dim=-1, keepdim=True) + epsilon)
    logger.info(f"normalized_scores: {normalized_scores[-1, -1, -1, :]}")

    # then scale the normalized scores by the scales
    scaled_scores = normalized_scores * route_scale
    logger.info(f"scaled_scores: {scaled_scores[-1, -1, -1, :]}")
    return scaled_scores, top_k_experts_indices


def test_grouped_gate_against_reference():
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)

    hf_config = DeepseekV3Config()
    moe_gate = MoEGate(hf_config, use_bitonic_sort=False).eval()
    # set the e_score_correction_bias to a random values
    moe_gate.e_score_correction_bias.data = torch.randn(
        moe_gate.e_score_correction_bias.data.shape, dtype=torch.bfloat16
    )

    seq_lens = [32, 128, 2048]
    batch_size = 1

    for seq_len in seq_lens:
        torch_input = torch.randn(batch_size, seq_len, hf_config.hidden_size, dtype=torch.bfloat16)

        reference_topk_indices, reference_scores = moe_gate.forward(torch_input)
        grouped_fn_topk_indices, grouped_fn_scores = moe_gate.grouped_forward(torch_input)

        assert torch.allclose(reference_topk_indices, grouped_fn_topk_indices)
        assert torch.allclose(reference_scores, grouped_fn_scores)


# Test parameter combinations - curated to avoid combinatorial explosion
# Format: (num_batches, batch_size, seq_len)
GROUPED_GATE_TEST_PARAMS = [
    # Basic cases with batch_size=1
    (1, 1, 1),  # Minimal case
    # (1, 1, 33),  # Just over one tile (edge case)
    # # Varying batch_size
    # (1, 8, 512),  # Larger sequence
    # (7, 7, 81),  # batch_size=8 with into second face
    # # Stress tests
    # (1, 1, 8192),
]


@pytest.mark.parametrize("num_batches,batch_size,seq_len", GROUPED_GATE_TEST_PARAMS)
def test_grouped_gate(device, num_batches, batch_size, seq_len):
    """
    Test grouped_gate operation with various batch sizes, sequence lengths, and batch counts.
    Route scale is randomized per test case.
    """
    # Create a deterministic seed based on test parameters
    seed = 0
    torch.manual_seed(seed)

    total_experts = 256
    n_groups = 8
    summed_experts_per_group = 2  # number of experts to sum per group
    topk_groups = 4  # top groups to keep
    n_activated_experts = 8  # chosen experts per token
    epsilon = 1e-20  # epsilon for stability

    # Random route_scale between 0.1 and 1.1
    route_scale = torch.rand(1).item() + 0.1

    logger.info(
        f"Testing: num_batches={num_batches}, batch_size={batch_size}, seq_len={seq_len}, route_scale={route_scale:.4f}"
    )

    # Use generate_distinct_sigmoid_inputs to avoid ties after sigmoid
    scores = generate_distinct_sigmoid_inputs((num_batches, batch_size, seq_len, total_experts), dtype=torch.bfloat16)
    bias = torch.randn(num_batches, batch_size, seq_len, total_experts, dtype=torch.bfloat16)

    torch_scores, torch_top_k_experts_indices = grouped_gate_golden(
        scores, bias, route_scale, epsilon, n_groups, summed_experts_per_group, topk_groups, n_activated_experts
    )

    ttnn_scores = ttnn.from_torch(scores, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_bias = ttnn.from_torch(bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_scores, ttnn_top_k_experts_indices = ttnn.experimental.deepseek_grouped_gate(
        ttnn_scores,
        ttnn_bias,
        n_groups=n_groups,
        summed_experts_per_group=summed_experts_per_group,
        topk_groups=topk_groups,
        n_activated_experts=n_activated_experts,
        route_scale=route_scale,
        epsilon=epsilon,
    )

    ttnn_scores = ttnn.to_torch(ttnn_scores)
    ttnn_top_k_experts_indices = ttnn.to_torch(ttnn_top_k_experts_indices)

    logger.info(f"torch_top_k_experts_indices: {torch_top_k_experts_indices[-1, -1, -1, :]}")
    logger.info(f"ttnn_top_k_experts_indices: {ttnn_top_k_experts_indices[-1, -1, -1, :]}")

    logger.info(f"torch_scores: {torch_scores[-1, -1, -1, :]}")
    logger.info(f"ttnn_scores: {ttnn_scores[-1, -1, -1, :]}")

    # Exhaustive validation: verify TTNN result is one of the valid outcomes
    # considering all possible tie-breaking decisions at group and expert levels
    assert_in_valid_outcomes(
        ttnn_scores,
        ttnn_top_k_experts_indices,
        scores,
        bias,
        n_groups,
        summed_experts_per_group,
        topk_groups,
        n_activated_experts,
        route_scale,
        epsilon,
        weight_rtol=0.02,
        weight_atol=0.01,
    )
