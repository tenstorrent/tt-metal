# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Validation utilities for MoE dispatch/combine tests."""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import torch
from loguru import logger

from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping


def score_activation(logits: torch.Tensor, score_func: str) -> torch.Tensor:
    """Router affinity activation matching the moe_grouped_topk ``score_func`` option.

    "sigmoid" for DeepSeek-V3 / Kimi, "sqrtsoftplus" (== sqrt(softplus(x)), beta=1, threshold=20)
    for DeepSeek-V4.
    """
    if score_func == "sigmoid":
        return torch.sigmoid(logits)
    if score_func == "sqrtsoftplus":
        return torch.sqrt(torch.nn.functional.softplus(logits))
    raise ValueError(f"Unsupported score_func '{score_func}'")


def grouped_gate_golden_act(
    logits,
    bias,
    route_scale,
    epsilon,
    n_groups,
    summed_experts_per_group,
    topk_groups,
    n_activated_experts,
    score_func="sigmoid",
):
    """Activation-parametrized port of MoEGate.grouped_gate_golden.

    Identical to the DeepSeek-V3 reference grouped gate (same top-k ordering convention as the
    moe_grouped_topk device op) except the router affinity activation is selectable. With
    ``n_groups == 1`` it collapses to a plain top-k, matching the single-group device path used by
    Kimi and DeepSeek-V4. Returns ``(top_k_indices, scaled_weights)``.
    """
    scores = score_activation(logits, score_func)
    biased_scores = scores + bias

    grouped_scores = biased_scores.reshape(scores.shape[:-1] + (n_groups, scores.shape[-1] // n_groups))
    top_p_experts_scores, _ = torch.topk(grouped_scores, summed_experts_per_group, dim=-1, sorted=True)
    summed_scores = top_p_experts_scores.sum(dim=-1, keepdim=False)

    _, top_k_groups_indices = torch.topk(summed_scores, topk_groups, dim=-1, sorted=True)
    group_mask = torch.ones(grouped_scores.shape[:-1], dtype=torch.bool, device=scores.device)
    group_mask.scatter_(-1, top_k_groups_indices, False)
    masked_grouped_scores = grouped_scores.masked_fill(group_mask.unsqueeze(-1), float("-inf"))
    masked_scores = masked_grouped_scores.reshape(scores.shape)

    _, top_k_experts_indices = torch.topk(masked_scores, n_activated_experts, dim=-1, sorted=True)
    chosen_scores = torch.gather(scores, dim=-1, index=top_k_experts_indices)
    normalized_scores = chosen_scores / (chosen_scores.sum(dim=-1, keepdim=True) + epsilon)
    scaled_scores = normalized_scores * route_scale
    return top_k_experts_indices, scaled_scores


def hash_gate_golden_act(
    logits,
    input_ids,
    tid2eid,
    route_scale,
    epsilon,
    n_activated_experts,
    score_func="sqrtsoftplus",
):
    """PyTorch golden for the DeepSeek-V4 hash gate (matches ttnn ... moe_hash_gate).

    Expert selection is the static ``tid2eid[input_ids]`` lookup (not top-k); weights are the
    ``score_func`` activation of the logits gathered at those indices, normalized across the selected
    experts and scaled. Returns ``(indices, scaled_weights)`` shaped ``[*leading, n_activated_experts]``.
    """
    experts = logits.shape[-1]
    leading = logits.shape[:-1]
    flat_logits = logits.reshape(-1, experts)
    flat_ids = input_ids.reshape(-1).long()

    indices = tid2eid[flat_ids][:, :n_activated_experts].long()  # [tokens, n_activated]
    scores = score_activation(flat_logits, score_func)
    chosen = torch.gather(scores, dim=-1, index=indices)
    normalized = chosen / (chosen.sum(dim=-1, keepdim=True) + epsilon)
    scaled = normalized * route_scale
    return (
        indices.reshape(*leading, n_activated_experts),
        scaled.reshape(*leading, n_activated_experts),
    )


def calculate_average_recall(
    predicted_experts: torch.Tensor,
    reference_experts: torch.Tensor,
    predicted_weights: Optional[torch.Tensor] = None,
    reference_weights: Optional[torch.Tensor] = None,
    weight_rtol: float = 0.0,
) -> float:
    """Calculate average recall of predicted expert selections vs reference.

    Plain mode (no weights): fraction of reference experts also selected by the device.

    Tie-aware mode (per-expert gate weights supplied, aligned position-for-position with the
    index tensors, and weight_rtol > 0): a reference expert the device did NOT select is still
    counted as recovered if the device selected some other expert whose gate weight is within
    weight_rtol (relative) of the missed expert's weight. DeepSeek uses grouped top-k gating, so
    at a crowded selection boundary the device fp32 gate and the torch reference can pick different
    experts that carry near-equal weight — a numerically-ambiguous swap that leaves the routed
    output essentially unchanged (the block-level PCC stays ~0.999, and pcc_scores remains the
    correctness backstop for the selected-weight distribution).
    """
    recall = 0.0
    n = predicted_experts.shape[0]
    tie_aware = predicted_weights is not None and reference_weights is not None and weight_rtol > 0.0
    for i in range(n):
        pred_idx = [e.item() for e in predicted_experts[i]]
        ref_idx = [e.item() for e in reference_experts[i]]
        pred_set = set(pred_idx)
        ref_set = set(ref_idx)
        if not ref_set:
            continue
        hits = len(pred_set & ref_set)
        if tie_aware:
            # Gate weight of each selected expert, aligned by position with its index tensor.
            ref_w = {ref_idx[j]: reference_weights[i][j].item() for j in range(len(ref_idx))}
            pred_w = {pred_idx[j]: predicted_weights[i][j].item() for j in range(len(pred_idx))}
            extra_w = [pred_w[e] for e in (pred_set - ref_set)]  # device-only experts' weights
            for missed in ref_set - pred_set:
                wm = ref_w[missed]
                for k in range(len(extra_w)):  # credit one unconsumed device-only expert of tied weight
                    we = extra_w[k]
                    if we is not None and abs(we - wm) <= weight_rtol * max(abs(wm), abs(we), 1e-6):
                        hits += 1
                        extra_w[k] = None
                        break
        recall += hits / len(ref_set)
    return recall / n


def distinct_logits(shape, lo: float = -6.0, hi: float = 6.0, dtype=torch.float32) -> torch.Tensor:
    """Gate logits whose per-row values are distinct (so the monotonic score_func activation, and
    hence top-k / gather ordering, is unambiguous). Each row is a random subset of a shared set of
    distinct candidates in ``[lo, hi]``. Shared by the moe_grouped_topk and moe_hash_gate tests.
    """
    row_size = shape[-1]
    num_rows = int(torch.tensor(shape[:-1]).prod().item())
    candidates = torch.linspace(lo, hi, row_size * 4, dtype=dtype).unique()
    if candidates.numel() < row_size:
        raise ValueError(f"Cannot generate {row_size} distinct logits in [{lo}, {hi}].")
    rows = [candidates[torch.randperm(candidates.numel())[:row_size]] for _ in range(num_rows)]
    return torch.stack(rows).to(dtype).reshape(shape)


def build_padding_config(device, num_real: int):
    """ROW_MAJOR uint32 ``[[num_real, 0]]`` right-padding config shared by the moe gate ops.

    pad_side 0 == right padding: the leading ``num_real`` token rows are real, the rest are padding
    (routed to the sentinel expert id == total_experts).
    """
    import ttnn

    return ttnn.from_torch(
        torch.tensor([[num_real, 0]], dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )


def assert_gate_output(
    tt_indices: torch.Tensor,
    tt_weights: torch.Tensor,
    ref_indices: torch.Tensor,
    ref_weights: torch.Tensor,
    n_activated_experts: int,
    total_experts: int,
    num_real: int,
    apply_padding: bool,
    *,
    exact_recall: bool = False,
    recall_threshold: float = 0.9,
    pcc_threshold: float = 0.97,
) -> Tuple[float, float]:
    """Shared expert-selection (recall) + weight (PCC) check for the moe gate ops.

    Flattens both sides to ``[tokens, n_activated_experts]``, verifies that any padded rows route to
    the out-of-range sentinel expert (== ``total_experts``), then compares only the real rows.
    ``exact_recall`` requires recall == 1.0 (deterministic hash routing); otherwise recall must be
    >= ``recall_threshold``. Returns ``(recall, weights_pcc)``.
    """
    # Cast indices to long once so the uint16 device output can be compared without promotion errors.
    tt_indices = tt_indices.reshape(-1, n_activated_experts).long()
    ref_indices = ref_indices.reshape(-1, n_activated_experts).long()
    tt_weights = tt_weights.reshape(-1, n_activated_experts)
    ref_weights = ref_weights.reshape(-1, n_activated_experts)

    if apply_padding:
        pad_rows = tt_indices[num_real:]
        assert torch.all(
            pad_rows == total_experts
        ), f"Padded rows not all sentinel ({total_experts}); got {torch.unique(pad_rows).tolist()}"
        tt_indices, ref_indices = tt_indices[:num_real], ref_indices[:num_real]
        tt_weights, ref_weights = tt_weights[:num_real], ref_weights[:num_real]

    recall = calculate_average_recall(tt_indices, ref_indices)
    if exact_recall:
        assert recall == 1.0, f"Expected exact routing (recall 1.0), got {recall:.4f}"
    else:
        assert recall >= recall_threshold, f"Recall {recall:.4f} < {recall_threshold}"

    weights_passed, weights_pcc = comp_pcc(tt_weights.float(), ref_weights.float(), pcc=pcc_threshold)
    logger.info(
        f"[{'PASS' if weights_passed else 'FAIL'}] recall={recall:.4f} weights_pcc={weights_pcc:.4f} "
        f"(recall_thr={'1.0' if exact_recall else recall_threshold}, pcc_thr={pcc_threshold})"
    )
    assert weights_passed, f"Weights PCC {weights_pcc:.4f} < {pcc_threshold}"
    return recall, weights_pcc


def trace_token_source(
    dispatch_group_idx: int,
    chip_id: int,
    token_id: int,
    topk_idx: int,
    indices: torch.Tensor,
    expert_dispatch_table: torch.Tensor,
    expert_token_counts: torch.Tensor,
    num_dispatch_groups: int,
    num_routed_experts: int,
    experts_per_chip: int,
) -> dict:
    """
    Trace the source of a token in the combine output.

    Given a failed token position, returns info about which expert processed it,
    which chip hosts that expert, and the send order of this token.

    Args:
        dispatch_group_idx: The dispatch group (EP rank) index
        chip_id: The destination chip ID
        token_id: The token index
        topk_idx: The top-k index
        indices: Expert indices tensor [dispatch_group_size, seq_len_per_chip, num_experts_per_tok]
        expert_dispatch_table: Expert to chip mapping [num_dispatch_groups, num_routed_experts]
        expert_token_counts: Token counts per expert [num_dispatch_groups, dispatch_group_size, experts_per_chip]
        num_dispatch_groups: Number of EP ranks
        num_routed_experts: Total number of routed experts
        experts_per_chip: Number of experts per chip

    Returns:
        Dict with expert_id, expert_chip, local_expert, send_order, total_tokens
    """
    expert_id = indices[chip_id, token_id, topk_idx].item()
    expert_chip = expert_dispatch_table[dispatch_group_idx, expert_id].item()

    # Calculate local_expert using same formula as create_expert_dispatch_table
    experts_per_group = num_routed_experts // num_dispatch_groups
    local_expert_in_group = expert_id % experts_per_group
    local_expert = local_expert_in_group % experts_per_chip

    total_tokens = expert_token_counts[dispatch_group_idx, 0, expert_id].item()

    # Find send order by counting tokens sent before this one
    # This mirrors the kernel's iteration order: for each token routed to this expert
    send_order = 0
    for c in range(indices.shape[0]):
        for t in range(indices.shape[1]):
            for k in range(indices.shape[2]):
                if indices[c, t, k].item() == expert_id:
                    if (c, t, k) == (chip_id, token_id, topk_idx):
                        return {
                            "expert_id": expert_id,
                            "expert_chip": expert_chip,
                            "local_expert": local_expert,
                            "send_order": send_order,
                            "total_tokens": total_tokens,
                        }
                    send_order += 1
    return None


@dataclass
class ValidationResult:
    """Result of tensor comparison."""

    passed: bool
    matches: int
    total: int
    mismatches: List[Tuple] = field(default_factory=list)
    name: str = "validation"
    # Set of (dispatch_group, chip) pairs that were actually validated
    validated_cells: set = field(default_factory=set)

    def log_summary(self):
        """Log match statistics."""
        pct = 100.0 * self.matches / self.total if self.total > 0 else 0.0
        status = "✅" if self.passed else "❌"
        logger.debug(f"{status} {self.name}: {self.matches}/{self.total} ({pct:.2f}%)")

    def log_mismatches(self, limit: int = 10):
        """Log first N mismatches."""
        if not self.mismatches:
            return

        logger.warning(f"Found {len(self.mismatches)} mismatches in {self.name}. Showing first {limit}:")
        for i, mismatch in enumerate(self.mismatches[:limit]):
            if len(mismatch) == 5:
                # Combine-style: (dispatch_group_idx, chip_id, token_id, topk_idx, max_diff)
                dispatch_group_idx, chip_id, token_id, topk_idx, max_diff = mismatch
                logger.error(
                    f"  [{i}] dispatch_group_idx={dispatch_group_idx}, chip={chip_id}, token={token_id}, topk={topk_idx}: "
                    f"max_diff={max_diff:.6f}"
                )
            elif len(mismatch) == 4:
                # Dispatch-style: (dispatch_group_idx, chip_id, expert_id, error_detail)
                dispatch_group_idx, chip_id, expert_id, error_detail = mismatch
                logger.error(
                    f"  [{i}] dispatch_group_idx={dispatch_group_idx}, chip={chip_id}, expert={expert_id}: {error_detail}"
                )
            else:
                # Generic fallback
                logger.error(f"  [{i}] {mismatch}")

    @classmethod
    def merge(cls, results: List["ValidationResult"], name: str = "merged") -> "ValidationResult":
        """Combine multiple ValidationResults into one."""
        passed = all(r.passed for r in results)
        matches = sum(r.matches for r in results)
        total = sum(r.total for r in results)
        mismatches = []
        for r in results:
            mismatches.extend(r.mismatches)
        return cls(passed=passed, matches=matches, total=total, mismatches=mismatches, name=name)

    def assert_passed(self, msg: Optional[str] = None):
        """Assert validation passed, with detailed error on failure."""
        if msg is None:
            msg = f"{self.name} failed"
        assert self.passed, f"{msg}! {self.matches}/{self.total} slots matched. Check logs for details."


def assert_output_shape(
    output: torch.Tensor,
    num_dispatch_groups: int,
    dispatch_group_size: int,
    context: str = "output",
):
    """Assert that output tensor has expected shape for mesh configuration."""
    assert (
        output.shape[0] == num_dispatch_groups
    ), f"Mismatch in {context} replicated dimension: expected {num_dispatch_groups}, got {output.shape[0]}"
    assert (
        output.shape[1] == dispatch_group_size
    ), f"Mismatch in {context} sharded dimension: expected {dispatch_group_size}, got {output.shape[1]}"


def validate_combine_output(
    torch_output: torch.Tensor,
    ttnn_output: torch.Tensor,
    indices: torch.Tensor,
    num_dispatch_groups: int,
    num_routed_experts: int,
    use_pcc: bool = False,
    pcc_threshold: float = 0.93,
    atol: float = 1e-4,
    rtol: float = 1e-4,
    verbose: bool = False,
    expert_dispatch_table: torch.Tensor = None,
    expert_token_counts: torch.Tensor = None,
    experts_per_chip: int = None,
) -> ValidationResult:
    """
    Validate combine output against torch reference (EP-rank aware).

    Args:
        torch_output: Reference output, shape [dispatch_group_size, seq_len_per_chip, num_experts_per_tok, emb_dim]
        ttnn_output: TTNN output, shape [num_dispatch_groups, dispatch_group_size, seq_len_per_chip, num_experts_per_tok, emb_dim]
        indices: Expert indices, shape [dispatch_group_size, seq_len_per_chip, num_experts_per_tok]
        num_dispatch_groups: Number of EP ranks
        num_routed_experts: Total number of routed experts
        use_pcc: If True, use PCC for comparison; if False, use allclose
        pcc_threshold: PCC threshold for pass/fail (default 0.93, only used if use_pcc=True)
        atol: Absolute tolerance for allclose (only used if use_pcc=False)
        rtol: Relative tolerance for allclose (only used if use_pcc=False)

    Returns:
        ValidationResult with match statistics and mismatches
    """
    dispatch_group_size = torch_output.shape[0]
    seq_len_per_chip = torch_output.shape[1]
    num_experts_per_tok = torch_output.shape[2]
    experts_per_dispatch_group = num_routed_experts // num_dispatch_groups

    matches = 0
    total_slots = 0
    mismatches = []
    validated_cells = set()

    for chip_id in range(dispatch_group_size):
        for token_id in range(seq_len_per_chip):
            for topk_idx in range(num_experts_per_tok):
                total_slots += 1

                # Determine which EP rank processed this (chip, token, topk)
                expert_id = indices[chip_id, token_id, topk_idx].item()
                dispatch_group_idx = expert_id // experts_per_dispatch_group
                validated_cells.add((dispatch_group_idx, chip_id))

                torch_data = torch_output[chip_id, token_id, topk_idx]
                ttnn_data = ttnn_output[dispatch_group_idx, chip_id, token_id, topk_idx]

                if use_pcc:
                    # Use PCC for comparison
                    _, pcc = comp_pcc(torch_data.float(), ttnn_data.float())
                    is_match = pcc >= pcc_threshold
                    metric_value = pcc
                    metric_detail = f"PCC={pcc:.6f} (threshold={pcc_threshold})"
                else:
                    # Use allclose for comparison
                    is_match = torch.allclose(torch_data, ttnn_data, atol=atol, rtol=rtol)
                    max_diff = torch.max(torch.abs(torch_data.float() - ttnn_data.float())).item()
                    metric_value = max_diff
                    metric_detail = f"max_diff={max_diff:.6f}"

                if is_match:
                    matches += 1
                else:
                    mismatches.append((dispatch_group_idx, chip_id, token_id, topk_idx, metric_value))

                    if verbose and len(mismatches) <= 10:
                        logger.error(
                            f"❌ Combine mismatch [{len(mismatches)}]: "
                            f"{expert_id=} {dispatch_group_idx=} {chip_id=} {token_id=} {topk_idx=}, "
                            f"{metric_detail}"
                        )
                        logger.error(f"   torch_data[:5] = {torch_data[:5].tolist()}")
                        logger.error(f"   ttnn_data[:5]  = {ttnn_data[:5].tolist()}")
                        logger.error(f"   torch_data[-5:] = {torch_data[-5:].tolist()}")
                        logger.error(f"   ttnn_data[-5:]  = {ttnn_data[-5:].tolist()}")
                        # Check if data is all zeros (indicates data didn't arrive)
                        if torch.all(ttnn_data == 0):
                            logger.error(f"   ⚠️  TTNN data is ALL ZEROS - data may not have arrived!")
                        elif torch.all(torch_data == 0):
                            logger.error(f"   ⚠️  Torch data is ALL ZEROS")
                        else:
                            # Show where first difference occurs
                            diff = torch.abs(torch_data.float() - ttnn_data.float())
                            first_diff_idx = torch.argmax(diff).item()
                            logger.error(
                                f"   Max diff at index {first_diff_idx}: "
                                f"torch={torch_data[first_diff_idx].item():.6f}, "
                                f"ttnn={ttnn_data[first_diff_idx].item():.6f}"
                            )
                            # Segment analysis: ✅ = good, ⚠️ = bad (ttnn=0 but torch not near zero)
                            is_ttnn_zero = ttnn_data == 0
                            is_torch_small = torch.abs(torch_data.float()) < 0.01
                            is_bad = is_ttnn_zero & ~is_torch_small

                            # Build segments
                            segments = []
                            if len(is_bad) > 0:
                                current_bad = is_bad[0].item()
                                current_len = 1
                                for i in range(1, len(is_bad)):
                                    if is_bad[i].item() == current_bad:
                                        current_len += 1
                                    else:
                                        segments.append((current_len, current_bad))
                                        current_bad = is_bad[i].item()
                                        current_len = 1
                                segments.append((current_len, current_bad))

                            # Merge small good segments (<=2) surrounded by bad into bad
                            merged = []
                            for length, bad in segments:
                                if not bad and length <= 2 and merged and merged[-1][1]:
                                    # Small good after bad - merge into previous bad
                                    merged[-1] = (merged[-1][0] + length, True)
                                elif merged and merged[-1][1] == bad:
                                    # Same type - merge
                                    merged[-1] = (merged[-1][0] + length, bad)
                                else:
                                    merged.append((length, bad))
                            segments = merged

                            segment_strs = [f"[{length},{'⚠️' if bad else '✅'}]" for length, bad in segments]
                            logger.error(f"   Segments: {' '.join(segment_strs)}")

                        # Trace token source if data available
                        if expert_dispatch_table is not None and expert_token_counts is not None:
                            trace = trace_token_source(
                                dispatch_group_idx,
                                chip_id,
                                token_id,
                                topk_idx,
                                indices,
                                expert_dispatch_table,
                                expert_token_counts,
                                num_dispatch_groups,
                                num_routed_experts,
                                experts_per_chip,
                            )
                            if trace:
                                logger.error(
                                    f"   Source: expert={trace['expert_id']} on chip={trace['expert_chip']} "
                                    f"(local_expert={trace['local_expert']}), "
                                    f"send_order={trace['send_order']}/{trace['total_tokens']}"
                                )

    passed = len(mismatches) == 0
    return ValidationResult(
        passed=passed,
        matches=matches,
        total=total_slots,
        mismatches=mismatches,
        name="combine output",
        validated_cells=validated_cells,
    )


def validate_roundtrip_output(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    indices: torch.Tensor,
    num_dispatch_groups: int,
    num_routed_experts: int,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> ValidationResult:
    """
    Validate dispatch→combine round-trip (EP-rank aware).

    Args:
        input_tensor: Original input, shape [dispatch_group_size, seq_len_per_chip, emb_dim]
        output_tensor: Round-trip output, shape [num_dispatch_groups, dispatch_group_size, seq_len_per_chip, num_experts_per_tok, emb_dim]
        indices: Expert indices, shape [dispatch_group_size, seq_len_per_chip, num_experts_per_tok]
        num_dispatch_groups: Number of EP ranks
        num_routed_experts: Total number of routed experts
        atol: Absolute tolerance for comparison
        rtol: Relative tolerance for comparison

    Returns:
        ValidationResult with match statistics and mismatches
    """
    dispatch_group_size = input_tensor.shape[0]
    seq_len_per_chip = input_tensor.shape[1]
    num_experts_per_tok = indices.shape[2]
    experts_per_dispatch_group = num_routed_experts // num_dispatch_groups

    matches = 0
    total_slots = 0
    mismatches = []
    validated_cells = set()

    for chip_id in range(dispatch_group_size):
        for token_id in range(seq_len_per_chip):
            for topk_idx in range(num_experts_per_tok):
                total_slots += 1

                # Determine which EP rank processed this (chip, token, topk)
                expert_id = indices[chip_id, token_id, topk_idx].item()
                dispatch_group_idx = expert_id // experts_per_dispatch_group
                validated_cells.add((dispatch_group_idx, chip_id))

                # Input token
                x_data = input_tensor[chip_id, token_id]
                # Output from the EP rank that processed this token
                y_data = output_tensor[dispatch_group_idx, chip_id, token_id, topk_idx]

                if torch.allclose(x_data, y_data, atol=atol, rtol=rtol):
                    matches += 1
                else:
                    max_diff = torch.max(torch.abs(x_data.float() - y_data.float())).item()
                    mismatches.append((dispatch_group_idx, chip_id, token_id, topk_idx, max_diff))

    passed = len(mismatches) == 0
    return ValidationResult(
        passed=passed,
        matches=matches,
        total=total_slots,
        mismatches=mismatches,
        name="roundtrip",
        validated_cells=validated_cells,
    )


def log_combine_mismatch_details(
    mismatches: List[Tuple],
    torch_output: torch.Tensor,
    ttnn_output: torch.Tensor,
    limit: int = 10,
    use_pcc: bool = False,
):
    """Log detailed mismatch information for combine validation."""
    for i, (dispatch_group_idx, chip_id, token_id, topk_idx, metric_value) in enumerate(mismatches[:limit]):
        torch_sample = torch_output[chip_id, token_id, topk_idx, :5]
        ttnn_sample = ttnn_output[dispatch_group_idx, chip_id, token_id, topk_idx, :5]
        metric_str = f"PCC={metric_value:.6f}" if use_pcc else f"max_diff={metric_value:.6f}"
        logger.error(
            f"  [{i}] Mismatch at dispatch_group_idx={dispatch_group_idx}, chip={chip_id}, token={token_id}, topk={topk_idx}: "
            f"{metric_str}"
        )
        logger.error(f"      torch[:5]={torch_sample}")
        logger.error(f"      ttnn[:5]={ttnn_sample}")


def log_per_chip_statistics(
    mismatches: List[Tuple],
    dispatch_group_size: int,
    seq_len_per_chip: int,
    num_experts_per_tok: int,
):
    """Log per-chip match statistics."""
    logger.debug("\nPer-chip statistics:")
    for chip_id in range(dispatch_group_size):
        chip_mismatches = [m for m in mismatches if m[1] == chip_id]
        chip_total = seq_len_per_chip * num_experts_per_tok
        chip_matches = chip_total - len(chip_mismatches)
        logger.debug(f"  Chip {chip_id}: {chip_matches}/{chip_total} matches ({100.0*chip_matches/chip_total:.2f}%)")


# Type for dispatch validation comparators
# Returns (match: bool, error_detail: Optional[str])
DispatchComparator = Callable[[torch.Tensor, torch.Tensor, int, int, int], Tuple[bool, Optional[str]]]

# Type for composed tensor comparators
# (actual_chip, expected_chip, group, chip) -> (match, error_detail)
ComposedComparator = Callable[[torch.Tensor, torch.Tensor, int, int], Tuple[bool, Optional[str]]]


def compare_exact(actual: torch.Tensor, expected: torch.Tensor, _g: int, _c: int) -> Tuple[bool, Optional[str]]:
    """Exact element-wise comparison."""
    if torch.equal(actual, expected):
        return True, None
    diff = (actual != expected).sum().item()
    return False, f"{diff}/{actual.numel()} elements differ"


def compare_pcc(threshold: float = 0.99) -> ComposedComparator:
    """Return a PCC comparator with the given threshold."""

    def _compare(actual: torch.Tensor, expected: torch.Tensor, _g: int, _c: int) -> Tuple[bool, Optional[str]]:
        _, pcc = comp_pcc(actual.float(), expected.float())
        return (True, None) if pcc >= threshold else (False, f"PCC={pcc:.4f} < {threshold}")

    return _compare


def compare_recall(
    threshold: float = 0.999,
    predicted_weights: Optional[torch.Tensor] = None,
    reference_weights: Optional[torch.Tensor] = None,
    weight_rtol: float = 0.0,
) -> ComposedComparator:
    """Return a recall comparator with the given threshold.

    When gate-weight tensors are supplied (same [num_groups, group_size, ...] layout as the index
    tensors, and NOT reordered relative to them) and weight_rtol > 0, recall is tie-aware: a
    boundary swap between experts of near-equal gate weight is credited (see
    calculate_average_recall). The per-cell weight slices are looked up by the (_g, _c) indices.
    """

    def _compare(
        actual: torch.Tensor, expected: torch.Tensor, _g: int, _c: int, verbose_histogram: bool = False
    ) -> Tuple[bool, Optional[str]]:
        pred_w = predicted_weights[_g, _c] if predicted_weights is not None else None
        ref_w = reference_weights[_g, _c] if reference_weights is not None else None
        r = calculate_average_recall(actual, expected, pred_w, ref_w, weight_rtol)
        if r >= threshold:
            return (True, f"recall={r:.4f} >= {threshold}")
        else:
            from collections import Counter

            total_elements = len(actual)
            mismatches = []

            # not very efficient
            for i, (a, e) in enumerate(
                zip(torch.sort(actual, dim=-1).values.long(), torch.sort(expected, dim=-1).values.long())
            ):
                match, error_detail = compare_exact(a, e, 0, 0)
                if not match:
                    mismatches.append(error_detail)
            detail = ""
            if verbose_histogram:
                if mismatches:
                    num_errors = len(mismatches)
                    total_percent = (num_errors / total_elements) * 100

                    # Header showing (1329/4096 total) [32.4%]
                    detail = (
                        f"\n*** Mismatch Histogram ({num_errors}/{total_elements} total) [{total_percent:.1f}%] ***"
                    )

                    print(detail)

                    counts = Counter(mismatches)
                    num_matches = total_elements - num_errors
                    match_label = "0 errors; MATCH"
                    counts[match_label] = num_matches
                    # Sort by frequency (most common first)
                    sorted_counts = counts.most_common()

                    max_label_len = max(len(str(label)) for label in counts.keys())
                    max_bar_width = 30
                    max_count = max(counts.values())
                    scale = max_count / max_bar_width if max_count > max_bar_width else 1

                    for label, count in sorted_counts:
                        bar = "█" * int(count / scale)
                        item_percent = (count / total_elements) * 100

                        # Format: Label | Bar (padded) | count/total | [percentage]
                        print(
                            f"{str(label).ljust(max_label_len)} | "
                            f"{bar.ljust(max_bar_width)} "
                            f"{str(count).rjust(len(str(total_elements)))}/{total_elements} "
                            f"[{item_percent:5.1f}%]"
                        )

                    print("-" * (max_label_len + max_bar_width + 25) + "\n")
                else:
                    detail = f"*** All {total_elements}/{total_elements} matched! [0.0% errors] ***"

        return (False, f"recall={r:.4f} < {threshold};")

    return _compare


def validate_composed(
    actual: torch.Tensor,
    expected: torch.Tensor,
    num_groups: int,
    group_size: int,
    compare_fn: ComposedComparator,
    name: str = "composed",
    broadcast_groups: int = 0,
) -> ValidationResult:
    """
    Validate a composed tensor per (group, chip) cell.

    Iterates over actual[group, chip] vs expected[group, chip], calling compare_fn
    for each cell. Populates validated_cells for the grid visualizer.

    Args:
        actual: Composed TTNN output, shape [num_groups, group_size, ...]
        expected: Reference tensor, shape [num_groups, group_size, ...]
        num_groups: Number of dispatch groups to iterate over
        group_size: Number of chips per group (mesh rows)
        compare_fn: (actual_chip, expected_chip, group, chip) -> (match, error_detail)
        name: Name for logging/result
        broadcast_groups: If > 0, broadcast validated_cells and mismatches across this many
            groups in the visualizer. Use for SP-replicated tensors where num_groups=1 but
            the grid has multiple dispatch group columns.
    """
    display_groups = broadcast_groups if broadcast_groups > 0 else num_groups
    matches, mismatches = 0, []
    validated_cells = set()
    for g in range(num_groups):
        for c in range(group_size):
            # Mark all display columns as validated for this chip
            for dg in range(display_groups) if broadcast_groups > 0 else [g]:
                validated_cells.add((dg, c))
            match, error_detail = compare_fn(actual[g, c], expected[g, c], g, c)
            if match:
                matches += 1
            else:
                if broadcast_groups > 0:
                    for dg in range(display_groups):
                        mismatches.append((dg, c, error_detail or "mismatch"))
                else:
                    mismatches.append((g, c, error_detail or "mismatch"))
    return ValidationResult(
        passed=len(mismatches) == 0,
        matches=matches,
        total=num_groups * group_size,
        mismatches=mismatches,
        name=name,
        validated_cells=validated_cells,
    )


def validate_replication(
    tensor: torch.Tensor,
    name: str = "replication",
) -> ValidationResult:
    """Validate tensor is replicated across dim 1 (chips) within each group (dim 0)."""
    num_groups = tensor.shape[0]
    group_size = tensor.shape[1]
    matches, total, mismatches = 0, 0, []
    for g in range(num_groups):
        ref = tensor[g, 0]
        for j in range(1, group_size):
            total += 1
            if torch.allclose(tensor[g, j].int(), ref.int(), atol=0, rtol=0):
                matches += 1
            else:
                mismatches.append((g, j, f"group {g} row {j} differs from row 0"))
    return ValidationResult(
        passed=len(mismatches) == 0,
        matches=matches,
        total=total,
        mismatches=mismatches,
        name=name,
    )


def _get_valid_slots(
    expert_dispatch_table: torch.Tensor,
    num_dispatch_groups: int,
    num_routed_experts: int,
    dispatch_group_size: int,
    experts_per_chip: int,
) -> set:
    """Build set of (dispatch_group, chip, local_expert) slots that should be validated.

    Uses the dispatch table directly as the source of truth for which slots exist.
    """
    valid_slots = set()
    experts_per_group = num_routed_experts // num_dispatch_groups

    for dispatch_group in range(num_dispatch_groups):
        for global_expert_id in range(num_routed_experts):
            chip_id = expert_dispatch_table[dispatch_group, global_expert_id].item()
            if chip_id != -1:
                # Map global expert to local expert within chip
                local_expert_in_group = global_expert_id % experts_per_group
                local_expert_id = local_expert_in_group % experts_per_chip
                valid_slots.add((dispatch_group, chip_id, local_expert_id))

    return valid_slots


def validate_dispatch_data(
    torch_data: torch.Tensor,
    ttnn_data: torch.Tensor,
    expert_region_offsets: torch.Tensor,
    expert_token_counts: torch.Tensor,
    expert_dispatch_table: torch.Tensor,
    num_dispatch_groups: int,
    dispatch_group_size: int,
    experts_per_chip: int,
    compare_fn: DispatchComparator,
    name: str = "data",
    verbose: bool = True,
) -> ValidationResult:
    """
    Generic dispatch data validation.

    Iterates over (dispatch_group_idx, chip, expert) slots, compares using compare_fn.
    Skips experts not present in EP rank (dispatch_table == -1).

    Args:
        torch_data: Reference data, shape [num_dispatch_groups, dispatch_group_size, total_buffer_rows, ...]
        ttnn_data: TTNN data, same shape as torch_data
        expert_region_offsets: Expert region offsets (shared across source devices in a
            dispatch group), shape [num_dispatch_groups, dispatch_group_size, num_routed_experts].
            Gives the expert region start position for each expert directly; no need to
            read expert_offsets at chip 0.
        expert_token_counts: Token counts per expert, shape [num_dispatch_groups, dispatch_group_size, num_routed_experts]
        expert_dispatch_table: Expert to chip mapping, shape [num_dispatch_groups, num_routed_experts]
        num_dispatch_groups: Number of EP ranks
        dispatch_group_size: Number of chips in dispatch group
        experts_per_chip: Number of experts per chip
        compare_fn: Comparison function (torch_slot, ttnn_slot, dispatch_group_idx, chip_id, expert_id) -> (match, error_detail)
        name: Name for logging
        verbose: Whether to log detailed mismatch info

    Returns:
        ValidationResult with match statistics
    """
    num_routed_experts = expert_dispatch_table.shape[1]
    valid_slots = _get_valid_slots(
        expert_dispatch_table, num_dispatch_groups, num_routed_experts, dispatch_group_size, experts_per_chip
    )

    matches = 0
    total_slots = 0
    mismatches = []
    validated_cells = set()

    for r in range(num_dispatch_groups):
        for dst_chip_id in range(dispatch_group_size):
            for expert_id in range(experts_per_chip):
                if (r, dst_chip_id, expert_id) not in valid_slots:
                    # Expert not present in this dispatch group, skip comparison
                    continue

                total_slots += 1
                validated_cells.add((r, dst_chip_id))
                global_expert_idx = ExpertMapping.get_global_expert_idx(
                    group=r,
                    chip=dst_chip_id,
                    local_expert=expert_id,
                    experts_per_chip=experts_per_chip,
                    dispatch_group_size=dispatch_group_size,
                    num_dispatch_groups=num_dispatch_groups,
                    is_col_major=True,
                )
                count = expert_token_counts[r, 0, global_expert_idx].item()

                # expert_region_offsets directly gives the expert region start position
                start = int(expert_region_offsets[r, dst_chip_id, global_expert_idx].item())
                torch_slot = torch_data[r, dst_chip_id, start : start + count]
                ttnn_slot = ttnn_data[r, dst_chip_id, start : start + count]

                match, error_detail = compare_fn(torch_slot, ttnn_slot, r, dst_chip_id, expert_id)

                if match:
                    matches += 1
                    logger.debug(f"✅ {r} {name} {dst_chip_id=} {expert_id=} {count=}")
                else:
                    logger.error(f"❌ {r} {name} {dst_chip_id=} {expert_id=} {count=}")
                    mismatches.append((r, dst_chip_id, expert_id, error_detail or "mismatch"))

                    if verbose:
                        # Log per-slot details
                        for slot in range(count):
                            slot_match, slot_error = compare_fn(
                                torch_slot[slot : slot + 1],
                                ttnn_slot[slot : slot + 1],
                                r,
                                dst_chip_id,
                                expert_id,
                            )
                            if not slot_match:
                                logger.error(f"    Slot {slot}: {slot_error}")

    passed = len(mismatches) == 0
    return ValidationResult(
        passed=passed,
        matches=matches,
        total=total_slots,
        mismatches=mismatches,
        name=name,
        validated_cells=validated_cells,
    )


def validate_dispatch_buffer(
    torch_dispatched: torch.Tensor,
    ttnn_dispatched: torch.Tensor,
    expert_region_offsets: torch.Tensor,
    expert_token_counts: torch.Tensor,
    expert_dispatch_table: torch.Tensor,
    num_dispatch_groups: int,
    dispatch_group_size: int,
    experts_per_chip: int,
    verbose: bool = True,
) -> ValidationResult:
    """
    Validate dispatch buffer against torch reference.

    Args:
        torch_dispatched: Reference dispatched buffer
        ttnn_dispatched: TTNN dispatched buffer
        expert_region_offsets: Expert region offsets for slicing flat buffers
        expert_token_counts: Token counts per expert
        expert_dispatch_table: Expert to chip mapping
        num_dispatch_groups: Number of EP ranks
        dispatch_group_size: Number of chips in dispatch group
        experts_per_chip: Number of experts per chip
        verbose: Whether to log detailed mismatch info

    Returns:
        ValidationResult with match statistics
    """

    def compare_buffer(
        torch_slot: torch.Tensor,
        ttnn_slot: torch.Tensor,
        _r: int,
        _chip: int,
        _expert: int,
    ) -> Tuple[bool, Optional[str]]:
        match = torch.allclose(torch_slot, ttnn_slot, atol=1e-6)
        if not match:
            max_diff = torch.max(torch.abs(torch_slot.float() - ttnn_slot.float())).item()
            return False, f"max_diff={max_diff:.6f}"
        return True, None

    return validate_dispatch_data(
        torch_dispatched,
        ttnn_dispatched,
        expert_region_offsets,
        expert_token_counts,
        expert_dispatch_table,
        num_dispatch_groups,
        dispatch_group_size,
        experts_per_chip,
        compare_fn=compare_buffer,
        name="buffer",
        verbose=verbose,
    )


def validate_dispatch_buffer_pcc(
    torch_dispatched: torch.Tensor,
    ttnn_dispatched: torch.Tensor,
    expert_region_offsets: torch.Tensor,
    expert_token_counts: torch.Tensor,
    expert_dispatch_table: torch.Tensor,
    num_dispatch_groups: int,
    dispatch_group_size: int,
    experts_per_chip: int,
    pcc_threshold: float = 0.99,
    verbose: bool = True,
) -> ValidationResult:
    """
    Validate dispatch buffer against torch reference using PCC (for data with numerical differences).

    Use this instead of validate_dispatch_buffer when comparing outputs that have gone through
    quantized operations (e.g., expert outputs with bf4/bf8 weights).

    Args:
        torch_dispatched: Reference dispatched buffer
        ttnn_dispatched: TTNN dispatched buffer
        expert_region_offsets: Expert region offsets for slicing flat buffers
        expert_token_counts: Token counts per expert
        expert_dispatch_table: Expert to chip mapping
        num_dispatch_groups: Number of EP ranks
        dispatch_group_size: Number of chips in dispatch group
        experts_per_chip: Number of experts per chip
        pcc_threshold: PCC threshold for pass/fail (default 0.99)
        verbose: Whether to log detailed mismatch info

    Returns:
        ValidationResult with match statistics
    """

    def compare_pcc(
        torch_slot: torch.Tensor,
        ttnn_slot: torch.Tensor,
        _r: int,
        _chip: int,
        _expert: int,
    ) -> Tuple[bool, Optional[str]]:
        if torch_slot.numel() == 0:
            return True, None
        _, pcc = comp_pcc(torch_slot.float(), ttnn_slot.float())
        if pcc >= pcc_threshold:
            return True, None
        return False, f"PCC={pcc:.4f} < {pcc_threshold}"

    return validate_dispatch_data(
        torch_dispatched,
        ttnn_dispatched,
        expert_region_offsets,
        expert_token_counts,
        expert_dispatch_table,
        num_dispatch_groups,
        dispatch_group_size,
        experts_per_chip,
        compare_fn=compare_pcc,
        name="buffer_pcc",
        verbose=verbose,
    )


def validate_dispatch_metadata(
    torch_metadata: torch.Tensor,
    ttnn_metadata: torch.Tensor,
    expert_region_offsets: torch.Tensor,
    expert_token_counts: torch.Tensor,
    expert_dispatch_table: torch.Tensor,
    num_dispatch_groups: int,
    dispatch_group_size: int,
    experts_per_chip: int,
    verbose: bool = True,
) -> ValidationResult:
    """
    Validate dispatch metadata against torch reference.

    Metadata is 3 fields per token; all are validated:
    - Linearized mesh coord conversion (field 0)
    - Direct comparison of fields 1-2 (global token idx, top-k slot)

    Args:
        torch_metadata: Reference metadata
        ttnn_metadata: TTNN metadata
        expert_region_offsets: Expert region offsets for slicing flat buffers
        expert_token_counts: Token counts per expert
        expert_dispatch_table: Expert to chip mapping
        num_dispatch_groups: Number of EP ranks
        dispatch_group_size: Number of chips in dispatch group
        experts_per_chip: Number of experts per chip
        verbose: Whether to log detailed mismatch info

    Returns:
        ValidationResult with match statistics
    """
    num_routed_experts = expert_dispatch_table.shape[1]
    valid_slots = _get_valid_slots(
        expert_dispatch_table, num_dispatch_groups, num_routed_experts, dispatch_group_size, experts_per_chip
    )

    matches = 0
    total_slots = 0
    mismatches = []
    validated_cells = set()

    for r in range(num_dispatch_groups):
        for dst_chip_id in range(dispatch_group_size):
            for expert_id in range(experts_per_chip):
                if (r, dst_chip_id, expert_id) not in valid_slots:
                    # Expert not present in this dispatch group, skip comparison
                    continue

                total_slots += 1
                validated_cells.add((r, dst_chip_id))
                global_expert_idx = ExpertMapping.get_global_expert_idx(
                    group=r,
                    chip=dst_chip_id,
                    local_expert=expert_id,
                    experts_per_chip=experts_per_chip,
                    dispatch_group_size=dispatch_group_size,
                    num_dispatch_groups=num_dispatch_groups,
                    is_col_major=True,
                )
                count = expert_token_counts[r, 0, global_expert_idx].item()

                # expert_region_offsets directly gives the expert region start position
                start = int(expert_region_offsets[r, dst_chip_id, global_expert_idx].item())

                # Compare fields 1-2 (global token idx, top-k slot) directly
                out = ttnn_metadata[r, dst_chip_id, start : start + count, 1:3]
                ref = torch_metadata[r, dst_chip_id, start : start + count, 1:3]

                # Both Torch and TTNN embed linearized mesh coord in field 0
                out_linearized_mesh_coord = ttnn_metadata[r, dst_chip_id, start : start + count, 0]
                ref_linearized_mesh_coord = torch_metadata[r, dst_chip_id, start : start + count, 0]

                metadata_match = torch.allclose(out, ref, atol=1e-6)
                coord_match = torch.allclose(
                    out_linearized_mesh_coord.float(), ref_linearized_mesh_coord.float(), atol=1e-6
                )

                if metadata_match and coord_match:
                    matches += 1
                    logger.debug(f"✅ {r} Metadata {dst_chip_id=} {expert_id=} {count=}")
                else:
                    error_detail = f"metadata={metadata_match}, coord={coord_match}"
                    logger.error(f"❌ {r} Metadata {dst_chip_id=} {expert_id=} {count=} ({error_detail})")
                    mismatches.append((r, dst_chip_id, expert_id, error_detail))

                    if verbose:
                        for slot in range(count):
                            torch_data = torch_metadata[r, dst_chip_id, start + slot, :3]
                            kernel_data = ttnn_metadata[r, dst_chip_id, start + slot, :3]
                            slot_data_match = torch.allclose(torch_data, kernel_data, atol=1e-6)
                            if not slot_data_match:
                                logger.error(
                                    f"    Slot {slot}: Metadata mismatch at chip={dst_chip_id}, expert={expert_id}: "
                                    f"ref_coord={ref_linearized_mesh_coord[slot].item()}, "
                                    f"out_coord={out_linearized_mesh_coord[slot].item()}, "
                                    f"torch={torch_data.tolist()}, kernel={kernel_data.tolist()}"
                                )

    passed = len(mismatches) == 0
    return ValidationResult(
        passed=passed,
        matches=matches,
        total=total_slots,
        mismatches=mismatches,
        name="metadata",
        validated_cells=validated_cells,
    )
