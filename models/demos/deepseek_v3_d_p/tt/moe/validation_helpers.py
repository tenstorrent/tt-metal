# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Validation utilities for MoE dispatch/combine tests."""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import torch
from loguru import logger

from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping


def calculate_average_recall(predicted_experts: torch.Tensor, reference_experts: torch.Tensor) -> float:
    """Calculate average recall of predicted expert selections vs reference."""
    recall = 0
    for i in range(predicted_experts.shape[0]):
        pred_set = set(e.item() for e in predicted_experts[i])
        ref_set = set(e.item() for e in reference_experts[i])
        recall += len(pred_set & ref_set) / len(ref_set) if ref_set else 0
    return recall / predicted_experts.shape[0]


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


def compare_recall(threshold: float = 0.999) -> ComposedComparator:
    """Return a recall comparator with the given threshold."""

    def _compare(
        actual: torch.Tensor, expected: torch.Tensor, _g: int, _c: int, verbose_histogram: bool = False
    ) -> Tuple[bool, Optional[str]]:
        r = calculate_average_recall(actual, expected)
        if r >= threshold:
            return (True, None)
        else:
            from collections import Counter

            total_elements = len(actual)
            mismatches = []

            for i, (a, e) in enumerate(zip(actual.long(), expected.long())):
                match, error_detail = compare_exact(a, e, 0, 0)
                if not match:
                    mismatches.append(error_detail)

            if mismatches:
                num_errors = len(mismatches)
                total_percent = (num_errors / total_elements) * 100

                # Header showing (1329/4096 total) [32.4%]
                detail = f"\n*** Mismatch Histogram ({num_errors}/{total_elements} total) [{total_percent:.1f}%] ***"

                if verbose_histogram:
                    print(detail)

                    counts = Counter(mismatches)
                    # Sort by frequency (most common errors first)
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

        return (False, f"recall={r:.4f} < {threshold}; *** {detail} ***")

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
        torch_data: Reference data, shape [num_dispatch_groups, dispatch_group_size, experts_per_chip, max_tokens, ...]
        ttnn_data: TTNN data, same shape as torch_data
        expert_token_counts: Token counts per expert, shape [num_dispatch_groups, dispatch_group_size, experts_per_chip]
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

                torch_slot = torch_data[r, dst_chip_id, expert_id, :count]
                ttnn_slot = ttnn_data[r, dst_chip_id, expert_id, :count]

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
    expert_token_counts: torch.Tensor,
    expert_dispatch_table: torch.Tensor,
    num_dispatch_groups: int,
    dispatch_group_size: int,
    experts_per_chip: int,
    verbose: bool = True,
) -> ValidationResult:
    """
    Validate dispatch metadata against torch reference.

    Handles:
    - Linearized mesh coord conversion (field 0)
    - Direct comparison of fields 1-3
    - Weight bfloat16 bit reinterpretation (field 4)

    Args:
        torch_metadata: Reference metadata
        ttnn_metadata: TTNN metadata
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

                # Compare fields 1-3 directly
                out = ttnn_metadata[r, dst_chip_id, expert_id, :count, 1:4]
                ref = torch_metadata[r, dst_chip_id, expert_id, :count, 1:4]

                # Both Torch and TTNN now embed linearized mesh coord in field 0
                out_linearized_mesh_coord = ttnn_metadata[r, dst_chip_id, expert_id, :count, 0]
                ref_linearized_mesh_coord = torch_metadata[r, dst_chip_id, expert_id, :count, 0]

                # Compare weights (metadata[4]):
                # TTNN stores raw bfloat16 bits as uint16 in int32 - convert to bfloat16
                out_weight_bf16 = (
                    ttnn_metadata[r, dst_chip_id, expert_id, :count, 4].to(torch.int16).view(torch.bfloat16)
                )
                # Torch stores bfloat16 value directly
                ref_weight_bf16 = (
                    torch_metadata[r, dst_chip_id, expert_id, :count, 4].to(torch.int16).view(torch.bfloat16)
                )

                metadata_match = torch.allclose(out, ref, atol=1e-6)
                coord_match = torch.allclose(
                    out_linearized_mesh_coord.float(), ref_linearized_mesh_coord.float(), atol=1e-6
                )

                gate_weight_match, gate_pcc = comp_pcc(ref_weight_bf16.float(), out_weight_bf16.float(), pcc=0.99)

                if metadata_match and coord_match and gate_weight_match:
                    matches += 1
                    logger.debug(f"✅ {r} Metadata {dst_chip_id=} {expert_id=} {count=}")
                else:
                    error_detail = f"metadata={metadata_match}, coord={coord_match}, weight={gate_weight_match}"
                    logger.error(f"❌ {r} Metadata {dst_chip_id=} {expert_id=} {count=} ({error_detail})")
                    mismatches.append((r, dst_chip_id, expert_id, error_detail))

                    if verbose:
                        for slot in range(count):
                            torch_data = torch_metadata[r, dst_chip_id, expert_id, slot, :4]
                            kernel_data = ttnn_metadata[r, dst_chip_id, expert_id, slot, :4]
                            slot_data_match = torch.allclose(torch_data, kernel_data, atol=1e-6)
                            if not slot_data_match:
                                logger.error(
                                    f"    Slot {slot}: Metadata mismatch at chip={dst_chip_id}, expert={expert_id}: "
                                    f"ref_coord={ref_linearized_mesh_coord[slot].item()}, "
                                    f"out_coord={out_linearized_mesh_coord[slot].item()}, "
                                    f"torch={torch_data.tolist()}, kernel={kernel_data.tolist()}"
                                )
                            if not gate_weight_match:
                                logger.error(
                                    f"    Slot {slot}: Weight mismatch gate pcc={gate_pcc:.3f}: "
                                    f"ref={ref_weight_bf16[slot].item()}, out={out_weight_bf16[slot].item()}"
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
