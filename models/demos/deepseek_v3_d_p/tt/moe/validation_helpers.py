"""Validation utilities for MoE dispatch/combine tests."""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import torch
from loguru import logger


@dataclass
class ValidationResult:
    """Result of tensor comparison."""

    passed: bool
    matches: int
    total: int
    mismatches: List[Tuple] = field(default_factory=list)
    name: str = "validation"

    def log_summary(self):
        """Log match statistics."""
        pct = 100.0 * self.matches / self.total if self.total > 0 else 0.0
        status = "✅" if self.passed else "❌"
        logger.info(f"{status} {self.name}: {self.matches}/{self.total} ({pct:.2f}%)")

    def log_mismatches(self, limit: int = 10):
        """Log first N mismatches."""
        if not self.mismatches:
            return

        logger.warning(f"Found {len(self.mismatches)} mismatches in {self.name}. Showing first {limit}:")
        for i, mismatch in enumerate(self.mismatches[:limit]):
            if len(mismatch) == 5:
                # Combine-style: (ep_rank, chip_id, token_id, topk_idx, max_diff)
                ep_rank, chip_id, token_id, topk_idx, max_diff = mismatch
                logger.error(
                    f"  [{i}] ep_rank={ep_rank}, chip={chip_id}, token={token_id}, topk={topk_idx}: "
                    f"max_diff={max_diff:.6f}"
                )
            elif len(mismatch) == 4:
                # Dispatch-style: (ep_rank, chip_id, expert_id, error_detail)
                ep_rank, chip_id, expert_id, error_detail = mismatch
                logger.error(f"  [{i}] ep_rank={ep_rank}, chip={chip_id}, expert={expert_id}: {error_detail}")
            else:
                # Generic fallback
                logger.error(f"  [{i}] {mismatch}")

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
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> ValidationResult:
    """
    Validate combine output against torch reference (EP-rank aware).

    Args:
        torch_output: Reference output, shape [dispatch_group_size, seq_len_per_chip, num_experts_per_tok, hidden_dim]
        ttnn_output: TTNN output, shape [num_dispatch_groups, dispatch_group_size, seq_len_per_chip, num_experts_per_tok, hidden_dim]
        indices: Expert indices, shape [dispatch_group_size, seq_len_per_chip, num_experts_per_tok]
        num_dispatch_groups: Number of EP ranks
        num_routed_experts: Total number of routed experts
        atol: Absolute tolerance for comparison
        rtol: Relative tolerance for comparison

    Returns:
        ValidationResult with match statistics and mismatches
    """
    dispatch_group_size = torch_output.shape[0]
    seq_len_per_chip = torch_output.shape[1]
    num_experts_per_tok = torch_output.shape[2]
    experts_per_rank = num_routed_experts // num_dispatch_groups

    matches = 0
    total_slots = 0
    mismatches = []

    for chip_id in range(dispatch_group_size):
        for token_id in range(seq_len_per_chip):
            for topk_idx in range(num_experts_per_tok):
                total_slots += 1

                # Determine which EP rank processed this (chip, token, topk)
                expert_id = indices[chip_id, token_id, topk_idx].item()
                ep_rank = expert_id // experts_per_rank

                torch_data = torch_output[chip_id, token_id, topk_idx]
                ttnn_data = ttnn_output[ep_rank, chip_id, token_id, topk_idx]

                if torch.allclose(torch_data, ttnn_data, atol=atol, rtol=rtol):
                    matches += 1
                else:
                    max_diff = torch.max(torch.abs(torch_data.float() - ttnn_data.float())).item()
                    mismatches.append((ep_rank, chip_id, token_id, topk_idx, max_diff))

    passed = len(mismatches) == 0
    return ValidationResult(
        passed=passed,
        matches=matches,
        total=total_slots,
        mismatches=mismatches,
        name="combine output",
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
        input_tensor: Original input, shape [dispatch_group_size, seq_len_per_chip, hidden_dim]
        output_tensor: Round-trip output, shape [num_dispatch_groups, dispatch_group_size, seq_len_per_chip, num_experts_per_tok, hidden_dim]
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
    experts_per_rank = num_routed_experts // num_dispatch_groups

    matches = 0
    total_slots = 0
    mismatches = []

    for chip_id in range(dispatch_group_size):
        for token_id in range(seq_len_per_chip):
            for topk_idx in range(num_experts_per_tok):
                total_slots += 1

                # Determine which EP rank processed this (chip, token, topk)
                expert_id = indices[chip_id, token_id, topk_idx].item()
                ep_rank = expert_id // experts_per_rank

                # Input token
                x_data = input_tensor[chip_id, token_id]
                # Output from the EP rank that processed this token
                y_data = output_tensor[ep_rank, chip_id, token_id, topk_idx]

                if torch.allclose(x_data, y_data, atol=atol, rtol=rtol):
                    matches += 1
                else:
                    max_diff = torch.max(torch.abs(x_data.float() - y_data.float())).item()
                    mismatches.append((ep_rank, chip_id, token_id, topk_idx, max_diff))

    passed = len(mismatches) == 0
    return ValidationResult(
        passed=passed,
        matches=matches,
        total=total_slots,
        mismatches=mismatches,
        name="roundtrip",
    )


def log_combine_mismatch_details(
    mismatches: List[Tuple],
    torch_output: torch.Tensor,
    ttnn_output: torch.Tensor,
    limit: int = 10,
):
    """Log detailed mismatch information for combine validation."""
    for i, (ep_rank, chip_id, token_id, topk_idx, max_diff) in enumerate(mismatches[:limit]):
        torch_sample = torch_output[chip_id, token_id, topk_idx, :5]
        ttnn_sample = ttnn_output[ep_rank, chip_id, token_id, topk_idx, :5]
        logger.error(
            f"  [{i}] Mismatch at ep_rank={ep_rank}, chip={chip_id}, token={token_id}, topk={topk_idx}: "
            f"max_diff={max_diff:.6f}"
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
    logger.info("\nPer-chip statistics:")
    for chip_id in range(dispatch_group_size):
        chip_mismatches = [m for m in mismatches if m[1] == chip_id]
        chip_total = seq_len_per_chip * num_experts_per_tok
        chip_matches = chip_total - len(chip_mismatches)
        logger.info(f"  Chip {chip_id}: {chip_matches}/{chip_total} matches ({100.0*chip_matches/chip_total:.2f}%)")


# Type for dispatch validation comparators
# Returns (match: bool, error_detail: Optional[str])
DispatchComparator = Callable[[torch.Tensor, torch.Tensor, int, int, int], Tuple[bool, Optional[str]]]


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

    Iterates over (ep_rank, chip, expert) slots, compares using compare_fn.
    Skips experts not present in EP rank (dispatch_table == -1).

    Args:
        torch_data: Reference data, shape [num_dispatch_groups, dispatch_group_size, experts_per_chip, max_tokens, ...]
        ttnn_data: TTNN data, same shape as torch_data
        expert_token_counts: Token counts per expert, shape [num_dispatch_groups, dispatch_group_size, experts_per_chip]
        expert_dispatch_table: Expert to chip mapping, shape [num_dispatch_groups, num_routed_experts]
        num_dispatch_groups: Number of EP ranks
        dispatch_group_size: Number of chips in dispatch group
        experts_per_chip: Number of experts per chip
        compare_fn: Comparison function (torch_slot, ttnn_slot, ep_rank, chip_id, expert_id) -> (match, error_detail)
        name: Name for logging
        verbose: Whether to log detailed mismatch info

    Returns:
        ValidationResult with match statistics
    """
    matches = 0
    total_slots = 0
    mismatches = []

    for r in range(num_dispatch_groups):
        for dst_chip_id in range(dispatch_group_size):
            for expert_id in range(experts_per_chip):
                # Compute global expert ID and check if it's present in this EP rank
                global_expert_id = dst_chip_id * experts_per_chip + expert_id
                if expert_dispatch_table[r, global_expert_id].item() == -1:
                    # Expert not present in this EP rank, skip comparison
                    logger.info(
                        f"⏭️ {r} {name} {dst_chip_id=} {expert_id=} (expert {global_expert_id} not in EP rank {r})"
                    )
                    continue

                total_slots += 1
                count = expert_token_counts[r, dst_chip_id, expert_id].item()

                torch_slot = torch_data[r, dst_chip_id, expert_id, :count]
                ttnn_slot = ttnn_data[r, dst_chip_id, expert_id, :count]

                match, error_detail = compare_fn(torch_slot, ttnn_slot, r, dst_chip_id, expert_id)

                if match:
                    matches += 1
                    logger.info(f"✅ {r} {name} {dst_chip_id=} {expert_id=} {count=}")
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
    matches = 0
    total_slots = 0
    mismatches = []

    for r in range(num_dispatch_groups):
        for dst_chip_id in range(dispatch_group_size):
            for expert_id in range(experts_per_chip):
                # Compute global expert ID and check if it's present in this EP rank
                global_expert_id = dst_chip_id * experts_per_chip + expert_id
                if expert_dispatch_table[r, global_expert_id].item() == -1:
                    logger.info(
                        f"⏭️ {r} Metadata {dst_chip_id=} {expert_id=} (expert {global_expert_id} not in EP rank {r})"
                    )
                    continue

                total_slots += 1
                count = expert_token_counts[r, dst_chip_id, expert_id].item()

                # Compare fields 1-3 directly
                out = ttnn_metadata[r, dst_chip_id, expert_id, :count, 1:4]
                ref = torch_metadata[r, dst_chip_id, expert_id, :count, 1:4]

                # Torch computes "logical sender chip id"
                # while TTNN embeds real linearized mesh coord
                out_linearized_mesh_coord = ttnn_metadata[r, dst_chip_id, expert_id, :count, 0]
                ref_linearized_mesh_coord = (
                    r + torch_metadata[r, dst_chip_id, expert_id, :count, 0] * num_dispatch_groups
                )

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
                weight_match = torch.allclose(out_weight_bf16, ref_weight_bf16, atol=1e-3)

                if metadata_match and coord_match and weight_match:
                    matches += 1
                    logger.info(f"✅ {r} Metadata {dst_chip_id=} {expert_id=} {count=}")
                else:
                    error_detail = f"metadata={metadata_match}, coord={coord_match}, weight={weight_match}"
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
                            if not weight_match:
                                logger.error(
                                    f"    Slot {slot}: Weight mismatch: "
                                    f"ref={ref_weight_bf16[slot].item()}, out={out_weight_bf16[slot].item()}"
                                )

    passed = len(mismatches) == 0
    return ValidationResult(
        passed=passed,
        matches=matches,
        total=total_slots,
        mismatches=mismatches,
        name="metadata",
    )
